"""
News sentiment engine using FinBERT.

FinBERT (ProsusAI/finbert) is a BERT model fine-tuned specifically on
financial communications (earnings calls, SEC filings, Bloomberg/Reuters).
It classifies text as positive / neutral / negative with probabilities.

Inference order — environment-aware:

  LOCAL (Mac, IS_CI=False):
    1. Local FinBERT via transformers (MPS-accelerated on Apple Silicon)
    2. HuggingFace Inference API (if HF_API_TOKEN is set and local fails)
    3. Keyword heuristic fallback

  CI (GitHub Actions, IS_CI=True):
    1. HuggingFace Inference API (full internet, no MPS)
    2. Local FinBERT on CPU (runner has ~7 GB RAM, ~440 MB model)
    3. Keyword heuristic fallback

Score range: -1.0 (strongly bearish) → +1.0 (strongly bullish)
"""
import time
import requests
from typing import Dict, List, Optional
from loguru import logger

from config.settings import HF_API_TOKEN, HF_FINBERT_URL, IS_CI
from src.data.news_client import NewsClient


# ── Local FinBERT pipeline (singleton, lazy-loaded) ───────────────────────────

_local_pipeline = None   # transformers pipeline instance


def _get_local_pipeline():
    """Load the local FinBERT pipeline on first call; reuse thereafter."""
    global _local_pipeline
    if _local_pipeline is not None:
        return _local_pipeline

    try:
        from transformers import pipeline as hf_pipeline
        import torch

        # Prefer MPS (Apple Silicon GPU) → CPU
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        # Forward our HF token so the hub doesn't warn about unauthenticated access
        import os
        if HF_API_TOKEN and not os.environ.get("HF_TOKEN"):
            os.environ["HF_TOKEN"] = HF_API_TOKEN

        logger.info(f"Loading FinBERT locally on {device.upper()} …")
        _local_pipeline = hf_pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=device,
            truncation=True,
            max_length=512,
            top_k=None,      # return all labels + probabilities
        )
        logger.info("FinBERT local pipeline ready.")
        return _local_pipeline
    except Exception as exc:
        logger.warning(f"Could not load local FinBERT: {exc}")
        return None


# ── Scoring helpers ───────────────────────────────────────────────────────────

def _label_to_score(labels: List[Dict]) -> float:
    """
    Convert FinBERT label probabilities to a scalar in [-1, +1].
    score = P(positive) - P(negative)
    """
    if not labels:
        return 0.0
    score = 0.0
    for lbl in labels:
        label = lbl.get("label", "").lower()
        prob  = float(lbl.get("score", 0))
        if label == "positive":
            score += prob
        elif label == "negative":
            score -= prob
    return round(max(-1.0, min(1.0, score)), 4)


def _local_finbert_score(texts: List[str], batch_size: int = 8) -> List[Dict]:
    """
    Run FinBERT locally via the transformers pipeline.
    Returns per-text result lists identical to what _api_finbert_score returns.
    """
    pipe = _get_local_pipeline()
    if pipe is None:
        return []

    all_results = []
    try:
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            outputs = pipe(batch)
            # Each output is a list of {label, score} dicts (top_k=None)
            for item in outputs:
                if isinstance(item, list):
                    all_results.append(item)
                elif isinstance(item, dict):
                    all_results.append([item])
    except Exception as exc:
        logger.warning(f"Local FinBERT inference failed: {exc}")
        return []

    return all_results


def _api_finbert_score(texts: List[str], batch_size: int = 10) -> List[Dict]:
    """
    Fall back to the HuggingFace Inference API when local model is unavailable.
    """
    if not HF_API_TOKEN:
        return []

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    all_results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        payload = {"inputs": batch, "options": {"wait_for_model": True}}
        try:
            resp = requests.post(HF_FINBERT_URL, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, list):
                        all_results.append(item)
                    elif isinstance(item, dict):
                        all_results.append([item])
        except requests.HTTPError as exc:
            if resp.status_code == 503:
                logger.info("FinBERT model loading on HF … waiting 15 s")
                time.sleep(15)
            else:
                logger.debug(f"HF API error: {exc}")
                break
        except Exception as exc:
            logger.debug(f"HF API unavailable: {exc}")
            break
        time.sleep(0.5)

    return all_results


def _keyword_fallback(text: str) -> float:
    """
    Simple keyword heuristic when FinBERT is completely unavailable.
    Not production-grade — only used as a last resort.
    """
    pos_words = [
        "beat", "exceeds", "record", "growth", "upgrade", "outperform",
        "strong", "raised guidance", "expansion", "acquisition", "partnership",
        "buy", "bullish", "rebound", "rally", "all-time high", "positive"
    ]
    neg_words = [
        "miss", "below", "disappoint", "downgrade", "underperform", "weak",
        "cut", "layoff", "loss", "decline", "bearish", "sell", "warning",
        "concern", "investigation", "lawsuit", "recall", "bankrupt"
    ]
    t = text.lower()
    score = sum(0.1 for w in pos_words if w in t) - sum(0.1 for w in neg_words if w in t)
    return round(max(-1.0, min(1.0, score)), 4)


# ── Main analyser class ───────────────────────────────────────────────────────

class SentimentAnalyser:
    def __init__(self):
        self.news_client = NewsClient()
        # Eagerly load the local model on first SentimentAnalyser instantiation
        _get_local_pipeline()

    def analyse(self, symbol: str) -> Dict:
        """
        Fetch news for `symbol` and return a sentiment summary dict:
          {
            "symbol":           "AAPL",
            "article_count":    25,
            "sentiment_score":  0.42,       # aggregate FinBERT score [-1, +1]
            "sentiment_label":  "Positive", # Positive / Neutral / Negative
            "top_headlines":    [...],       # 5 most impactful headlines
            "raw_articles":     [...],       # full list with per-article scores
          }
        """
        result = {
            "symbol":          symbol,
            "article_count":   0,
            "sentiment_score": 0.0,
            "sentiment_label": "Neutral",
            "top_headlines":   [],
            "raw_articles":    [],
        }

        articles = self.news_client.get_all_news(symbol, max_articles=35)
        if not articles:
            logger.debug(f"No news articles found for {symbol}")
            return result

        # Combine headline + summary for richer context (FinBERT max 512 tokens)
        texts = [
            f"{a['headline']}. {a['summary']}"[:512]
            for a in articles
            if a.get("headline")
        ]

        # ── Score using best available method ──────────────────────────────────
        # On CI: HF API → local CPU → keyword
        # On Mac: local MPS → HF API → keyword
        scored = False
        if texts:
            if IS_CI:
                # CI path: no MPS, full internet — HF API is fastest and free
                results = _api_finbert_score(texts)
                if results and len(results) == len(texts):
                    for i, art in enumerate(articles[:len(texts)]):
                        art["sentiment"] = _label_to_score(results[i])
                    scored = True
                    logger.debug(f"Sentiment for {symbol}: HF API used (CI)")

                if not scored:
                    results = _local_finbert_score(texts)
                    if results and len(results) == len(texts):
                        for i, art in enumerate(articles[:len(texts)]):
                            art["sentiment"] = _label_to_score(results[i])
                        scored = True
                        logger.debug(f"Sentiment for {symbol}: local CPU FinBERT used (CI)")
            else:
                # Local Mac path: MPS is fastest
                results = _local_finbert_score(texts)
                if results and len(results) == len(texts):
                    for i, art in enumerate(articles[:len(texts)]):
                        art["sentiment"] = _label_to_score(results[i])
                    scored = True
                    logger.debug(f"Sentiment for {symbol}: local MPS FinBERT used")

                if not scored:
                    results = _api_finbert_score(texts)
                    if results and len(results) == len(texts):
                        for i, art in enumerate(articles[:len(texts)]):
                            art["sentiment"] = _label_to_score(results[i])
                        scored = True
                        logger.debug(f"Sentiment for {symbol}: HF API used")

            # Final fallback: keyword heuristic
            if not scored:
                logger.debug(f"Sentiment for {symbol}: keyword fallback used")
                for art in articles:
                    art["sentiment"] = _keyword_fallback(
                        f"{art.get('headline','')} {art.get('summary','')}"
                    )

        # ── Aggregate ──────────────────────────────────────────────────────────
        scores = [a["sentiment"] for a in articles if a.get("sentiment") is not None]
        agg    = sum(scores) / len(scores) if scores else 0.0

        result["article_count"]   = len(articles)
        result["sentiment_score"] = round(agg, 4)
        result["raw_articles"]    = articles

        if agg > 0.15:
            result["sentiment_label"] = "Positive"
        elif agg < -0.15:
            result["sentiment_label"] = "Negative"
        else:
            result["sentiment_label"] = "Neutral"

        # Top 5 headlines sorted by absolute sentiment (most impactful first)
        sorted_arts = sorted(
            [a for a in articles if a.get("sentiment") is not None],
            key=lambda x: abs(x["sentiment"]),
            reverse=True,
        )
        result["top_headlines"] = [
            {"headline": a["headline"], "sentiment": a["sentiment"], "source": a["source"]}
            for a in sorted_arts[:5]
        ]

        logger.debug(
            f"Sentiment for {symbol}: score={agg:.3f} ({result['sentiment_label']}) "
            f"from {len(articles)} articles"
        )
        return result
