"""
QuantKnight Dashboard Generator.

Reads portfolio.json and data/trades/*.json, embeds all data inline,
and writes a self-contained dark-theme HTML dashboard to
data/reports/index.html.

Deployed to GitHub Pages after each EOD run.
URL: https://vedantiitkgp.github.io/quantknight
"""
import json
import os
from datetime import date, datetime, timedelta
from typing import Dict, List

_TRADES_DIR  = "data/trades"
_REPORTS_DIR = "data/reports"


def _load_equity_curve(days: int = 90) -> List[Dict]:
    """Build equity curve from the last `days` of trade JSON files."""
    curve = []
    today = date.today()
    for delta in range(days, -1, -1):
        d     = today - timedelta(days=delta)
        path  = f"{_TRADES_DIR}/{d}.json"
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                t = json.load(f)
            curve.append({
                "date":           str(d),
                "cumulative_pnl": t.get("cumulative_pnl", 0),
                "daily_pnl":      t.get("daily_pnl", 0),
                "realized_pnl":   t.get("realized_pnl", 0),
            })
        except Exception:
            pass
    return curve


def _load_recent_trades(days: int = 30) -> List[Dict]:
    """Collect all entries + exits from the last `days` of trade files."""
    recent = []
    today  = date.today()
    for delta in range(days, -1, -1):
        d    = today - timedelta(days=delta)
        path = f"{_TRADES_DIR}/{d}.json"
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                t = json.load(f)
            for entry in t.get("entries", []):
                recent.append({**entry, "_type": "entry", "_date": str(d)})
            for ex in t.get("exits", []):
                recent.append({**ex, "_type": "exit", "_date": str(d)})
        except Exception:
            pass
    return recent


def _win_rate(recent_trades: List[Dict]) -> float:
    exits = [r for r in recent_trades if r.get("_type") == "exit"]
    if not exits:
        return 0.0
    wins = [e for e in exits if e.get("pnl_dollars", 0) >= 0]
    return round(len(wins) / len(exits) * 100, 1)


def generate_dashboard(portfolio: Dict, today_trades: Dict) -> str:
    """
    Generate self-contained HTML dashboard and write to data/reports/index.html.
    Returns the output path.
    """
    os.makedirs(_REPORTS_DIR, exist_ok=True)

    equity_curve   = _load_equity_curve(90)
    recent_trades  = _load_recent_trades(30)
    win_rate       = _win_rate(recent_trades)

    data_json = json.dumps({
        "portfolio":     portfolio,
        "today_trades":  today_trades,
        "equity_curve":  equity_curve,
        "recent_trades": recent_trades,
        "win_rate":      win_rate,
        "generated_at":  datetime.utcnow().isoformat() + "Z",
    }, default=str)

    html = _build_html(data_json)
    path = f"{_REPORTS_DIR}/index.html"
    with open(path, "w") as f:
        f.write(html)
    return path


def _build_html(data_json: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>QuantKnight Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/marked@9/marked.min.js"></script>
<style>
  /* ── Markdown content inside thesis cards ── */
  .md h1, .md h2, .md h3 {{ font-size: 0.85rem; font-weight: 700; margin: 10px 0 4px; color: var(--text); }}
  .md h4 {{ font-size: 0.8rem; font-weight: 600; margin: 8px 0 2px; color: var(--text); }}
  /* Section breaks inside thesis cards — separator above every h3 except the first */
  .thesis h3 {{ margin-top: 20px; padding-top: 14px; border-top: 1px solid rgba(255,255,255,0.07); }}
  .thesis h3:first-child {{ border-top: none; margin-top: 6px; padding-top: 0; }}
  .md p {{ margin: 4px 0 8px; }}
  .md ul, .md ol {{ margin: 4px 0 8px; padding-left: 20px; }}
  .md li {{ margin: 2px 0; }}
  .md strong {{ color: var(--text); }}
  .md em {{ opacity: 0.85; }}
  .md hr {{ border: none; border-top: 1px solid var(--border); margin: 8px 0; }}
  .md table {{ width: 100%; border-collapse: collapse; font-size: 0.78rem; margin: 6px 0; }}
  .md th {{ background: rgba(255,255,255,0.04); padding: 4px 8px; text-align: left; border: 1px solid var(--border); }}
  .md td {{ padding: 4px 8px; border: 1px solid var(--border); }}
  .md code {{ background: rgba(255,255,255,0.06); padding: 1px 4px; border-radius: 3px; font-family: monospace; font-size: 0.85em; }}
  :root {{
    --bg:      #0d1117;
    --surface: #161b22;
    --border:  #30363d;
    --text:    #e6edf3;
    --muted:   #7d8590;
    --green:   #3fb950;
    --red:     #f85149;
    --yellow:  #d29922;
    --blue:    #58a6ff;
    --purple:  #bc8cff;
    --accent:  #1f6feb;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; min-height: 100vh; }}

  /* ── Header ── */
  header {{ background: var(--surface); border-bottom: 1px solid var(--border); padding: 16px 24px; display: flex; align-items: center; justify-content: space-between; }}
  .logo {{ font-size: 1.3rem; font-weight: 700; letter-spacing: -0.5px; }}
  .logo span {{ color: var(--blue); }}
  .last-updated {{ color: var(--muted); font-size: 0.8rem; }}

  /* ── Layout ── */
  main {{ max-width: 1200px; margin: 0 auto; padding: 24px 16px; display: flex; flex-direction: column; gap: 24px; }}

  /* ── Cards ── */
  .cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }}
  @media (max-width: 768px) {{ .cards {{ grid-template-columns: repeat(2, 1fr); }} }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 20px; }}
  .card-label {{ color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }}
  .card-value {{ font-size: 1.6rem; font-weight: 700; font-variant-numeric: tabular-nums; }}
  .card-sub {{ color: var(--muted); font-size: 0.8rem; margin-top: 4px; }}
  .positive {{ color: var(--green); }}
  .negative {{ color: var(--red); }}
  .neutral  {{ color: var(--text); }}

  /* Win rate bar */
  .win-bar {{ background: var(--border); border-radius: 4px; height: 6px; margin-top: 10px; overflow: hidden; }}
  .win-bar-fill {{ height: 100%; border-radius: 4px; background: var(--green); transition: width 0.6s ease; }}

  /* ── Chart ── */
  .chart-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 20px; }}
  .section-title {{ font-size: 0.9rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 16px; }}
  .chart-wrap {{ position: relative; height: 200px; }}

  /* ── Table ── */
  .table-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }}
  .table-header {{ padding: 16px 20px; border-bottom: 1px solid var(--border); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.875rem; }}
  th {{ padding: 10px 16px; text-align: left; color: var(--muted); font-weight: 500; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid var(--border); white-space: nowrap; }}
  td {{ padding: 12px 16px; border-bottom: 1px solid var(--border); vertical-align: top; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: rgba(255,255,255,0.02); }}
  .empty-state {{ padding: 32px; text-align: center; color: var(--muted); }}

  /* ── Badges ── */
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }}
  .badge-approved {{ background: rgba(63,185,80,0.15); color: var(--green); }}
  .badge-watch    {{ background: rgba(210,153,34,0.15); color: var(--yellow); }}
  .badge-rejected {{ background: rgba(248,81,73,0.15); color: var(--red); }}
  .badge-long  {{ background: rgba(88,166,255,0.15); color: var(--blue); }}
  .badge-short {{ background: rgba(248,81,73,0.15); color: var(--red); }}

  /* ── Tabs ── */
  .tabs {{ display: flex; gap: 0; border-bottom: 1px solid var(--border); padding: 0 20px; }}
  .tab {{ padding: 12px 16px; font-size: 0.85rem; cursor: pointer; color: var(--muted); border-bottom: 2px solid transparent; margin-bottom: -1px; transition: all 0.15s; user-select: none; }}
  .tab.active {{ color: var(--text); border-bottom-color: var(--blue); }}
  .tab:hover:not(.active) {{ color: var(--text); }}
  .tab-content {{ display: none; }}
  .tab-content.active {{ display: block; }}

  /* ── Reason / thesis text ── */
  .reason {{ color: var(--muted); font-size: 0.8rem; margin-top: 4px; line-height: 1.4; }}
  .mono {{ font-family: 'SF Mono', SFMono-Regular, Consolas, monospace; }}
  .thesis-row td {{ padding: 0 16px 12px 16px !important; border-bottom: 1px solid var(--border); }}
  .thesis {{ background: rgba(31,111,235,0.05); border-left: 3px solid var(--accent); border-radius: 0 4px 4px 0; padding: 10px 14px; margin-bottom: 6px; font-size: 0.82rem; line-height: 1.55; }}
  .thesis-bull {{ border-left-color: var(--green); }}
  .thesis-bear {{ border-left-color: var(--red); }}
  .thesis-memo {{ border-left-color: var(--purple); }}
  .thesis-label {{ font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; opacity: 0.7; }}
  details summary {{ cursor: pointer; color: var(--blue); font-size: 0.78rem; font-weight: 600; padding: 4px 0; user-select: none; }}
  details summary:hover {{ opacity: 0.8; }}
  details[open] summary {{ margin-bottom: 8px; }}
  .thesis-summary-bull {{ color: var(--green) !important; }}
  .thesis-summary-bear {{ color: var(--red) !important; }}
  .thesis-summary-memo {{ color: var(--purple) !important; }}

  /* ── News Articles ── */
  .news-list {{ display: flex; flex-direction: column; gap: 6px; margin-top: 8px; }}
  .news-item {{ display: flex; gap: 10px; align-items: flex-start; padding: 8px 12px; background: rgba(255,255,255,0.02); border-radius: 6px; border: 1px solid var(--border); }}
  .news-sentiment {{ font-size: 0.75rem; font-weight: 700; padding: 2px 6px; border-radius: 10px; white-space: nowrap; flex-shrink: 0; }}
  .news-positive {{ background: rgba(63,185,80,0.15); color: var(--green); }}
  .news-negative {{ background: rgba(248,81,73,0.15); color: var(--red); }}
  .news-neutral  {{ background: rgba(125,133,144,0.15); color: var(--muted); }}
  .news-body {{ flex: 1; min-width: 0; }}
  .news-headline {{ font-size: 0.82rem; font-weight: 500; line-height: 1.35; }}
  .news-meta {{ font-size: 0.72rem; color: var(--muted); margin-top: 2px; }}
  .news-summary {{ font-size: 0.78rem; color: var(--muted); margin-top: 4px; line-height: 1.4; }}
</style>
</head>
<body>
<script>window.__DATA__ = {data_json};</script>

<header>
  <div class="logo">♞ Quant<span>Knight</span></div>
  <div class="last-updated" id="last-updated">—</div>
</header>

<main>
  <!-- ── Metric Cards ── -->
  <div class="cards">
    <div class="card">
      <div class="card-label">Total Equity</div>
      <div class="card-value" id="equity">—</div>
      <div class="card-sub" id="equity-sub">Starting $150,000</div>
    </div>
    <div class="card">
      <div class="card-label">Cash Available</div>
      <div class="card-value" id="cash">—</div>
      <div class="card-sub" id="open-pos">— open positions</div>
    </div>
    <div class="card">
      <div class="card-label">Cumulative P&amp;L</div>
      <div class="card-value" id="cum-pnl">—</div>
      <div class="card-sub" id="daily-pnl">Today: —</div>
    </div>
    <div class="card">
      <div class="card-label">Win Rate (30d)</div>
      <div class="card-value" id="win-rate">—</div>
      <div class="win-bar"><div class="win-bar-fill" id="win-bar-fill" style="width:0%"></div></div>
    </div>
  </div>

  <!-- ── Equity Curve ── -->
  <div class="chart-card">
    <div class="section-title">Equity Curve — 90 Days</div>
    <div class="chart-wrap">
      <canvas id="equity-chart"></canvas>
    </div>
  </div>

  <!-- ── Open Positions ── -->
  <div class="table-card">
    <div class="table-header">
      <div class="section-title" style="margin:0">Open Positions</div>
    </div>
    <div id="positions-container">
      <div class="empty-state">No open positions</div>
    </div>
  </div>

  <!-- ── Trade Log ── -->
  <div class="table-card">
    <div class="tabs">
      <div class="tab active" onclick="switchTab('entries')">Today's Entries</div>
      <div class="tab" onclick="switchTab('exits')">Today's Exits</div>
      <div class="tab" onclick="switchTab('recent')">Last 30 Days</div>
    </div>
    <div id="tab-entries" class="tab-content active">
      <div id="entries-container"><div class="empty-state">No entries today</div></div>
    </div>
    <div id="tab-exits" class="tab-content">
      <div id="exits-container"><div class="empty-state">No exits today</div></div>
    </div>
    <div id="tab-recent" class="tab-content">
      <div id="recent-container"><div class="empty-state">No recent trades</div></div>
    </div>
  </div>
</main>

<script>
const D = window.__DATA__;
const port   = D.portfolio;
const today  = D.today_trades;
const curve  = D.equity_curve;
const recent = D.recent_trades;

// ── Markdown + date helpers ────────────────────────────────────────────────────
function preprocess(s) {{
  // ══════════ SECTION HEADER ══════════  →  ### SECTION HEADER
  return s
    .replace(/^[═─]{{3,}}\\s*(.+?)\\s*[═─]{{3,}}\\s*$/gm, '### $1')
    .replace(/^[═─]{{3,}}\\s*$/gm, '---');
}}
function md(text) {{
  if (!text) return '';
  const s = preprocess(String(text));
  try {{
    if (typeof marked !== 'undefined' && typeof marked.parse === 'function') {{
      return marked.parse(s);
    }}
  }} catch(e) {{}}
  // Inline fallback: convert the most common LLM Markdown patterns
  return s
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/^---$/gm,'<hr>')
    .replace(/^### (.+)$/gm,'<h3>$1</h3>')
    .replace(/^## (.+)$/gm,'<h2>$1</h2>')
    .replace(/^# (.+)$/gm,'<h2>$1</h2>')
    .replace(/\\*\\*([^*\\n]+)\\*\\*/g,'<strong>$1</strong>')
    .replace(/\\*([^*\\n]+)\\*/g,'<em>$1</em>')
    .replace(/^[-*•] (.+)$/gm,'<li>$1</li>')
    .replace(/(<\\/li>\\n<li>)/g,'$1')
    .replace(/(<li>[\\s\\S]*?<\\/li>)/g,'<ul>$1</ul>')
    .replace(/\\n{{2,}}/g,'</p><p>')
    .replace(/\\n/g,'<br>');
}}
function fmtDate(d) {{
  if (!d) return '';
  const s = String(d);
  // Convert Unix timestamp (10 digits) to YYYY-MM-DD
  if (/^\\d{{10}}$/.test(s)) {{
    return new Date(parseInt(s) * 1000).toISOString().slice(0, 10);
  }}
  return s.slice(0, 10);
}}

// ── Formatters ────────────────────────────────────────────────────────────────
const fmt$ = v => v == null ? '—' : '$' + Math.abs(v).toLocaleString('en-US', {{minimumFractionDigits:0, maximumFractionDigits:0}});
const fmtPnl = v => {{
  if (v == null) return '—';
  const s = v >= 0 ? '+' : '';
  return s + '$' + Math.abs(v).toLocaleString('en-US', {{minimumFractionDigits:0, maximumFractionDigits:0}});
}};
const fmtPct = v => v == null ? '' : ` (${{v >= 0 ? '+' : ''}}${{Number(v).toFixed(1)}}%)`;
const colorClass = v => v > 0 ? 'positive' : v < 0 ? 'negative' : 'neutral';
const badge = (verdict) => {{
  if (!verdict) return '';
  const cls = verdict.toLowerCase();
  return `<span class="badge badge-${{cls}}">${{verdict}}</span>`;
}};
const dirBadge = dir => `<span class="badge badge-${{dir?.toLowerCase()}}">${{dir}}</span>`;

// ── Header cards ──────────────────────────────────────────────────────────────
document.getElementById('last-updated').textContent =
  'Updated ' + new Date(D.generated_at).toLocaleString('en-US', {{timeZone:'America/New_York', dateStyle:'medium', timeStyle:'short'}}) + ' ET';

const equity = port.total_equity || 0;
const pnl    = port.cumulative_pnl || 0;
document.getElementById('equity').textContent = fmt$(equity);
document.getElementById('equity').className   = 'card-value';

document.getElementById('cash').textContent   = fmt$(port.cash);
document.getElementById('open-pos').textContent = `${{(port.positions||[]).length}} open position${{(port.positions||[]).length !== 1 ? 's' : ''}}`;

const pnlEl = document.getElementById('cum-pnl');
pnlEl.textContent  = fmtPnl(pnl);
pnlEl.className    = 'card-value ' + colorClass(pnl);

const dailyR = today.realized_pnl || 0;
const dailyU = today.unrealized_pnl || 0;
document.getElementById('daily-pnl').textContent = `Today: ${{fmtPnl(dailyR + dailyU)}}`;

const wr = D.win_rate || 0;
document.getElementById('win-rate').textContent = wr.toFixed(0) + '%';
document.getElementById('win-rate').className   = 'card-value ' + colorClass(wr - 50);
document.getElementById('win-bar-fill').style.width = Math.min(wr, 100) + '%';

// ── Equity Curve Chart ────────────────────────────────────────────────────────
if (curve && curve.length > 0) {{
  const labels = curve.map(c => c.date);
  const baseline = 150000;
  const equities = curve.map(c => baseline + (c.cumulative_pnl || 0));

  new Chart(document.getElementById('equity-chart'), {{
    type: 'line',
    data: {{
      labels,
      datasets: [{{
        label: 'Portfolio Equity',
        data: equities,
        borderColor: '#58a6ff',
        backgroundColor: 'rgba(88,166,255,0.08)',
        borderWidth: 2,
        pointRadius: 0,
        pointHoverRadius: 4,
        fill: true,
        tension: 0.3,
      }}, {{
        label: 'Baseline $150k',
        data: labels.map(() => baseline),
        borderColor: 'rgba(125,133,144,0.4)',
        borderWidth: 1,
        borderDash: [4, 4],
        pointRadius: 0,
        fill: false,
      }}]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          backgroundColor: '#161b22',
          borderColor: '#30363d',
          borderWidth: 1,
          callbacks: {{
            label: ctx => ' ' + fmt$(ctx.raw),
          }}
        }}
      }},
      scales: {{
        x: {{
          grid: {{ color: 'rgba(48,54,61,0.5)' }},
          ticks: {{ color: '#7d8590', maxTicksLimit: 6, font: {{ size: 11 }} }}
        }},
        y: {{
          grid: {{ color: 'rgba(48,54,61,0.5)' }},
          ticks: {{ color: '#7d8590', font: {{ size: 11 }}, callback: v => fmt$(v) }}
        }}
      }}
    }}
  }});
}} else {{
  document.getElementById('equity-chart').parentElement.innerHTML =
    '<div class="empty-state">No historical data yet — runs after first EOD</div>';
}}

// ── Open Positions Table ──────────────────────────────────────────────────────
const positions = port.positions || [];
const posEl = document.getElementById('positions-container');
if (positions.length > 0) {{
  posEl.innerHTML = `
  <table>
    <thead><tr>
      <th>Symbol</th><th>Direction</th><th>Shares</th>
      <th>Entry</th><th>Stop</th><th>Target</th><th>Score</th><th>Entered</th>
    </tr></thead>
    <tbody>
    ${{positions.map(p => `
      <tr>
        <td><strong>${{p.symbol}}</strong></td>
        <td>${{dirBadge(p.direction)}}</td>
        <td class="mono">${{p.shares}}</td>
        <td class="mono">$${{Number(p.entry_price).toFixed(2)}}</td>
        <td class="mono">${{p.stop_loss ? '$' + Number(p.stop_loss).toFixed(2) : '—'}}</td>
        <td class="mono">${{p.target    ? '$' + Number(p.target).toFixed(2)    : '—'}}</td>
        <td>${{p.composite_score ? Number(p.composite_score).toFixed(1) : '—'}}</td>
        <td>${{p.entry_date || '—'}}</td>
      </tr>
      <tr class="thesis-row"><td colspan="8">
        <div class="reason">${{p.reason || ''}}</div>
        ${{p.bull_thesis ? `<details style="margin-top:8px"><summary class="thesis-summary-bull">Bull case ▸</summary><div class="thesis thesis-bull md">${{md(p.bull_thesis)}}</div></details>` : ''}}
        ${{p.bear_risks  ? `<details style="margin-top:6px"><summary class="thesis-summary-bear">Bear risks ▸</summary><div class="thesis thesis-bear md">${{md(p.bear_risks)}}</div></details>` : ''}}
        ${{p.full_memo   ? `<details style="margin-top:6px"><summary class="thesis-summary-memo">Risk Manager Verdict ▸</summary><div class="thesis thesis-memo md">${{md(p.full_memo)}}</div></details>` : ''}}
        ${{(p.full_news && p.full_news.length) ? `
        <details style="margin-top:8px">
          <summary>News articles at entry (${{p.full_news.length}}) ▸</summary>
          <div class="news-list" style="margin-top:8px">
            ${{p.full_news.map(n => {{
              const sc = n.sentiment || 0;
              const cls = sc > 0.05 ? 'news-positive' : sc < -0.05 ? 'news-negative' : 'news-neutral';
              const lbl = sc > 0.05 ? '▲ ' + sc.toFixed(2) : sc < -0.05 ? '▼ ' + Math.abs(sc).toFixed(2) : '● ' + sc.toFixed(2);
              return `<div class="news-item">
                <span class="news-sentiment ${{cls}}">${{lbl}}</span>
                <div class="news-body">
                  <div class="news-headline">${{n.headline}}</div>
                  <div class="news-meta">${{n.source || ''}}${{n.published ? ' · ' + fmtDate(n.published) : ''}}</div>
                  ${{n.summary ? '<div class="news-summary">' + n.summary + '</div>' : ''}}
                </div>
              </div>`;
            }}).join('')}}
          </div>
        </details>` : ''}}
      </td></tr>
    `).join('')}}
    </tbody>
  </table>`;
}}

// ── Trade Log helpers ─────────────────────────────────────────────────────────
function renderEntries(entries, container) {{
  if (!entries || entries.length === 0) {{
    container.innerHTML = '<div class="empty-state">No entries</div>';
    return;
  }}
  container.innerHTML = `
  <table>
    <thead><tr>
      <th>Date</th><th>Symbol</th><th>Verdict</th><th>Shares</th><th>Entry $</th><th>Cost</th><th>Stop</th><th>Target</th>
    </tr></thead>
    <tbody>
    ${{entries.map(e => `
      <tr>
        <td style="color:var(--muted);font-size:0.8rem">${{e._date || e.entry_date || '—'}}</td>
        <td><strong>${{e.symbol}}</strong> ${{dirBadge(e.direction)}}</td>
        <td>${{badge(e.verdict)}}</td>
        <td class="mono">${{e.shares}}</td>
        <td class="mono">$${{Number(e.entry_price).toFixed(2)}}</td>
        <td class="mono">$${{e.cost ? Number(e.cost).toLocaleString('en-US', {{maximumFractionDigits:0}}) : '—'}}</td>
        <td class="mono">${{e.stop_loss ? '$' + Number(e.stop_loss).toFixed(2) : '—'}}</td>
        <td class="mono">${{e.target    ? '$' + Number(e.target).toFixed(2)    : '—'}}</td>
      </tr>
      ${{(e.reason || e.bull_thesis) ? `
      <tr class="thesis-row"><td colspan="8">
        <div class="reason">${{e.reason || ''}}</div>
        ${{e.bull_thesis ? `<details style="margin-top:8px"><summary class="thesis-summary-bull">Bull case ▸</summary><div class="thesis thesis-bull md">${{md(e.bull_thesis)}}</div></details>` : ''}}
        ${{e.bear_risks  ? `<details style="margin-top:6px"><summary class="thesis-summary-bear">Bear risks ▸</summary><div class="thesis thesis-bear md">${{md(e.bear_risks)}}</div></details>` : ''}}
        ${{e.full_memo   ? `<details style="margin-top:6px"><summary class="thesis-summary-memo">Risk Manager Verdict ▸</summary><div class="thesis thesis-memo md">${{md(e.full_memo)}}</div></details>` : ''}}
      </td></tr>` : ''}}
    `).join('')}}
    </tbody>
  </table>`;
}}

function renderExits(exits, container) {{
  if (!exits || exits.length === 0) {{
    container.innerHTML = '<div class="empty-state">No exits</div>';
    return;
  }}
  container.innerHTML = `
  <table>
    <thead><tr>
      <th>Symbol</th><th>Dir</th><th>Shares</th><th>Entry $</th><th>Exit $</th><th>P&amp;L $</th><th>P&amp;L %</th><th>Reason</th>
    </tr></thead>
    <tbody>
    ${{exits.map(e => {{
      const pnl = e.pnl_dollars || 0;
      const cls = colorClass(pnl);
      return `
        <tr>
          <td><strong>${{e.symbol}}</strong></td>
          <td>${{dirBadge(e.direction)}}</td>
          <td class="mono">${{e.shares}}</td>
          <td class="mono">$${{Number(e.entry_price).toFixed(2)}}</td>
          <td class="mono">$${{Number(e.exit_price).toFixed(2)}}</td>
          <td class="mono ${{cls}}">${{fmtPnl(pnl)}}</td>
          <td class="mono ${{cls}}">${{fmtPct(e.pnl_pct)}}</td>
          <td style="color:var(--muted);font-size:0.8rem">${{e.reason || '—'}}</td>
        </tr>`;
    }}).join('')}}
    </tbody>
  </table>`;
}}

function renderRecentTrades(trades, container) {{
  if (!trades || trades.length === 0) {{
    container.innerHTML = '<div class="empty-state">No recent trades</div>';
    return;
  }}
  const exits   = trades.filter(t => t._type === 'exit').slice(0, 50);
  const entries = trades.filter(t => t._type === 'entry').slice(0, 50);
  if (exits.length > 0) {{
    // Show exits with P&L when available
    renderExits(exits, container);
  }} else if (entries.length > 0) {{
    // Fall back to showing entries if no exits yet
    renderEntries(entries, container);
  }} else {{
    container.innerHTML = '<div class="empty-state">No recent trades</div>';
  }}
}}

// ── Render trade tabs — fall back to recent data when today is empty ──────────
const todayEntries = today.entries || [];
const todayExits   = today.exits   || [];
const recentEntries = recent.filter(t => t._type === 'entry').slice(0, 40);
const recentExits   = recent.filter(t => t._type === 'exit').slice(0, 40);

// Entries tab
if (todayEntries.length > 0) {{
  renderEntries(todayEntries, document.getElementById('entries-container'));
}} else if (recentEntries.length > 0) {{
  document.querySelector('.tab[onclick*="entries"]').textContent = 'Entries (recent)';
  renderEntries(recentEntries, document.getElementById('entries-container'));
}} else {{
  document.getElementById('entries-container').innerHTML = '<div class="empty-state">No entries yet</div>';
}}

// Exits tab
if (todayExits.length > 0) {{
  renderExits(todayExits, document.getElementById('exits-container'));
}} else if (recentExits.length > 0) {{
  document.querySelector('.tab[onclick*="exits"]').textContent = 'Exits (recent)';
  renderExits(recentExits, document.getElementById('exits-container'));
}} else {{
  document.getElementById('exits-container').innerHTML = '<div class="empty-state">No exits yet</div>';
}}

renderRecentTrades(recent, document.getElementById('recent-container'));

// ── Tabs ──────────────────────────────────────────────────────────────────────
function switchTab(name) {{
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById('tab-' + name).classList.add('active');
}}
</script>
</body>
</html>"""
