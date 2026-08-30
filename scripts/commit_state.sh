#!/usr/bin/env bash
# commit_state.sh — commit portfolio state back to the repo after each run.
#
# Usage:  bash scripts/commit_state.sh [mode]
#   mode: morning | midmorning | midday | preclose | eod  (default: "update")
#
# Requires: GITHUB_TOKEN env var (automatically set by GitHub Actions)
#           git configured with user.name / user.email

set -euo pipefail

MODE="${1:-update}"
TODAY=$(date -u +%Y-%m-%d)
TIME_UTC=$(date -u +%H:%M)

# ── Git identity (bot account for Actions) ────────────────────────────────────
git config user.name  "quantknight-bot"
git config user.email "quantknight-bot@users.noreply.github.com"

# ── Stage data files (portfolio state, trades, reports, SQLite DB) ─────────────
git add -f data/portfolio.json        2>/dev/null || true
git add -f data/quant_engine.db       2>/dev/null || true
git add -f "data/trades/${TODAY}.json" 2>/dev/null || true
git add -f data/trades/               2>/dev/null || true
git add -f data/reports/              2>/dev/null || true

# ── Check if there is anything to commit ─────────────────────────────────────
if git diff --cached --quiet; then
    echo "No state changes to commit (mode: ${MODE})."
    exit 0
fi

# ── Commit ────────────────────────────────────────────────────────────────────
git commit -m "chore(state): ${MODE} run — ${TODAY} ${TIME_UTC} UTC

Automated portfolio state commit by QuantKnight intraday engine.
Mode: ${MODE} | Date: ${TODAY} | Time: ${TIME_UTC} UTC"

# ── Push (with rebase to handle concurrent pushes) ────────────────────────────
# GITHUB_TOKEN is injected by Actions; set the remote URL to use it.
REPO_URL="https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"

# Fetch and rebase in case another push (e.g. a code fix) landed while this
# run was executing.  Data files (portfolio.json, trades/, reports/) have no
# meaningful merge conflicts — rebase always wins cleanly.
git fetch "${REPO_URL}" main 2>/dev/null || true
git rebase "FETCH_HEAD" 2>/dev/null || git rebase --abort

git push "${REPO_URL}" HEAD:main

echo "State committed and pushed (mode: ${MODE})."
