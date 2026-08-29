#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  setup_mac.sh  —  One-click Mac setup for the Stock Engine
#
#  What this does:
#    1. Checks for Python 3.11+ and Homebrew
#    2. Installs PostgreSQL 16 via Homebrew (if not present)
#    3. Creates the Python virtual environment
#    4. Installs all Python dependencies
#    5. Creates the .env file from .env.example (if not present)
#    6. Initialises the PostgreSQL database and schema
#    7. Optionally installs the launchd scheduler
#
#  Usage:
#    chmod +x setup_mac.sh
#    ./setup_mac.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║       Stock Engine — Mac Setup                       ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── 1. Python version check ───────────────────────────────────────────────────
PYTHON=$(command -v python3.12 || command -v python3.11 || command -v python3 || "")
if [[ -z "${PYTHON}" ]]; then
  echo "ERROR: Python 3.11+ is required. Install via: brew install python@3.12"
  exit 1
fi
PY_VERSION=$("${PYTHON}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅  Python ${PY_VERSION} found at ${PYTHON}"

# ── 2. Homebrew check ─────────────────────────────────────────────────────────
if ! command -v brew &>/dev/null; then
  echo "ERROR: Homebrew not found. Install via:"
  echo '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
  exit 1
fi
echo "✅  Homebrew found"

# ── 3. PostgreSQL ─────────────────────────────────────────────────────────────
if ! command -v psql &>/dev/null; then
  echo "Installing PostgreSQL 16 via Homebrew …"
  brew install postgresql@16
  brew services start postgresql@16
  echo 'export PATH="/opt/homebrew/opt/postgresql@16/bin:$PATH"' >> ~/.zprofile
  export PATH="/opt/homebrew/opt/postgresql@16/bin:$PATH"
else
  echo "✅  PostgreSQL found"
fi

# Ensure Postgres is running
if ! pg_isready -q 2>/dev/null; then
  echo "Starting PostgreSQL …"
  brew services start postgresql@16 2>/dev/null || pg_ctl -D /opt/homebrew/var/postgresql@16 start
  sleep 3
fi

# Create DB user + database
echo "Setting up database …"
createuser  --superuser engine_admin 2>/dev/null || true
createdb    --owner=engine_admin quant_engine 2>/dev/null || true
echo "✅  Database 'quant_engine' ready"

# ── 4. Virtual environment ────────────────────────────────────────────────────
VENV_DIR="${PROJECT_DIR}/.venv"
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating Python virtual environment …"
  "${PYTHON}" -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
echo "✅  Virtual environment: ${VENV_DIR}"

# ── 5. Install Python packages ────────────────────────────────────────────────
echo ""
echo "Installing Python dependencies (this may take 3–5 minutes) …"
pip install --upgrade pip --quiet
pip install -r "${PROJECT_DIR}/requirements.txt" --quiet
echo "✅  Dependencies installed"

# ── 6. .env file ─────────────────────────────────────────────────────────────
ENV_FILE="${PROJECT_DIR}/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
  cp "${PROJECT_DIR}/.env.example" "${ENV_FILE}"
  echo ""
  echo "⚠️   Created .env from template."
  echo "    Open it now and fill in your API keys:"
  echo "    open ${ENV_FILE}"
  echo ""
  echo "    Required keys:"
  echo "      FMP_API_KEY       → https://site.financialmodelingprep.com"
  echo "      ANTHROPIC_API_KEY → https://console.anthropic.com"
  echo "      TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID (for phone alerts)"
  echo "      HF_API_TOKEN      → https://huggingface.co/settings/tokens"
else
  echo "✅  .env already exists"
fi

# ── 7. Initialise DB schema ───────────────────────────────────────────────────
echo ""
echo "Initialising database schema …"
cd "${PROJECT_DIR}"
"${VENV_DIR}/bin/python" scripts/init_db.py
echo "✅  Schema initialised"

# ── 8. Optional: install scheduler ───────────────────────────────────────────
echo ""
read -r -p "Install launchd scheduler (auto-run pipeline Mon–Fri at 19:05)? [y/N] " answer
if [[ "${answer,,}" == "y" ]]; then
  bash "${PROJECT_DIR}/scripts/install_scheduler.sh"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Setup complete!                                     ║"
echo "║                                                      ║"
echo "║  Next steps:                                         ║"
echo "║  1. Fill in your API keys in .env                   ║"
echo "║  2. Run a dry-run to verify everything works:       ║"
echo "║     source .venv/bin/activate                       ║"
echo "║     python -m pipeline.run_pipeline --dry-run       ║"
echo "║  3. Run the backtest:                               ║"
echo "║     python scripts/run_backtest.py                  ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
