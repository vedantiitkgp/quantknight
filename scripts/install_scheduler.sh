#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  install_scheduler.sh
#
#  Installs a macOS launchd job that fires the nightly pipeline
#  every weekday at 7:00 PM local time.
#
#  Usage:
#    chmod +x scripts/install_scheduler.sh
#    ./scripts/install_scheduler.sh
#
#  To uninstall:
#    launchctl unload ~/Library/LaunchAgents/com.stockengine.pipeline.plist
#    rm ~/Library/LaunchAgents/com.stockengine.pipeline.plist
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${PROJECT_DIR}/.venv/bin/python"
PLIST_PATH="${HOME}/Library/LaunchAgents/com.stockengine.pipeline.plist"
LOG_DIR="${PROJECT_DIR}/logs"

# ── Validate ──────────────────────────────────────────────────────────────────
if [[ ! -f "${VENV_PYTHON}" ]]; then
  echo "ERROR: Virtual environment not found at ${VENV_PYTHON}"
  echo "Run './setup_mac.sh' first to create the environment."
  exit 1
fi

mkdir -p "${LOG_DIR}"

# ── Write plist ────────────────────────────────────────────────────────────────
cat > "${PLIST_PATH}" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.stockengine.pipeline</string>

  <key>ProgramArguments</key>
  <array>
    <string>${VENV_PYTHON}</string>
    <string>-m</string>
    <string>pipeline.run_pipeline</string>
  </array>

  <key>WorkingDirectory</key>
  <string>${PROJECT_DIR}</string>

  <!-- Fire at 19:05 Mon–Fri (5 min after market data settles) -->
  <key>StartCalendarInterval</key>
  <array>
    <dict>
      <key>Weekday</key><integer>1</integer>
      <key>Hour</key><integer>19</integer>
      <key>Minute</key><integer>5</integer>
    </dict>
    <dict>
      <key>Weekday</key><integer>2</integer>
      <key>Hour</key><integer>19</integer>
      <key>Minute</key><integer>5</integer>
    </dict>
    <dict>
      <key>Weekday</key><integer>3</integer>
      <key>Hour</key><integer>19</integer>
      <key>Minute</key><integer>5</integer>
    </dict>
    <dict>
      <key>Weekday</key><integer>4</integer>
      <key>Hour</key><integer>19</integer>
      <key>Minute</key><integer>5</integer>
    </dict>
    <dict>
      <key>Weekday</key><integer>5</integer>
      <key>Hour</key><integer>19</integer>
      <key>Minute</key><integer>5</integer>
    </dict>
  </array>

  <key>StandardOutPath</key>
  <string>${LOG_DIR}/pipeline.log</string>

  <key>StandardErrorPath</key>
  <string>${LOG_DIR}/pipeline_err.log</string>

  <!-- Restart on crash -->
  <key>KeepAlive</key>
  <false/>

  <!-- Run even if Mac was asleep at trigger time (fires within 10 min window) -->
  <key>ProcessType</key>
  <string>Background</string>
</dict>
</plist>
PLIST

# ── Load into launchd ─────────────────────────────────────────────────────────
# Unload first in case it was already registered
launchctl unload "${PLIST_PATH}" 2>/dev/null || true
launchctl load -w "${PLIST_PATH}"

echo ""
echo "✅  Scheduler installed successfully."
echo "    Pipeline will run Mon–Fri at 19:05 local time."
echo ""
echo "    Logs:  ${LOG_DIR}/pipeline.log"
echo ""
echo "    To trigger a manual run now:"
echo "    launchctl start com.stockengine.pipeline"
echo ""
echo "    To check status:"
echo "    launchctl list | grep stockengine"
