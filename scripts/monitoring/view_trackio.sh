#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  ./view_trackio.sh [project_name]

Examples:
  ./view_trackio.sh
  ./view_trackio.sh speculators

Behavior:
  1) Shows local data from this folder (latest log-like file)
  2) Prints latest Trackio run info (if available)
  3) Lists metrics for the latest run
  4) Launches Trackio dashboard UI

Env vars:
  SOURCE_DIR   Local data folder to read first (default: script folder)
EOF
  exit 0
fi

PROJECT="${1:-speculators}"
VENV_PATH="${VENV_PATH:-$HOME/virtual_envs/Dflash/bin/activate}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${SOURCE_DIR:-$SCRIPT_DIR}"

if [[ -f "$VENV_PATH" ]]; then
  # shellcheck source=/dev/null
  source "$VENV_PATH"
fi

if ! command -v trackio >/dev/null 2>&1; then
  echo "Error: 'trackio' command not found."
  echo "Tip: activate your env or set VENV_PATH to the right activate script."
  exit 1
fi

echo "Local source dir: $SOURCE_DIR"
latest_local_log="$(
  SOURCE_DIR="$SOURCE_DIR" python3 - <<'PY'
import os
from pathlib import Path

source = Path(os.environ["SOURCE_DIR"])
patterns = ("*.log", "*.out", "*.txt")
files = []
for pattern in patterns:
    files.extend(source.glob(pattern))
files = [p for p in files if p.is_file()]
if files:
    files.sort(key=lambda p: p.stat().st_mtime)
    print(files[-1])
PY
)"

if [[ -n "$latest_local_log" ]]; then
  echo "Latest local log-like file: $latest_local_log"
  echo "----- tail -n 40 -----"
  tail -n 40 "$latest_local_log" || true
  echo "----------------------"
else
  echo "No .log/.out/.txt files found in $SOURCE_DIR"
fi

echo
echo "Project: $PROJECT"
echo "Collecting latest run..."

runs_json="$(trackio list runs --project "$PROJECT" --json 2>/dev/null || true)"
latest_run="$(
  printf '%s' "$runs_json" | python3 - <<'PY'
import json
import sys

raw = sys.stdin.read().strip()
if not raw:
    print("")
    raise SystemExit(0)

try:
    data = json.loads(raw)
except Exception:
    print("")
    raise SystemExit(0)

runs = []
if isinstance(data, list):
    runs = data
elif isinstance(data, dict):
    for key in ("runs", "items", "data"):
        value = data.get(key)
        if isinstance(value, list):
            runs = value
            break

if not runs:
    print("")
    raise SystemExit(0)

if isinstance(runs[0], dict):
    # Prefer timestamp-like fields when present.
    def ts(item):
        for key in ("updated_at", "end_time", "start_time", "created_at", "timestamp"):
            value = item.get(key)
            if value is not None:
                return str(value)
        return ""

    runs = sorted(runs, key=ts)
    print(runs[-1].get("name", ""))
else:
    print(str(runs[-1]))
PY
)"

if [[ -n "$latest_run" ]]; then
  echo "Latest run: $latest_run"
  echo
  trackio get run --project "$PROJECT" --run "$latest_run" || true
  echo
  echo "Metrics for latest run:"
  trackio list metrics --project "$PROJECT" --run "$latest_run" || true
else
  echo "No latest run found (or run list unavailable)."
fi

echo
echo "Starting dashboard..."
exec trackio show --project "$PROJECT"
