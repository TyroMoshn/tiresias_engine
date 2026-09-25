#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIRESIAS_ROOT="${TIRESIAS_ROOT:-$SCRIPT_DIR/..}"
TIRESIAS_DATA_ROOT="${TIRESIAS_DATA_ROOT:-$TIRESIAS_ROOT/data}"
export TIRESIAS_ROOT
export TIRESIAS_DATA_ROOT
export PYTHONPATH="$SCRIPT_DIR:$TIRESIAS_ROOT:${PYTHONPATH:-}"

# Resource profile: 'performance' (desktop / server) or 'eco' (low-spec laptop / weak VPS)
export TIRESIAS_PROFILE="${TIRESIAS_PROFILE:-eco}"

# Locate Python (venv or system)
if [ -f "$SCRIPT_DIR/../venv/bin/python3" ]; then
    PYTHON="$SCRIPT_DIR/../venv/bin/python3"
elif [ -f "$SCRIPT_DIR/venv/bin/python3" ]; then
    PYTHON="$SCRIPT_DIR/venv/bin/python3"
else
    PYTHON="python3"
fi

echo "================================================================="
echo "TIRESIAS_ENGINE Serving Server (Linux Launcher)"
echo "  - Profile:   $TIRESIAS_PROFILE"
echo "  - Python:    $PYTHON"
echo "  - Data Root: $TIRESIAS_DATA_ROOT"
echo "================================================================="

exec "$PYTHON" -m uvicorn app.main:app --host "${TIRESIAS_HOST:-0.0.0.0}" --port "${TIRESIAS_PORT:-8000}"
