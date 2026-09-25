#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIRESIAS_ROOT="${TIRESIAS_ROOT:-$SCRIPT_DIR/..}"
TIRESIAS_DATA_ROOT="${TIRESIAS_DATA_ROOT:-$TIRESIAS_ROOT/data}"
export TIRESIAS_ROOT
export TIRESIAS_DATA_ROOT
export PYTHONPATH="$SCRIPT_DIR:$TIRESIAS_ROOT:${PYTHONPATH:-}"

# Default to eco profile for tests on low-spec systems
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
echo "Running TIRESIAS_ENGINE Serving Core Tests (Linux Launcher)"
echo "  - Profile:   $TIRESIAS_PROFILE"
echo "  - Python:    $PYTHON"
echo "  - Data Root: $TIRESIAS_DATA_ROOT"
echo "================================================================="

"$PYTHON" "$SCRIPT_DIR/autotest/test_server.py"
