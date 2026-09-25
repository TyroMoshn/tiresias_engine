#!/usr/bin/env bash
set -e

# TIRESIAS ENGINE - Documentation Freshness Checker (Linux / POSIX)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIRESIAS_ROOT="${TIRESIAS_ROOT:-$SCRIPT_DIR/..}"
export TIRESIAS_ROOT
export PYTHONPATH="$TIRESIAS_ROOT:${PYTHONPATH:-}"

# Dynamic Python discovery without hardcoded usernames:
if [ -n "${PYTHON_EXE:-}" ] && [ -x "$PYTHON_EXE" ]; then
    PYTHON="$PYTHON_EXE"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    PYTHON="$CONDA_PREFIX/bin/python"
elif [ -x "$TIRESIAS_ROOT/venv/bin/python3" ]; then
    PYTHON="$TIRESIAS_ROOT/venv/bin/python3"
elif [ -x "$TIRESIAS_ROOT/venv/bin/python" ]; then
    PYTHON="$TIRESIAS_ROOT/venv/bin/python"
elif [ -x "$TIRESIAS_ROOT/.venv/bin/python3" ]; then
    PYTHON="$TIRESIAS_ROOT/.venv/bin/python3"
elif [ -x "$TIRESIAS_ROOT/.venv/bin/python" ]; then
    PYTHON="$TIRESIAS_ROOT/.venv/bin/python"
elif [ -x "$HOME/tiresias/venv/bin/python" ]; then
    PYTHON="$HOME/tiresias/venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
elif command -v python &>/dev/null; then
    PYTHON="python"
else
    echo "ERROR: Python not found in PATH or virtual environments." >&2
    exit 1
fi

cd "$TIRESIAS_ROOT"
exec "$PYTHON" -m tools.check_docs "$@"
