#!/usr/bin/env bash
set -e

# TIRESIAS ENGINE - Automated Updater Launcher (Linux / VPS / Server)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIRESIAS_ROOT="${TIRESIAS_ROOT:-$SCRIPT_DIR/..}"
TIRESIAS_DATA_ROOT="${TIRESIAS_DATA_ROOT:-$TIRESIAS_ROOT/data}"
export TIRESIAS_ROOT
export TIRESIAS_DATA_ROOT
export PYTHONPATH="$TIRESIAS_ROOT:${PYTHONPATH:-}"

# Locate Python interpreter dynamically without hardcoded paths or usernames:
if [ -n "${PYTHON_EXE:-}" ] && [ -x "$PYTHON_EXE" ]; then
    PYTHON="$PYTHON_EXE"
else
    # Gather potential Python candidates
    CANDIDATES=()
    [ -n "${VIRTUAL_ENV:-}" ] && CANDIDATES+=("$VIRTUAL_ENV/bin/python")
    [ -n "${CONDA_PREFIX:-}" ] && CANDIDATES+=("$CONDA_PREFIX/bin/python")
    CANDIDATES+=(
        "$TIRESIAS_ROOT/venv/bin/python3"
        "$TIRESIAS_ROOT/venv/bin/python"
        "$TIRESIAS_ROOT/.venv/bin/python3"
        "$TIRESIAS_ROOT/.venv/bin/python"
        "$TIRESIAS_ROOT/env/bin/python"
        "$HOME/tiresias/venv/bin/python"
        "$HOME/.venv/bin/python"
    )

    # Dynamic scan of conda environments
    for cdir in "$HOME/.conda/envs" "$HOME/miniconda3/envs" "$HOME/anaconda3/envs" "/opt/conda/envs"; do
        if [ -d "$cdir" ]; then
            for edir in "$cdir"/*; do
                [ -x "$edir/bin/python" ] && CANDIDATES+=("$edir/bin/python")
            done
        fi
    done

    # System PATH
    command -v python3 &>/dev/null && CANDIDATES+=("$(command -v python3)")
    command -v python &>/dev/null && CANDIDATES+=("$(command -v python)")

    PYTHON=""
    # PASS 1: Prioritize interpreters with full pipeline dependencies (duckdb + faiss)
    for cand in "${CANDIDATES[@]}"; do
        if [ -x "$cand" ] && "$cand" -c "import duckdb, faiss" &>/dev/null; then
            PYTHON="$cand"
            break
        fi
    done

    # PASS 2: Fallback to interpreters with at least duckdb
    if [ -z "$PYTHON" ]; then
        for cand in "${CANDIDATES[@]}"; do
            if [ -x "$cand" ] && "$cand" -c "import duckdb" &>/dev/null; then
                PYTHON="$cand"
                break
            fi
        done
    fi

    # PASS 3: Final fallback to any working python interpreter
    if [ -z "$PYTHON" ]; then
        for cand in "${CANDIDATES[@]}"; do
            if [ -x "$cand" ] && "$cand" -c "import sys" &>/dev/null; then
                PYTHON="$cand"
                break
            fi
        done
    fi

    if [ -z "$PYTHON" ]; then
        echo "ERROR: Python interpreter not found in PATH or standard virtual environments." >&2
        echo "Please activate a virtual environment or set PYTHON_EXE=/path/to/python" >&2
        exit 1
    fi
fi

cd "$TIRESIAS_ROOT"

# If arguments were passed from CLI, run directly
if [ $# -gt 0 ]; then
    echo "[RUN] $PYTHON -m tools.updater $@"
    exec "$PYTHON" -m tools.updater "$@"
fi

# Interactive menu if launched without arguments
echo "================================================================"
echo "          TIRESIAS ENGINE - AUTOMATED ARTIFACT UPDATER"
echo "================================================================"
echo "Python:    $PYTHON"
echo "Data Root: $TIRESIAS_DATA_ROOT"
echo ""
echo "Choose an action:"
echo "  [1] Full Update Cycle (--all: backup, uploaders, build, deploy)"
echo "  [2] Build Index Pipeline only (--step build)"
echo "  [3] Extract Uploaders stats via DuckDB (--step uploaders)"
echo "  [4] Backup Database only (--step backup)"
echo "  [5] Deploy Artifacts to Targets (--step deploy)"
echo "  [6] Check Prerequisites (--step prereq)"
echo "  [0] Exit"
echo "================================================================"
read -r -p "Enter choice (default: 1): " ACTION
ACTION="${ACTION:-1}"

case "$ACTION" in
    1)
        echo ""
        echo "Starting Full Update Cycle..."
        "$PYTHON" -m tools.updater --all
        ;;
    2)
        echo ""
        echo "Starting Build Pipeline..."
        "$PYTHON" -m tools.updater --step build
        ;;
    3)
        echo ""
        echo "Extracting Uploaders..."
        "$PYTHON" -m tools.updater --step uploaders
        ;;
    4)
        echo ""
        echo "Backing up database..."
        "$PYTHON" -m tools.updater --step backup
        ;;
    5)
        echo ""
        echo "Deploying to configured targets..."
        "$PYTHON" -m tools.updater --step deploy
        ;;
    6)
        echo ""
        echo "Checking prerequisites..."
        "$PYTHON" -m tools.updater --step prereq
        ;;
    0)
        exit 0
        ;;
    *)
        echo "Invalid choice."
        exit 1
        ;;
esac
