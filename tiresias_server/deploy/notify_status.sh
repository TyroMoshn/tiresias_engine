#!/usr/bin/env bash
# Non-intrusive desktop status notification for TIRESIAS Serving API on Linux Lite display :0
set -u

export DISPLAY="${DISPLAY:-:0}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/1000/bus}"
export XAUTHORITY="${XAUTHORITY:-/home/tyro/.Xauthority}"

ACTION="${1:-started}"
PORT="${2:-8000}"
INFO="${3:-5.16M vectors}"
PROFILE="${4:-eco}"

if command -v notify-send >/dev/null 2>&1; then
    if [ "$ACTION" = "started" ]; then
        notify-send \
            -a "TIRESIAS" \
            -u normal \
            -i dialog-information \
            "TIRESIAS Recommender: ACTIVE" \
            "Status: Online on port $PORT\nProfile: $PROFILE | Ready: $INFO"
    elif [ "$ACTION" = "stopped" ]; then
        notify-send \
            -a "TIRESIAS" \
            -u normal \
            -i dialog-warning \
            "TIRESIAS Recommender: STOPPED" \
            "Server shutdown complete."
    elif [ "$ACTION" = "error" ]; then
        notify-send \
            -a "TIRESIAS" \
            -u critical \
            -i dialog-error \
            "TIRESIAS Recommender: ERROR" \
            "${2:-An error occurred}"
    fi
fi
