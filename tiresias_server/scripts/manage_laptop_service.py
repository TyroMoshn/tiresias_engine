#!/usr/bin/env python3
"""
Remote management script for TIRESIAS serving daemon on the laptop testbed over SSH.
Supports install, start, stop, restart, status, logs commands.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

SSH_TARGET = "laptop"


def run_remote(cmd: str, timeout: int = 30) -> subprocess.CompletedProcess:
    ssh_cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        SSH_TARGET,
        cmd,
    ]
    return subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)


def cmd_install() -> int:
    print(f"Installing systemd user service on {SSH_TARGET}...")
    install_script = """
set -e
mkdir -p ~/.config/systemd/user
chmod +x /home/tyro/tiresias/tiresias_server/deploy/notify_status.sh
cp /home/tyro/tiresias/tiresias_server/deploy/tiresias.service ~/.config/systemd/user/tiresias.service
systemctl --user daemon-reload
echo "Service installed and daemon reloaded."
"""
    res = run_remote(install_script)
    print(res.stdout)
    if res.returncode != 0:
        print(f"[FAIL] Install error:\n{res.stderr}", file=sys.stderr)
        return res.returncode
    print("[PASS] tiresias.service installed in ~/.config/systemd/user/")
    return 0


def cmd_start() -> int:
    print(f"Starting tiresias.service on {SSH_TARGET}...")
    res = run_remote("systemctl --user start tiresias.service")
    if res.returncode != 0:
        print(f"[FAIL] Failed to start:\n{res.stderr}", file=sys.stderr)
        return res.returncode
    print("[PASS] Service start command dispatched.")
    return cmd_status()


def cmd_stop() -> int:
    print(f"Stopping tiresias.service on {SSH_TARGET}...")
    res = run_remote("systemctl --user stop tiresias.service")
    if res.returncode != 0:
        print(f"[FAIL] Failed to stop:\n{res.stderr}", file=sys.stderr)
        return res.returncode
    print("[PASS] Service stopped.")
    return 0


def cmd_restart() -> int:
    print(f"Restarting tiresias.service on {SSH_TARGET}...")
    res = run_remote("systemctl --user restart tiresias.service")
    if res.returncode != 0:
        print(f"[FAIL] Failed to restart:\n{res.stderr}", file=sys.stderr)
        return res.returncode
    print("[PASS] Service restart dispatched.")
    return cmd_status()


def cmd_status() -> int:
    print(f"Checking tiresias.service status on {SSH_TARGET}...")
    status_cmd = """
systemctl --user status tiresias.service --no-pager -l || true
echo ""
echo "=== HEALTH CHECK ==="
curl -s -m 3 http://localhost:8000/api/v1/system/health || echo "API is currently offline."
"""
    res = run_remote(status_cmd)
    print(res.stdout)
    return 0


def cmd_logs(lines: int = 40) -> int:
    res = run_remote(f"journalctl --user -u tiresias.service -n {lines} --no-pager")
    print(res.stdout)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage TIRESIAS daemon on laptop testbed")
    parser.add_argument(
        "action",
        choices=["install", "start", "stop", "restart", "status", "logs"],
        default="status",
        nargs="?",
    )
    parser.add_argument("-n", "--lines", type=int, default=40, help="Log lines to display")

    args = parser.parse_args()
    if args.action == "install":
        return cmd_install()
    elif args.action == "start":
        return cmd_start()
    elif args.action == "stop":
        return cmd_stop()
    elif args.action == "restart":
        return cmd_restart()
    elif args.action == "status":
        return cmd_status()
    elif args.action == "logs":
        return cmd_logs(args.lines)
    return 0


if __name__ == "__main__":
    sys.exit(main())
