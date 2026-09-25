#!/usr/bin/env python3
"""
Fast one-shot synchronization script from PC to Laptop over SSH using compressed tar streaming.
Zero LLM token consumption, skips __pycache__, .git, and .venv.
"""
from __future__ import annotations

import io
import os
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVER_DIR = HERE.parent

SSH_TARGET = "laptop"
REMOTE_TARGET_DIR = "/home/tyro/tiresias/tiresias_server"


def sync() -> int:
    def tar_filter(tarinfo):
        parts = tarinfo.name.replace("\\", "/").split("/")
        if any(p in (".git", "__pycache__", ".venv", ".pytest_cache", ".system_generated", "scratch") for p in parts):
            return None
        return tarinfo

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(SERVER_DIR, arcname=".", filter=tar_filter)
    gz_data = buf.getvalue()

    print(f"Compressed {SERVER_DIR.name} into {len(gz_data):,} bytes. Streaming to {SSH_TARGET}:{REMOTE_TARGET_DIR}...")
    remote_cmd = f"mkdir -p {shlex.quote(REMOTE_TARGET_DIR)} && tar -xzf - -C {shlex.quote(REMOTE_TARGET_DIR)}"

    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", SSH_TARGET, remote_cmd],
        input=gz_data,
        capture_output=True,
        timeout=60,
    )
    if proc.returncode != 0:
        print(f"[FAIL] SSH Tar Sync failed: {proc.stderr.decode('utf-8', errors='replace')}", file=sys.stderr)
        return proc.returncode

    print(f"[PASS] Successfully synced all server code to {SSH_TARGET}:{REMOTE_TARGET_DIR} in 1 step.")
    return 0


if __name__ == "__main__":
    sys.exit(sync())
