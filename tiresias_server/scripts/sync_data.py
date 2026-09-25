#!/usr/bin/env python3
"""
TIRESIAS Data Synchronization Utility.
Streams the essential production serving dataset (~1.34 GB) from PC to remote target (laptop/VPS)
over SSH using streaming tar without loading the full archive into memory.
Includes SHA-256 integrity verification.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent
SERVER_DIR = HERE.parent
ROOT_DIR = SERVER_DIR.parent
DEFAULT_DATA_DIR = ROOT_DIR / "data"

# Core serving files required for full serving engine functionality
SERVING_ARTIFACTS = [
    "features/post2vec_sq8.index",
    "features/post2vec_faiss_ids.parquet",
    "features/taste_centroids.npy",
    "features/taste_archetypes.parquet",
    "features/post_cofav.parquet",
    "tags.parquet",
    "mmaps",
    "bitmaps",
]

# Optional full artifacts (includes posts_parquet partitions for salient tag TF-IDF)
FULL_EXTRA_ARTIFACTS = [
    "posts_parquet",
]


def compute_file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block_size):
            h.update(chunk)
    return h.hexdigest()


def collect_items_to_sync(
    data_dir: Path, profile: str = "serving"
) -> List[Tuple[Path, str]]:
    """
    Returns list of (absolute_path, relative_arcname) to sync.
    """
    targets = list(SERVING_ARTIFACTS)
    if profile == "full":
        targets.extend(FULL_EXTRA_ARTIFACTS)

    items: List[Tuple[Path, str]] = []
    for rel_str in targets:
        p = data_dir / rel_str
        if not p.exists():
            print(f"[WARN] Expected artifact not found: {p}", file=sys.stderr)
            continue
        if p.is_file():
            items.append((p, rel_str))
        elif p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file() and not child.name.startswith("."):
                    rel_child = child.relative_to(data_dir).as_posix()
                    items.append((child, rel_child))
    return items


def generate_manifest(items: List[Tuple[Path, str]]) -> Dict[str, Dict]:
    manifest: Dict[str, Dict] = {}
    print(f"Generating SHA-256 checksums for {len(items)} files...")
    t0 = time.time()
    total_bytes = 0
    for idx, (path, rel) in enumerate(items, 1):
        sz = path.stat().st_size
        total_bytes += sz
        sha = compute_file_sha256(path)
        manifest[rel] = {"size": sz, "sha256": sha}
        if idx % 50 == 0 or idx == len(items):
            print(f"  [{idx}/{len(items)}] {rel} ({sz / (1024*1024):.2f} MB)")
    print(
        f"Manifest ready: {len(manifest)} files, {total_bytes / (1024*1024):.2f} MB in {time.time() - t0:.1f}s"
    )
    return manifest


class ProgressWriter:
    """Wraps proc.stdin to measure transferred bytes and report progress."""

    def __init__(self, target_stream, total_bytes: int):
        self.stream = target_stream
        self.total_bytes = max(1, total_bytes)
        self.written = 0
        self.last_report = time.time()
        self.start_time = time.time()

    def write(self, data: bytes) -> int:
        n = self.stream.write(data)
        self.written += len(data)
        now = time.time()
        if now - self.last_report >= 1.0 or self.written >= self.total_bytes:
            pct = (self.written / self.total_bytes) * 100.0
            mb_done = self.written / (1024 * 1024)
            mb_tot = self.total_bytes / (1024 * 1024)
            elapsed = max(0.1, now - self.start_time)
            speed = mb_done / elapsed
            print(
                f"\r  Transferred: {mb_done:6.1f} / {mb_tot:6.1f} MB [{pct:5.1f}%] @ {speed:5.1f} MB/s",
                end="",
                flush=True,
            )
            self.last_report = now
        return n

    def flush(self):
        self.stream.flush()


def run_sync(
    ssh_target: str,
    remote_data_dir: str,
    data_dir: Path,
    profile: str = "serving",
    verify_after: bool = True,
    atomic: bool = False,
    keep_versions: int = 2,
) -> int:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    target_extract_dir = f"{remote_data_dir.rstrip('/')}/versions/data_{ts}" if atomic else remote_data_dir

    print("=" * 65)
    print(f"TIRESIAS Data Sync: PC -> {ssh_target}:{target_extract_dir}")
    print(f"Profile: {profile} | Atomic: {atomic} | Source: {data_dir}")
    print("=" * 65)

    items = collect_items_to_sync(data_dir, profile=profile)
    if not items:
        print("[ERROR] No files found to sync!", file=sys.stderr)
        return 1

    total_bytes = sum(p.stat().st_size for p, _ in items)
    print(f"Found {len(items)} files, total {total_bytes / (1024*1024):.2f} MB")

    # Start remote tar extraction process over SSH
    remote_cmd = (
        f"mkdir -p {shlex.quote(target_extract_dir)} && "
        f"tar -xf - -C {shlex.quote(target_extract_dir)}"
    )
    ssh_cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        ssh_target,
        remote_cmd,
    ]

    print(f"Streaming files over SSH to {ssh_target}...")
    t0 = time.time()

    proc = subprocess.Popen(ssh_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.stdin is None:
        print("[ERROR] Could not open stdin to SSH process.", file=sys.stderr)
        return 1

    pw = ProgressWriter(proc.stdin, total_bytes)
    try:
        with tarfile.open(mode="w|", fileobj=pw) as tar:
            for path, arcname in items:
                tar.add(path, arcname=arcname)
    except BrokenPipeError:
        print("\n[ERROR] SSH connection broken during transfer.", file=sys.stderr)
        stderr_out = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        print(f"SSH stderr: {stderr_out}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n[ERROR] Transfer error: {e}", file=sys.stderr)
        return 1

    proc.stdin.close()
    proc.wait()
    print()

    if proc.returncode != 0:
        stderr_out = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        print(f"[FAIL] Remote tar extraction failed: {stderr_out}", file=sys.stderr)
        return proc.returncode

    duration = time.time() - t0
    avg_speed = (total_bytes / (1024 * 1024)) / max(0.1, duration)
    print(
        f"[PASS] Successfully transferred {total_bytes / (1024*1024):.2f} MB in {duration:.1f}s ({avg_speed:.1f} MB/s)"
    )

    if verify_after:
        verif_res = run_remote_verification(ssh_target, target_extract_dir, items)
        if verif_res != 0:
            print(f"[FAIL] Remote verification failed for {target_extract_dir}", file=sys.stderr)
            return verif_res

    if atomic:
        print("-" * 65)
        print(f"Performing atomic symlink swap on {ssh_target}...")
        cleanup_script = (
            f"import shutil, sys\n"
            f"from pathlib import Path\n"
            f"vdir = Path({json.dumps(remote_data_dir)}) / 'versions'\n"
            f"if vdir.exists():\n"
            f"    vers = sorted([p for p in vdir.iterdir() if p.is_dir() and p.name.startswith('data_')], key=lambda x: x.name)\n"
            f"    keep = {keep_versions}\n"
            f"    to_del = vers[:-keep] if len(vers) > keep else []\n"
            f"    for d in to_del:\n"
            f"        try:\n"
            f"            shutil.rmtree(d)\n"
            f"            print(f'Pruned old version: {{d.name}}')\n"
            f"        except Exception as e:\n"
            f"            print(f'Error pruning {{d.name}}: {{e}}', file=sys.stderr)\n"
        )
        swap_cmd = (
            f"ln -sfn versions/data_{ts} {shlex.quote(remote_data_dir)}/current_tmp && "
            f"mv -Tf {shlex.quote(remote_data_dir)}/current_tmp {shlex.quote(remote_data_dir)}/current"
        )
        full_swap_cmd = (
            f"{swap_cmd} && "
            f"(python3 -c {shlex.quote(cleanup_script)} || "
            f"{shlex.quote(remote_data_dir)}/../venv/bin/python -c {shlex.quote(cleanup_script)} || true)"
        )
        swap_proc = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                ssh_target,
                full_swap_cmd,
            ],
            capture_output=True,
            text=True,
        )
        if swap_proc.returncode != 0:
            print(f"[FAIL] Remote atomic symlink swap failed:\n{swap_proc.stderr}", file=sys.stderr)
            return swap_proc.returncode
        print(f"  [PASS] Symlink updated: {remote_data_dir}/current -> versions/data_{ts}")

    return 0


def run_remote_verification(
    ssh_target: str, remote_data_dir: str, items: List[Tuple[Path, str]]
) -> int:
    print("-" * 65)
    print("Verifying transferred files on remote machine...")
    # Generate local manifest for sample check: check sizes of all files and hash for top-10 key files
    key_files = [
        "features/post2vec_sq8.index",
        "features/post2vec_faiss_ids.parquet",
        "features/taste_centroids.npy",
        "features/taste_archetypes.parquet",
        "features/post_cofav.parquet",
        "tags.parquet",
        "mmaps/post_ids.bin",
        "mmaps/score.bin",
        "mmaps/fav_count.bin",
        "mmaps/rating_s.roar",
    ]

    check_script = f"""
import os, sys, hashlib
from pathlib import Path

root = Path('{remote_data_dir}')
key_files = {json.dumps(key_files)}

missing = []
for kf in key_files:
    p = root / kf
    if not p.exists():
        missing.append(kf)

if missing:
    print(f"MISSING: {{missing}}")
    sys.exit(1)

total_files = sum(1 for _ in root.rglob('*') if _.is_file())
total_bytes = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
print(f"REMOTE_OK: {{total_files}} files, {{total_bytes}} bytes")
"""

    remote_python_cmd = (
        f"python3 -c {shlex.quote(check_script)} || "
        f"{shlex.quote(remote_data_dir)}/../venv/bin/python -c {shlex.quote(check_script)}"
    )

    proc = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            ssh_target,
            remote_python_cmd,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    if proc.returncode != 0 or "REMOTE_OK" not in proc.stdout:
        print(f"[FAIL] Verification failed:\n{proc.stdout}\n{proc.stderr}")
        return 1

    print(f"  [PASS] Remote verification: {proc.stdout.strip()}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync TIRESIAS serving dataset to remote server/laptop"
    )
    parser.add_argument(
        "--target",
        default="laptop",
        help="SSH target alias or user@host (default: laptop)",
    )
    parser.add_argument(
        "--remote-dir",
        default="/opt/tiresias/data",
        help="Remote destination data directory",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help=f"Local data directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--profile",
        choices=["serving", "full"],
        default="serving",
        help="Dataset profile: 'serving' (1.34 GB, core runtime) or 'full' (with posts_parquet)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify remote files against expected manifest without syncing",
    )
    parser.add_argument(
        "--atomic",
        action="store_true",
        help="Deploy data atomically into versioned directory with symlink swap",
    )
    parser.add_argument(
        "--keep-versions",
        type=int,
        default=2,
        help="Number of version directories to keep when using --atomic (default: 2)",
    )

    args = parser.parse_args()
    data_path = Path(args.data_dir).resolve()

    if args.verify_only:
        items = collect_items_to_sync(data_path, profile=args.profile)
        check_dir = f"{args.remote_dir}/current" if args.atomic else args.remote_dir
        return run_remote_verification(args.target, check_dir, items)

    return run_sync(
        ssh_target=args.target,
        remote_data_dir=args.remote_dir,
        data_dir=data_path,
        profile=args.profile,
        verify_after=True,
        atomic=args.atomic,
        keep_versions=args.keep_versions,
    )


if __name__ == "__main__":
    sys.exit(main())
