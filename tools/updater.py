#!/usr/bin/env python3
"""
TIRESIAS ENGINE - Automated Updater & Orchestrator
Safe, device-agnostic pipeline for updating recommendation artifacts from raw dumps,
managing maintenance mode, and deploying to target serving nodes.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
CONFIG_FILE = Path(__file__).resolve().parent / "deploy_targets.json"


def log(msg: str, level: str = "INFO") -> None:
    now = datetime.datetime.now().strftime("%H:%M:%S")
    prefix = {
        "INFO": "[\033[94mINFO\033[0m]",
        "SUCCESS": "[\033[92mSUCCESS\033[0m]",
        "WARN": "[\033[93mWARN\033[0m]",
        "ERROR": "[\033[91mERROR\033[0m]",
    }.get(level, f"[{level}]")
    print(f"{prefix} [{now}] {msg}", flush=True)


def check_prerequisites(data_root: Path) -> bool:
    """Verifies that required raw data dumps exist before starting."""
    log(f"Checking prerequisites in {data_root}...")
    posts_csv = data_root / "posts.csv"
    tags_csv = data_root / "tags.csv"
    pools_csv = data_root / "pools.csv"

    missing = []
    if not posts_csv.exists():
        missing.append("posts.csv (raw posts export)")
    if not tags_csv.exists():
        missing.append("tags.csv (raw tags export)")
    if not pools_csv.exists():
        missing.append("pools.csv (raw pools export)")

    if missing:
        log(f"Missing required raw dumps: {', '.join(missing)}", "ERROR")
        return False

    posts_size_gb = posts_csv.stat().st_size / (1024**3)
    log(f"Found posts.csv ({posts_size_gb:.2f} GB), tags.csv, pools.csv.", "SUCCESS")

    # Check Python environment dependencies
    missing_deps = []
    for mod in ("duckdb", "faiss", "scipy", "polars"):
        try:
            __import__(mod)
        except ImportError:
            missing_deps.append(mod)

    if missing_deps:
        log(f"Warning: Current Python ({sys.executable}) is missing: {', '.join(missing_deps)}.", "WARN")
        log("The build pipeline will fail without these packages. Make sure to run in an environment with full dependencies.", "WARN")
    else:
        log(f"Python environment ({sys.executable}) has all required packages (duckdb, faiss, scipy, polars).", "SUCCESS")

    return True


def backup_database(data_root: Path) -> Optional[Path]:
    """Safely backs up tiresias_user.db containing boards and user feedback."""
    db_path = data_root / "tiresias_user.db"
    if not db_path.exists():
        log(f"Database {db_path} does not exist yet (first run?) - skipping backup.")
        return None

    backups_dir = data_root / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = backups_dir / f"tiresias_user_{timestamp}.db"

    log(f"Creating database backup: {backup_file.name}...")
    shutil.copy2(db_path, backup_file)

    # Check size
    orig_sz = db_path.stat().st_size
    bak_sz = backup_file.stat().st_size
    if orig_sz == bak_sz:
        log(f"Database backed up successfully ({orig_sz / 1024:.1f} KB).", "SUCCESS")
        return backup_file
    else:
        log(f"Backup verification failed: size mismatch ({bak_sz} != {orig_sz})", "ERROR")
        return None


def prepare_uploaders(data_root: Path, workers: int = 8) -> bool:
    """Extracts uploaders_uploads.csv from posts.csv using DuckDB."""
    posts_csv = data_root / "posts.csv"
    out_csv = data_root / "uploaders_uploads.csv"

    if out_csv.exists() and out_csv.stat().st_mtime > posts_csv.stat().st_mtime:
        log("uploaders_uploads.csv is already up to date with posts.csv - skipping extraction.")
        return True

    log("Extracting uploader statistics via DuckDB...")
    t0 = time.perf_counter()
    try:
        from build_index.uploaders_extract import extract_with_duckdb
        extract_with_duckdb(posts_csv, is_parquet=False, out_path=out_csv, workers=workers)
        dt = time.perf_counter() - t0
        log(f"uploaders_uploads.csv generated in {dt:.1f}s ({out_csv.stat().st_size / (1024**2):.1f} MB).", "SUCCESS")
        return True
    except Exception as e:
        log(f"DuckDB uploaders extraction failed: {e}", "ERROR")
        return False


def run_build_pipeline(data_root: Path, workers: int = 8) -> bool:
    """Runs the complete build_index pipeline."""
    # Pre-flight check for critical build dependencies
    missing_deps = []
    for mod in ("duckdb", "faiss", "scipy", "polars"):
        try:
            __import__(mod)
        except ImportError:
            missing_deps.append(mod)

    if missing_deps:
        log(f"Cannot start build pipeline: missing critical dependencies: {', '.join(missing_deps)}", "ERROR")
        log(f"Active Python interpreter: {sys.executable}", "ERROR")
        log("Please activate an environment containing faiss and duckdb, or set PYTHON_EXE.", "ERROR")
        return False

    log(f"Starting build_index pipeline with {workers} workers...")
    t0 = time.perf_counter()

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = f"{PROJECT_ROOT};{env.get('PYTHONPATH', '')}"

    cmd = [
        sys.executable,
        "-m",
        "build_index.main",
        "--root",
        str(data_root),
        "--do",
        "all",
        "--workers",
        str(workers),
        "--post2vec-sq8",
    ]

    log(f"Executing: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    dt = time.perf_counter() - t0

    if proc.returncode == 0:
        log(f"Build pipeline finished successfully in {dt / 60:.1f} minutes.", "SUCCESS")
        return True
    else:
        log(f"Build pipeline failed with exit code {proc.returncode}.", "ERROR")
        return False


def set_maintenance_api(api_url: str, enable: bool, admin_key: Optional[str] = None) -> bool:
    """Sends maintenance enable/disable command to serving node API with admin auth."""
    import urllib.request
    key = admin_key or os.environ.get("TIRESIAS_ADMIN_KEY")
    action = "enable" if enable else "disable"
    endpoint = f"{api_url.rstrip('/')}/api/v1/system/maintenance/{action}"
    headers = {"Content-Type": "application/json"}
    if key:
        headers["X-Admin-Key"] = key
        headers["Authorization"] = f"Bearer {key}"
    try:
        req = urllib.request.Request(endpoint, data=b"{}", headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                log(f"Serving API ({api_url}) maintenance: {action.upper()}.", "SUCCESS")
                return True
    except Exception as e:
        log(f"Could not reach {endpoint} (server might be offline or unauthorized): {e}", "WARN")
    return False


def reload_artifacts_api(api_url: str, admin_key: Optional[str] = None) -> bool:
    """Sends hot-reload artifacts command to serving node API with admin auth."""
    import urllib.request
    key = admin_key or os.environ.get("TIRESIAS_ADMIN_KEY")
    endpoint = f"{api_url.rstrip('/')}/api/v1/system/reload-artifacts"
    headers = {"Content-Type": "application/json"}
    if key:
        headers["X-Admin-Key"] = key
        headers["Authorization"] = f"Bearer {key}"
    try:
        req = urllib.request.Request(endpoint, data=b"{}", headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status == 200:
                log(f"Serving API ({api_url}) artifacts reloaded successfully.", "SUCCESS")
                return True
    except Exception as e:
        log(f"Artifact reload request to {endpoint} failed: {e}", "ERROR")
    return False


def check_health_api(api_url: str) -> Optional[Dict[str, Any]]:
    """Checks health status of an API endpoint."""
    import urllib.request
    endpoint = f"{api_url.rstrip('/')}/api/v1/system/health"
    try:
        req = urllib.request.Request(endpoint, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                return data
    except Exception:
        pass
    return None


def deploy_to_target(name: str, target_cfg: Dict[str, Any], data_root: Path) -> bool:
    """Deploys compiled artifacts to a specific target (local or SSH remote)."""
    if not target_cfg.get("enabled", False):
        log(f"Target '{name}' is disabled in config - skipping.")
        return True

    target_type = target_cfg.get("type", "ssh")
    log(f"Deploying artifacts to target '{name}' ({target_type})...")
    admin_key = target_cfg.get("admin_key") or os.environ.get("TIRESIAS_ADMIN_KEY")
    api_url = target_cfg.get("api_url")

    if target_type == "local":
        target_dir = target_cfg.get("data_dir")
        if api_url:
            set_maintenance_api(api_url, enable=True, admin_key=admin_key)

        if target_dir:
            resolved_target = (PROJECT_ROOT / target_dir).resolve() if not Path(target_dir).is_absolute() else Path(target_dir).resolve()
            if resolved_target != data_root.resolve():
                log(f"Syncing artifacts locally to {resolved_target}...")
                resolved_target.mkdir(parents=True, exist_ok=True)
                for aitem in ["features", "mmaps", "bitmaps", "tags.parquet"]:
                    src = data_root / aitem
                    dst = resolved_target / aitem
                    if src.exists():
                        if src.is_dir():
                            shutil.copytree(src, dst, dirs_exist_ok=True)
                        else:
                            shutil.copy2(src, dst)
        log(f"Local target '{name}' data is in-place ({data_root}).", "SUCCESS")

        if api_url:
            reload_artifacts_api(api_url, admin_key=admin_key)
            set_maintenance_api(api_url, enable=False, admin_key=admin_key)
            health = check_health_api(api_url)
            if health and health.get("status") in ("online", "ok"):
                posts = health.get("total_posts_indexed", 0)
                log(f"Local node '{name}' is online and healthy ({posts:,} posts indexed).", "SUCCESS")

        return True

    elif target_type == "ssh":
        host = target_cfg["host"]
        user = target_cfg["user"]
        remote_data = target_cfg["remote_data_dir"]
        service = target_cfg.get("service_name")
        restart_service = target_cfg.get("restart_service", False)
        atomic = target_cfg.get("atomic", True)

        # 1. Put remote server into maintenance if api_url is present
        if api_url:
            set_maintenance_api(api_url, enable=True, admin_key=admin_key)
        elif service and not restart_service:
            log(f"Stopping remote service '{service}' on {host}...")
            subprocess.run(["ssh", f"{user}@{host}", f"systemctl --user stop {service}"], check=False)

        # 2. High-speed cross-platform streaming sync via tar over SSH
        log(f"Streaming serving artifacts to {user}@{host}:{remote_data}...")
        try:
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))
            from tiresias_server.scripts.sync_data import run_sync

            ssh_target = f"{user}@{host}"
            res = run_sync(
                ssh_target=ssh_target,
                remote_data_dir=remote_data,
                data_dir=data_root,
                profile="serving",
                verify_after=True,
                atomic=atomic,
            )
            if res != 0:
                log(f"Artifact synchronization to '{name}' failed (code {res}).", "ERROR")
                if api_url:
                    set_maintenance_api(api_url, enable=False, admin_key=admin_key)
                return False
        except Exception as e:
            log(f"Artifact streaming failed: {e}", "ERROR")
            if api_url:
                set_maintenance_api(api_url, enable=False, admin_key=admin_key)
            return False

        # 3. Hot-reload artifacts & disable maintenance if api_url is present
        if api_url:
            reload_artifacts_api(api_url, admin_key=admin_key)
            set_maintenance_api(api_url, enable=False, admin_key=admin_key)

        # 4. Service restart via SSH if requested or needed
        if service:
            if restart_service:
                log(f"Restarting remote service '{service}' on {host}...")
                subprocess.run(["ssh", f"{user}@{host}", f"systemctl --user restart {service}"], check=False)
            elif not api_url:
                log(f"Starting remote service '{service}' on {host}...")
                subprocess.run(["ssh", f"{user}@{host}", f"systemctl --user start {service}"], check=False)

        # 5. Check remote health
        if api_url:
            log(f"Checking health on {api_url}...")
            for attempt in range(1, 10):
                time.sleep(2)
                health = check_health_api(api_url)
                if health and health.get("status") in ("online", "ok"):
                    posts = health.get("total_posts_indexed", 0)
                    log(f"Remote node '{name}' is online and healthy ({posts:,} posts indexed).", "SUCCESS")
                    return True
                log(f"Waiting for remote service to initialize ({attempt}/9)...")
            log(f"Remote node '{name}' health check timed out or failed.", "WARN")

        return True

    else:
        log(f"Unknown target type '{target_type}' for {name}", "ERROR")
        return False


def load_deploy_targets() -> Dict[str, Any]:
    if not CONFIG_FILE.exists():
        log(f"Configuration file {CONFIG_FILE} not found. Creating default...", "WARN")
        default_cfg = {
            "local": {"enabled": True, "type": "local", "api_url": "http://127.0.0.1:8000"}
        }
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(default_cfg, f, indent=2)
        return default_cfg

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Tiresias Engine Automated Updater & Deployer")
    parser.add_argument("--root", type=str, default=str(DEFAULT_DATA_ROOT), help="Path to data directory")
    parser.add_argument("--step", choices=["prereq", "backup", "uploaders", "build", "deploy", "all"], default="all",
                        help="Individual step to execute (default: all)")
    parser.add_argument("--target", type=str, default="all", help="Target node to deploy to (from deploy_targets.json)")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers for build pipeline")
    parser.add_argument("--skip-uploaders", action="store_true", help="Skip DuckDB uploaders extraction")
    parser.add_argument("--skip-deploy", action="store_true", help="Skip deploying to remote nodes")

    args = parser.parse_args()
    data_root = Path(args.root).resolve()

    log("=" * 60)
    log("TIRESIAS AUTOMATED ARTIFACT UPDATER")
    log(f"Data root: {data_root}")
    log("=" * 60)

    # 1. Prerequisites check
    if args.step in ("prereq", "all"):
        if not check_prerequisites(data_root):
            sys.exit(1)
        if args.step == "prereq":
            sys.exit(0)

    # 2. Database Backup
    if args.step in ("backup", "all"):
        backup_database(data_root)
        if args.step == "backup":
            sys.exit(0)

    # 3. Uploaders extraction (DuckDB)
    if args.step in ("uploaders", "all") and not args.skip_uploaders:
        if not prepare_uploaders(data_root, workers=args.workers):
            log("Uploaders extraction encountered an error.", "WARN")
        if args.step == "uploaders":
            sys.exit(0)

    # 4. Build pipeline
    if args.step in ("build", "all"):
        if not run_build_pipeline(data_root, workers=args.workers):
            log("Build pipeline failed. Aborting deployment.", "ERROR")
            sys.exit(1)
        if args.step == "build":
            sys.exit(0)

    # 5. Deployment
    if args.step in ("deploy", "all") and not args.skip_deploy:
        targets = load_deploy_targets()
        to_deploy = targets.keys() if args.target == "all" else [args.target]

        for tname in to_deploy:
            if tname in targets:
                deploy_to_target(tname, targets[tname], data_root)
            else:
                log(f"Target '{tname}' not defined in deploy_targets.json", "ERROR")

    log("=" * 60)
    log("ALL UPDATE STEPS COMPLETED SUCCESSFULLY!", "SUCCESS")
    log("=" * 60)


if __name__ == "__main__":
    main()
