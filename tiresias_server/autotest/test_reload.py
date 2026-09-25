#!/usr/bin/env python3
"""
TIRESIAS ENGINE - Stage 5 Automated Remote Updates & Hot-Reload Test Suite
Validates:
1. Engine artifact hot-reloading (zero-downtime reference switching, symlink dereferencing).
2. API authorization and RBAC barriers on reload endpoints (/api/v1/system/reload-artifacts, /api/v1/admin/system/reload-artifacts).
3. Updater helper functions (set_maintenance_api, reload_artifacts_api) and admin key authentication.
4. Maintenance mode protection and behavior under admin authentication.
5. Atomic deployment CLI argument parsing and version management.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import unittest.mock as mock
from pathlib import Path
from typing import Any, Dict

# Setup sys.path
HERE = Path(__file__).resolve().parent
SERVER_DIR = HERE.parent
ROOT_DIR = SERVER_DIR.parent
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient

from app.config import ServerConfig
from app.core.auth import UserRole
from app.core.engine import Engine
from app.main import app
from tools.updater import reload_artifacts_api, set_maintenance_api


def test_engine_reload_artifacts_direct():
    """Tests Engine.reload_artifacts directly on an engine instance."""
    print("Testing Engine.reload_artifacts() direct reloading...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        db_path = tmp_path / "test_engine.db"
        data_path = tmp_path / "data"
        data_path.mkdir(parents=True, exist_ok=True)

        cfg = ServerConfig(custom_db_path=db_path, data_root=data_path)
        engine = Engine(cfg)

        try:
            # 1. First reload on empty data root
            res = engine.reload_artifacts()
            assert isinstance(res, dict), f"Expected dict, got {type(res)}"
            assert res.get("status") == "ok", f"Expected status 'ok', got {res.get('status')}"
            assert res.get("message") == "Artifacts reloaded successfully"
            assert "total_posts_indexed" in res
            assert "faiss_ready" in res
            assert "faiss_total_vectors" in res
            assert "collab_archetypes" in res

            # 2. Check BoardEngine references are updated
            assert engine.boards.mmaps is engine.mmaps
            assert engine.boards.faiss is engine.faiss
            assert engine.boards.bitmaps is engine.bitmaps

            # 3. Test symlink dereferencing if supported by OS
            versions_dir = tmp_path / "versions"
            v1_dir = versions_dir / "data_20260101_000000"
            v2_dir = versions_dir / "data_20260102_000000"
            v1_dir.mkdir(parents=True, exist_ok=True)
            v2_dir.mkdir(parents=True, exist_ok=True)
            current_symlink = tmp_path / "current"

            symlink_supported = False
            try:
                current_symlink.symlink_to(v1_dir, target_is_directory=True)
                symlink_supported = True
            except (OSError, NotImplementedError):
                print("  [INFO] Filesystem does not permit symlink creation without elevated privileges; skipping live symlink swap check.")

            if symlink_supported:
                # Set data root to symlink path
                engine.config.data_root = current_symlink
                engine.reload_artifacts()
                assert engine.config.data_root == v1_dir.resolve(), f"Expected {v1_dir}, got {engine.config.data_root}"

                # Update symlink to v2
                current_symlink.unlink()
                current_symlink.symlink_to(v2_dir, target_is_directory=True)
                engine.reload_artifacts()
                assert engine.config.data_root == v2_dir.resolve(), f"Expected {v2_dir}, got {engine.config.data_root}"
                print("  [PASS] Symlink dereferencing and resolution verified")

            print("  [PASS] Engine.reload_artifacts() executed successfully and updated references")
        finally:
            engine.close()


def test_api_reload_artifacts_rbac():
    """Verifies that /reload-artifacts endpoints strictly enforce ADMIN authorization."""
    print("Testing API reload-artifacts RBAC barriers and access control...")
    test_admin_key = "secret_master_admin_token_for_tests_789"
    orig_env_key = os.environ.get("TIRESIAS_ADMIN_KEY")
    os.environ["TIRESIAS_ADMIN_KEY"] = test_admin_key

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            db_path = tmp_path / "test_api_reload.db"
            data_path = tmp_path / "data"
            data_path.mkdir(parents=True, exist_ok=True)

            cfg = ServerConfig(custom_db_path=db_path, data_root=data_path)
            engine = Engine(cfg)
            app.state.engine = engine
            client = TestClient(app)

            try:
                endpoints = [
                    "/api/v1/system/reload-artifacts",
                    "/api/v1/admin/system/reload-artifacts",
                ]

                # 1. Anonymous requests -> 401 Unauthorized
                for ep in endpoints:
                    res_anon = client.post(ep)
                    assert res_anon.status_code == 401, f"Expected 401 for anonymous {ep}, got {res_anon.status_code}"

                # 2. Standard user (role 10) -> 403 Forbidden
                u_info, u_token = engine.db.register_local_user("standard_dev", "userpass123")
                user_headers = {"Authorization": f"Bearer {u_token}"}
                for ep in endpoints:
                    res_user = client.post(ep, headers=user_headers)
                    assert res_user.status_code == 403, f"Expected 403 for standard user on {ep}, got {res_user.status_code}"

                # 3. Tester user (role 20) -> 403 Forbidden
                engine.db.set_user_role(u_info["user_id"], int(UserRole.TESTER))
                for ep in endpoints:
                    res_tester = client.post(ep, headers=user_headers)
                    assert res_tester.status_code == 403, f"Expected 403 for tester on {ep}, got {res_tester.status_code}"

                # 4. Master key via X-Admin-Key header -> 200 OK
                x_admin_headers = {"X-Admin-Key": test_admin_key}
                for ep in endpoints:
                    res_admin_x = client.post(ep, headers=x_admin_headers)
                    assert res_admin_x.status_code == 200, f"Expected 200 with X-Admin-Key on {ep}, got {res_admin_x.status_code}: {res_admin_x.text}"
                    data = res_admin_x.json()
                    assert data.get("status") == "ok"
                    assert data.get("message") == "Artifacts reloaded successfully"

                # 5. Master key via Bearer token -> 200 OK
                bearer_admin_headers = {"Authorization": f"Bearer {test_admin_key}"}
                for ep in endpoints:
                    res_bearer = client.post(ep, headers=bearer_admin_headers)
                    assert res_bearer.status_code == 200, f"Expected 200 with Bearer master key on {ep}, got {res_bearer.status_code}"

                # 6. Admin user with issued token (role 100) -> 200 OK
                adm_user, adm_token = engine.db.create_or_update_admin("big_boss", "bosspassword123")
                boss_headers = {"Authorization": f"Bearer {adm_token}"}
                for ep in endpoints:
                    res_boss = client.post(ep, headers=boss_headers)
                    assert res_boss.status_code == 200, f"Expected 200 for role 100 admin on {ep}, got {res_boss.status_code}"

                print("  [PASS] Reload endpoints RBAC barriers and Admin authentication verified 100%")
            finally:
                engine.close()
    finally:
        if orig_env_key is not None:
            os.environ["TIRESIAS_ADMIN_KEY"] = orig_env_key
        else:
            os.environ.pop("TIRESIAS_ADMIN_KEY", None)


def test_updater_api_helpers():
    """Tests set_maintenance_api and reload_artifacts_api from tools.updater."""
    print("Testing updater helper functions (set_maintenance_api, reload_artifacts_api)...")
    test_key = "updater_admin_key_test_54321"

    # 1. Test set_maintenance_api
    with mock.patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = mock.MagicMock()
        mock_resp.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        # Call enable
        ok = set_maintenance_api("http://127.0.0.1:8000", enable=True, admin_key=test_key)
        assert ok is True
        assert mock_urlopen.call_count == 1
        req = mock_urlopen.call_args[0][0]
        assert req.get_full_url() == "http://127.0.0.1:8000/api/v1/system/maintenance/enable"
        assert req.headers.get("X-admin-key") == test_key
        assert req.headers.get("Authorization") == f"Bearer {test_key}"

        # Call disable
        ok_dis = set_maintenance_api("http://127.0.0.1:8000", enable=False, admin_key=test_key)
        assert ok_dis is True
        req_dis = mock_urlopen.call_args[0][0]
        assert req_dis.get_full_url() == "http://127.0.0.1:8000/api/v1/system/maintenance/disable"

    # 2. Test reload_artifacts_api
    with mock.patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = mock.MagicMock()
        mock_resp.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        ok_reload = reload_artifacts_api("http://127.0.0.1:8000", admin_key=test_key)
        assert ok_reload is True
        assert mock_urlopen.call_count == 1
        req_rel = mock_urlopen.call_args[0][0]
        assert req_rel.get_full_url() == "http://127.0.0.1:8000/api/v1/system/reload-artifacts"
        assert req_rel.headers.get("X-admin-key") == test_key
        assert req_rel.headers.get("Authorization") == f"Bearer {test_key}"

    # 3. Test failure / offline handling
    with mock.patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
        ok_fail = set_maintenance_api("http://127.0.0.1:8000", enable=True, admin_key=test_key)
        assert ok_fail is False

        ok_fail_rel = reload_artifacts_api("http://127.0.0.1:8000", admin_key=test_key)
        assert ok_fail_rel is False

    print("  [PASS] updater API helpers verified with proper auth headers and error handling")


def test_maintenance_mode_and_recommendation_protection():
    """Verifies that maintenance mode properly returns 503 for recommendations."""
    print("Testing maintenance mode operation and 503 rejection during update...")
    test_key = "admin_key_maintenance_check"
    orig_env = os.environ.get("TIRESIAS_ADMIN_KEY")
    os.environ["TIRESIAS_ADMIN_KEY"] = test_key

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            db_path = tmp_path / "test_maint.db"
            cfg = ServerConfig(custom_db_path=db_path, data_root=tmp_path / "data")
            engine = Engine(cfg)
            app.state.engine = engine
            client = TestClient(app)

            try:
                headers = {"X-Admin-Key": test_key}

                # 1. Enable maintenance mode via API
                res_en = client.post("/api/v1/system/maintenance/enable", headers=headers)
                assert res_en.status_code == 200, res_en.text
                assert engine.is_maintenance_mode is True

                # 2. Recommendations return 503 Service Unavailable
                res_rec = client.get("/api/v1/recommend/feed")
                assert res_rec.status_code == 503, f"Expected 503, got {res_rec.status_code}"
                assert "Database update in progress" in res_rec.text

                res_sim = client.get("/api/v1/recommend/similar?post_id=100")
                assert res_sim.status_code == 503, f"Expected 503, got {res_sim.status_code}"

                # 3. Disable maintenance mode via API
                res_dis = client.post("/api/v1/system/maintenance/disable", headers=headers)
                assert res_dis.status_code == 200
                assert engine.is_maintenance_mode is False

                # 4. Recommendations no longer return 503
                res_rec2 = client.get("/api/v1/recommend/feed")
                assert res_rec2.status_code != 503, f"Unexpected 503 after maintenance disabled: {res_rec2.status_code}"

                print("  [PASS] Maintenance mode correctly blocks recommendation queries with 503")
            finally:
                engine.close()
    finally:
        if orig_env is not None:
            os.environ["TIRESIAS_ADMIN_KEY"] = orig_env
        else:
            os.environ.pop("TIRESIAS_ADMIN_KEY", None)


def test_sync_data_cli_parser():
    """Verifies sync_data.py CLI parser supports --atomic and --keep-versions."""
    print("Testing sync_data.py CLI arguments parser for atomic update support...")
    from tiresias_server.scripts.sync_data import main

    # Inspect ArgumentParser through sys.argv
    test_argv = ["sync_data.py", "--atomic", "--keep-versions", "3", "--verify-only"]
    with mock.patch("sys.argv", test_argv):
        with mock.patch("tiresias_server.scripts.sync_data.run_remote_verification", return_value=0):
            res = main()
            assert res == 0
    print("  [PASS] sync_data.py --atomic and --keep-versions flags verified")


def run_all():
    print("=" * 65)
    print("RUNNING TIRESIAS STAGE 5 HOT-RELOAD & REMOTE UPDATE TEST SUITE")
    print("=" * 65)
    test_engine_reload_artifacts_direct()
    test_api_reload_artifacts_rbac()
    test_updater_api_helpers()
    test_maintenance_mode_and_recommendation_protection()
    test_sync_data_cli_parser()
    print("=" * 65)
    print("ALL STAGE 5 HOT-RELOAD & UPDATE TESTS PASSED! (100% OK)")
    print("=" * 65)


if __name__ == "__main__":
    run_all()
