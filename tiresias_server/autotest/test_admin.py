#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

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


def test_admin_rbac_barriers():
    """Verifies that non-admin requests to /api/v1/admin/* return 401 or 403."""
    print("Testing admin endpoint RBAC barriers (401 / 403)...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_rbac.db"
        cfg = ServerConfig(custom_db_path=db_path)
        engine = Engine(cfg)
        app.state.engine = engine
        client = TestClient(app)

        try:
            # 1. Anonymous requests -> 401 Unauthorized
            for endpoint in ["/users", "/invites", "/system/stats"]:
                res = client.get(f"/api/v1/admin{endpoint}")
                assert res.status_code == 401, f"Expected 401 for anonymous {endpoint}, got {res.status_code}"

            res_post = client.post("/api/v1/admin/system/maintenance", json={"maintenance": True})
            assert res_post.status_code == 401, f"Expected 401 for anonymous maintenance post, got {res_post.status_code}"

            # 2. Register regular user (role 10)
            u_dict, u_token = engine.db.register_local_user(username="standard_user", password="password123")
            user_headers = {"Authorization": f"Bearer {u_token}"}

            for endpoint in ["/users", "/invites", "/system/stats"]:
                res = client.get(f"/api/v1/admin{endpoint}", headers=user_headers)
                assert res.status_code == 403, f"Expected 403 for user role on {endpoint}, got {res.status_code}"

            res_maint = client.post("/api/v1/admin/system/maintenance", json={"maintenance": True}, headers=user_headers)
            assert res_maint.status_code == 403, f"Expected 403 for user role on maintenance, got {res_maint.status_code}"

            # 3. Elevate user to TESTER (role 20) -> still 403 (needs ADMIN = 100)
            engine.db.set_user_role(u_dict["user_id"], int(UserRole.TESTER))
            res_tester = client.get("/api/v1/admin/users", headers=user_headers)
            assert res_tester.status_code == 403, f"Expected 403 for tester role, got {res_tester.status_code}"

            print("  [PASS] Non-admin access control strictly enforced (401 & 403)")
        finally:
            engine.db.close()


def test_admin_master_key_access():
    """Verifies admin access using X-Admin-Key master key and elevated token."""
    print("Testing admin access via X-Admin-Key and elevated token...")
    test_key = "test_master_admin_secret_key_12345"
    orig_key = os.environ.get("TIRESIAS_ADMIN_KEY")
    os.environ["TIRESIAS_ADMIN_KEY"] = test_key

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test_master.db"
            cfg = ServerConfig(custom_db_path=db_path)
            engine = Engine(cfg)
            app.state.engine = engine
            client = TestClient(app)

            try:
                # 1. Master key via X-Admin-Key header
                headers_x = {"X-Admin-Key": test_key}
                res_x = client.get("/api/v1/admin/users", headers=headers_x)
                assert res_x.status_code == 200, f"Expected 200 via X-Admin-Key, got {res_x.status_code}: {res_x.text}"

                # 2. Master key via Authorization: Bearer <key>
                headers_bearer = {"Authorization": f"Bearer {test_key}"}
                res_b = client.get("/api/v1/admin/users", headers=headers_bearer)
                assert res_b.status_code == 200, f"Expected 200 via Bearer master key, got {res_b.status_code}"

                # 3. Admin user with issued token (role 100)
                admin_user, admin_token = engine.db.create_or_update_admin(
                    username="boss_admin",
                    password="adminpassword123",
                )
                admin_headers = {"Authorization": f"Bearer {admin_token}"}
                res_adm = client.get("/api/v1/admin/users", headers=admin_headers)
                assert res_adm.status_code == 200, f"Expected 200 via admin user token, got {res_adm.status_code}"

                print("  [PASS] Master key & admin token access OK")
            finally:
                engine.db.close()
    finally:
        if orig_key is not None:
            os.environ["TIRESIAS_ADMIN_KEY"] = orig_key
        else:
            os.environ.pop("TIRESIAS_ADMIN_KEY", None)


def test_user_management_api():
    """Tests user listing, pagination, search, role update, and password reset."""
    print("Testing user management API (listing, pagination, search, role, password)...")
    test_key = "test_user_mgmt_admin_key"
    orig_key = os.environ.get("TIRESIAS_ADMIN_KEY")
    os.environ["TIRESIAS_ADMIN_KEY"] = test_key

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test_mgmt.db"
            cfg = ServerConfig(custom_db_path=db_path)
            engine = Engine(cfg)
            app.state.engine = engine
            client = TestClient(app)
            headers = {"X-Admin-Key": test_key}

            try:
                # Seed 5 users with distinct names and roles
                engine.db.register_local_user("alice_wonder", "alice_pwd123", display_name="Alice in Wonderland")
                engine.db.register_local_user("bob_builder", "bob_pwd123", display_name="Bob the Builder")
                engine.db.register_local_user("charlie_tester", "charlie_pwd123", display_name="Charlie Tester")
                engine.db.set_user_role("local:charlie_tester", int(UserRole.TESTER))
                engine.db.register_local_user("david_mod", "david_pwd123", display_name="David Moderator")
                engine.db.set_user_role("local:david_mod", int(UserRole.MODERATOR))

                # 1. User listing & total count
                res_all = client.get("/api/v1/admin/users", headers=headers)
                assert res_all.status_code == 200
                data_all = res_all.json()
                assert data_all["total"] >= 4
                assert len(data_all["users"]) >= 4

                # 2. Pagination (page_size = 2)
                res_p1 = client.get("/api/v1/admin/users?page=1&page_size=2", headers=headers)
                assert res_p1.status_code == 200
                data_p1 = res_p1.json()
                assert len(data_p1["users"]) == 2
                assert data_p1["page"] == 1
                assert data_p1["page_size"] == 2

                res_p2 = client.get("/api/v1/admin/users?page=2&page_size=2", headers=headers)
                assert res_p2.status_code == 200
                data_p2 = res_p2.json()
                assert len(data_p2["users"]) >= 1

                # Ensure page 1 and page 2 contain different users
                p1_ids = {u["user_id"] for u in data_p1["users"]}
                p2_ids = {u["user_id"] for u in data_p2["users"]}
                assert p1_ids.isdisjoint(p2_ids), "Paginated pages must not overlap"

                # 3. Search filter
                res_search = client.get("/api/v1/admin/users?search=alice", headers=headers)
                assert res_search.status_code == 200
                data_search = res_search.json()
                assert data_search["total"] == 1
                assert data_search["users"][0]["username"] == "alice_wonder"

                # 4. Role filter
                res_rf = client.get(f"/api/v1/admin/users?role={int(UserRole.TESTER)}", headers=headers)
                assert res_rf.status_code == 200
                data_rf = res_rf.json()
                assert all(u["role"] == int(UserRole.TESTER) for u in data_rf["users"])

                # 5. Role update
                res_role = client.post(
                    "/api/v1/admin/users/local:alice_wonder/role",
                    json={"role": int(UserRole.TESTER)},
                    headers=headers,
                )
                assert res_role.status_code == 200, res_role.text
                assert res_role.json()["role"] == int(UserRole.TESTER)
                account_alice = engine.db.get_user_account_info("local:alice_wonder")
                assert account_alice["role"] == int(UserRole.TESTER)

                # 6. Password reset
                res_pw = client.post(
                    "/api/v1/admin/users/local:alice_wonder/password",
                    json={"password": "new_alice_super_password"},
                    headers=headers,
                )
                assert res_pw.status_code == 200, res_pw.text

                # Test login with old password fails, new password succeeds
                login_old = client.post(
                    "/api/v1/auth/login",
                    json={"username": "alice_wonder", "password": "alice_pwd123"},
                )
                assert login_old.status_code == 401, "Old password must be rejected after reset"

                login_new = client.post(
                    "/api/v1/auth/login",
                    json={"username": "alice_wonder", "password": "new_alice_super_password"},
                )
                assert login_new.status_code == 200, "New password must authenticate successfully"

                # 7. Non-existent user updates return 404
                res_not_found = client.post(
                    "/api/v1/admin/users/non_existent_guy/role",
                    json={"role": 10},
                    headers=headers,
                )
                assert res_not_found.status_code == 404

                print("  [PASS] User listing, pagination, search, role update, and password reset OK")
            finally:
                engine.db.close()
    finally:
        if orig_key is not None:
            os.environ["TIRESIAS_ADMIN_KEY"] = orig_key
        else:
            os.environ.pop("TIRESIAS_ADMIN_KEY", None)


def test_invite_whitelist_api():
    """Tests invite whitelist listing, addition, and deletion via Admin API."""
    print("Testing invite whitelist management API...")
    test_key = "test_invite_admin_key"
    orig_key = os.environ.get("TIRESIAS_ADMIN_KEY")
    os.environ["TIRESIAS_ADMIN_KEY"] = test_key

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test_invites.db"
            cfg = ServerConfig(custom_db_path=db_path)
            engine = Engine(cfg)
            app.state.engine = engine
            client = TestClient(app)
            headers = {"X-Admin-Key": test_key}

            try:
                # 1. Initially empty
                res_list0 = client.get("/api/v1/admin/invites", headers=headers)
                assert res_list0.status_code == 200
                assert len(res_list0.json()) == 0

                # 2. Add invited users
                res_add1 = client.post(
                    "/api/v1/admin/invites",
                    json={"site_user_id": 10001, "username": "StarFox", "note": "Alpha tester"},
                    headers=headers,
                )
                assert res_add1.status_code == 200, res_add1.text
                assert res_add1.json()["site_user_id"] == 10001

                res_add2 = client.post(
                    "/api/v1/admin/invites",
                    json={"site_user_id": 10002, "username": "Falco", "note": "VIP Tester"},
                    headers=headers,
                )
                assert res_add2.status_code == 200

                # 3. List invites
                res_list1 = client.get("/api/v1/admin/invites", headers=headers)
                assert res_list1.status_code == 200
                invites = res_list1.json()
                assert len(invites) == 2
                ids = {inv["site_user_id"] for inv in invites}
                assert ids == {10001, 10002}

                # 4. Remove an invite
                res_del = client.delete("/api/v1/admin/invites/10001", headers=headers)
                assert res_del.status_code == 200
                assert engine.db.is_site_user_invited(10001) is False
                assert engine.db.is_site_user_invited(10002) is True

                # 5. Delete non-existent invite returns 404
                res_del_nf = client.delete("/api/v1/admin/invites/99999", headers=headers)
                assert res_del_nf.status_code == 404

                print("  [PASS] Invite whitelist add, list, delete OK")
            finally:
                engine.db.close()
    finally:
        if orig_key is not None:
            os.environ["TIRESIAS_ADMIN_KEY"] = orig_key
        else:
            os.environ.pop("TIRESIAS_ADMIN_KEY", None)


def test_system_telemetry_and_maintenance():
    """Tests system telemetry stats and maintenance mode toggle."""
    print("Testing system stats telemetry and maintenance mode toggle...")
    test_key = "test_telem_admin_key"
    orig_key = os.environ.get("TIRESIAS_ADMIN_KEY")
    os.environ["TIRESIAS_ADMIN_KEY"] = test_key

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test_telem.db"
            cfg = ServerConfig(custom_db_path=db_path)
            engine = Engine(cfg)
            app.state.engine = engine
            client = TestClient(app)
            headers = {"X-Admin-Key": test_key}

            try:
                # 1. Telemetry query
                res_stats = client.get("/api/v1/admin/system/stats", headers=headers)
                assert res_stats.status_code == 200, res_stats.text
                stats = res_stats.json()

                # Verify Host CPU metrics
                assert "cpu_percent" in stats
                assert stats["cpu_count"] >= 1
                assert stats["host_cpu"]["cpu_count"] >= 1

                # Verify Host RAM metrics
                assert stats["memory_total_mb"] > 0
                assert stats["memory_percent"] >= 0
                assert stats["host_memory"]["total_mb"] > 0

                # Verify Process RSS
                assert stats["process_rss_mb"] > 0
                assert stats["process_memory"]["rss_mb"] > 0

                # Verify Disk metrics
                assert stats["disk_total_gb"] > 0
                assert "disk_free_gb" in stats
                assert stats["disk"]["total_gb"] > 0

                # Verify DB file sizes & DB row metrics
                assert "db_user_size_mb" in stats
                assert "db_telemetry_size_mb" in stats
                assert "db_rows" in stats
                assert "users" in stats["db_rows"]
                assert "user_feedback" in stats["db_rows"]
                assert "user_boards" in stats["db_rows"]
                assert "allowed_invites" in stats["db_rows"]
                assert "auth_tokens" in stats["db_rows"]

                # Verify Engine stats
                eng = stats["engine"]
                assert eng["profile"] == cfg.profile
                assert eng["status"] == "online"
                assert eng["maintenance"] is False
                assert "uptime_seconds" in eng
                assert "candidate_budget" in eng

                # 2. Toggle Maintenance Mode ON
                res_maint_on = client.post(
                    "/api/v1/admin/system/maintenance",
                    json={"maintenance": True},
                    headers=headers,
                )
                assert res_maint_on.status_code == 200
                assert res_maint_on.json()["maintenance"] is True
                assert engine.is_maintenance_mode is True

                # Verify stats reflect maintenance mode
                stats_m = client.get("/api/v1/admin/system/stats", headers=headers).json()
                assert stats_m["engine"]["maintenance"] is True
                assert stats_m["engine"]["status"] == "maintenance"

                # 3. Toggle Maintenance Mode OFF
                res_maint_off = client.post(
                    "/api/v1/admin/system/maintenance",
                    json={"maintenance": False},
                    headers=headers,
                )
                assert res_maint_off.status_code == 200
                assert res_maint_off.json()["maintenance"] is False
                assert engine.is_maintenance_mode is False

                print("  [PASS] System stats telemetry & maintenance mode toggle OK")
            finally:
                engine.db.close()
    finally:
        if orig_key is not None:
            os.environ["TIRESIAS_ADMIN_KEY"] = orig_key
        else:
            os.environ.pop("TIRESIAS_ADMIN_KEY", None)


def test_user_deletion_cascading():
    """Tests user deletion cascading across boards, tokens, feedback, and settings."""
    print("Testing user deletion cascading across associated tables...")
    test_key = "test_del_admin_key"
    orig_key = os.environ.get("TIRESIAS_ADMIN_KEY")
    os.environ["TIRESIAS_ADMIN_KEY"] = test_key

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test_del.db"
            cfg = ServerConfig(custom_db_path=db_path)
            engine = Engine(cfg)
            app.state.engine = engine
            client = TestClient(app)
            headers = {"X-Admin-Key": test_key}

            try:
                # Register user
                u_info, tok = engine.db.register_local_user("victim_user", "password123")
                uid = u_info["user_id"]

                # Add associated records
                engine.db.record_feedback(uid, 5001, "like")
                engine.db.record_feedback(uid, 5002, "dislike")
                engine.db.record_seen_batch(uid, [5001, 5002, 5003])
                engine.db.add_tag_blacklist(uid, [12, 34])
                engine.db.increment_user_tag_likes(uid, [77, 88])
                engine.db.set_user_settings(uid, {"theme": "dark", "nsfw": "true"})

                # Create board and board posts
                board = engine.db.create_board(uid, "Victim Board", "Board to be deleted")
                bid = board["board_id"]
                engine.db.add_posts_to_board(bid, [5001, 5002])

                # Verify data exists
                assert engine.db.get_user_account_info(uid) is not None
                assert len(engine.db.get_user_feedback_history(uid)) == 2
                assert len(engine.db.get_user_seen_posts(uid)) == 3
                assert len(engine.db.get_user_tag_blacklist(uid)) == 2
                assert len(engine.db.list_user_boards(uid)) == 1

                # Delete user via Admin API
                res_del = client.delete(f"/api/v1/admin/users/{uid}", headers=headers)
                assert res_del.status_code == 200, res_del.text

                # Verify cascading deletion
                assert engine.db.get_user_account_info(uid) is None
                assert engine.db.get_user_feedback_count(uid) == 0
                assert len(engine.db.get_user_seen_posts(uid)) == 0
                assert len(engine.db.get_user_tag_blacklist(uid)) == 0
                assert len(engine.db.get_user_tag_likes(uid)) == 0
                assert len(engine.db.get_user_settings(uid)) == 0
                assert len(engine.db.list_user_boards(uid)) == 0
                assert len(engine.db.get_board_post_records(bid)) == 0

                # Repeated delete returns 404
                res_del2 = client.delete(f"/api/v1/admin/users/{uid}", headers=headers)
                assert res_del2.status_code == 404

                print("  [PASS] User cascading deletion verified cleanly")
            finally:
                engine.db.close()
    finally:
        if orig_key is not None:
            os.environ["TIRESIAS_ADMIN_KEY"] = orig_key
        else:
            os.environ.pop("TIRESIAS_ADMIN_KEY", None)


def test_web_admin_ui_serving():
    """Tests serving the standalone admin web dashboard at /admin and /admin/."""
    print("Testing Web Admin UI endpoint serving (/admin)...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_ui.db"
        cfg = ServerConfig(custom_db_path=db_path)
        engine = Engine(cfg)
        app.state.engine = engine
        client = TestClient(app)

        try:
            # 1. /admin
            res1 = client.get("/admin")
            assert res1.status_code == 200, f"Expected 200, got {res1.status_code}"
            assert "text/html" in res1.headers.get("content-type", "")
            assert "Tiresias Admin Console" in res1.text
            assert "System Status & Live Telemetry" in res1.text

            # 2. /admin/
            res2 = client.get("/admin/")
            assert res2.status_code == 200, f"Expected 200, got {res2.status_code}"
            assert "text/html" in res2.headers.get("content-type", "")
            assert "TIRESIAS" in res2.text

            print("  [PASS] /admin and /admin/ web dashboard served successfully")
        finally:
            engine.db.close()


def run_all():
    print("=" * 65)
    print("RUNNING TIRESIAS STAGE 4 ADMIN & DASHBOARD TEST SUITE")
    print("=" * 65)
    test_admin_rbac_barriers()
    test_admin_master_key_access()
    test_user_management_api()
    test_invite_whitelist_api()
    test_system_telemetry_and_maintenance()
    test_user_deletion_cascading()
    test_web_admin_ui_serving()
    print("=" * 65)
    print("ALL ADMIN PLATFORM & DASHBOARD TESTS PASSED! (100% OK)")
    print("=" * 65)


if __name__ == "__main__":
    run_all()
