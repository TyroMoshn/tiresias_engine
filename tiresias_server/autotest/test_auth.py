#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Setup paths
HERE = Path(__file__).resolve().parent
SERVER_DIR = HERE.parent
ROOT_DIR = SERVER_DIR.parent
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient

from app.config import ServerConfig
from app.core.auth import (
    UserRole,
    hash_password,
    verify_password,
    generate_secure_token,
    hash_token,
    get_admin_master_key,
    verify_user_access,
)
from app.core.rate_limit import auth_rate_limiter
from app.core.engine import Engine
from app.db.database import Database
from app.main import app
from app import cli


def test_auth_crypto_primitives():
    print("Testing auth cryptographic primitives...")
    pwd = "secret_password_123"
    h1, s1 = hash_password(pwd)
    assert h1 and s1, "Hash and salt must be non-empty"
    assert verify_password(pwd, h1, s1) is True, "Password verification must succeed"
    assert verify_password("wrong_password", h1, s1) is False, "Wrong password must fail"

    # Salting uniqueness
    h2, s2 = hash_password(pwd)
    assert s1 != s2, "Each salt must be unique"
    assert h1 != h2, "Hashes with different salts must differ"

    tok = generate_secure_token()
    assert len(tok) == 64, "Bearer token must be 64 hex characters"
    th = hash_token(tok)
    assert len(th) == 64, "SHA-256 hash must be 64 hex characters"
    print("  [PASS] PBKDF2 hashing, salting, and token generation OK")


def test_ephemeral_admin_key():
    print("Testing ephemeral admin master key generation when not configured...")
    # 1. Without TIRESIAS_ADMIN_KEY
    orig_key = os.environ.pop("TIRESIAS_ADMIN_KEY", None)
    try:
        # Reset cached key for testing
        import app.core.auth as auth_mod
        auth_mod._IN_MEMORY_ADMIN_KEY = None

        key1 = get_admin_master_key()
        assert key1 != "tiresias_dev_admin_key_2026", "Hardcoded dev master key must NOT be used"
        assert len(key1) == 48, f"Expected 48 hex chars (24 bytes), got {len(key1)}"

        # Repeated calls return the same in-memory key
        key2 = get_admin_master_key()
        assert key1 == key2, "Ephemeral admin master key must persist in process memory"

        # 2. When TIRESIAS_ADMIN_KEY is explicitly set
        os.environ["TIRESIAS_ADMIN_KEY"] = "my_custom_production_key_xyz"
        assert get_admin_master_key() == "my_custom_production_key_xyz"
    finally:
        if orig_key is not None:
            os.environ["TIRESIAS_ADMIN_KEY"] = orig_key
        else:
            os.environ.pop("TIRESIAS_ADMIN_KEY", None)

    print("  [PASS] Master admin key dynamic generation & isolation OK")


def test_database_auth_layer():
    print("Testing Database auth, seeds, and sandbox methods on isolated SQLite...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_auth_clean.db"

        # 1. Clean DB without TIRESIAS_DEV_SEEDS: Demo accounts must NOT exist
        orig_seeds = os.environ.pop("TIRESIAS_DEV_SEEDS", None)
        try:
            with Database(db_path) as db_clean:
                assert db_clean.get_user_account_info("user_demo") is None, "user_demo must not exist by default"
                assert db_clean.get_user_account_info("tester_demo") is None, "tester_demo must not exist by default"
                assert db_clean.get_user_account_info("admin_demo") is None, "admin_demo must not exist by default"
        finally:
            if orig_seeds is not None:
                os.environ["TIRESIAS_DEV_SEEDS"] = orig_seeds

        # 2. DB with TIRESIAS_DEV_SEEDS=true: Demo accounts must be created
        os.environ["TIRESIAS_DEV_SEEDS"] = "true"
        db_seeded_path = Path(tmp_dir) / "test_auth_seeded.db"
        with Database(db_seeded_path) as db:
            u_demo = db.get_user_account_info("user_demo")
            assert u_demo is not None, "user_demo must be seeded when TIRESIAS_DEV_SEEDS=true"
            assert u_demo["role"] == int(UserRole.USER)
            assert u_demo["has_password"] is True

            t_demo = db.get_user_account_info("tester_demo")
            assert t_demo is not None, "tester_demo must be seeded when TIRESIAS_DEV_SEEDS=true"
            assert t_demo["role"] == int(UserRole.TESTER)

            a_demo = db.get_user_account_info("admin_demo")
            assert a_demo is not None, "admin_demo must be seeded when TIRESIAS_DEV_SEEDS=true"
            assert a_demo["role"] == int(UserRole.ADMIN)

            # 3. Test e621 seamless onboarding
            user_e621, token_e621 = db.create_or_get_e621_user(
                site_user_id=54321,
                username="FenrirFox",
                device_info="Firefox 130",
            )
            assert user_e621["user_id"] == "e621:54321"
            assert user_e621["username"] == "FenrirFox"
            assert user_e621["site_source"] == "e621"
            assert user_e621["role"] == int(UserRole.USER)
            assert len(token_e621) == 64

            # Validate issued token
            val_user = db.validate_token(token_e621)
            assert val_user is not None
            assert val_user["user_id"] == "e621:54321"

            # 4. Test local registration with namespacing
            user_loc, token_loc = db.register_local_user(
                username="FenrirFox",  # Same display username!
                password="mypassword",
            )
            assert user_loc["user_id"] == "local:fenrirfox"
            assert user_loc["site_source"] == "direct"
            assert user_loc["user_id"] != user_e621["user_id"], "Namespaces must isolate accounts!"

            # Re-registering same local username must raise ValueError
            try:
                db.register_local_user("FenrirFox", "anotherpass")
                assert False, "Duplicate local username must fail"
            except ValueError:
                pass

            # Prefixed 'e621:' username must be rejected
            try:
                db.register_local_user("e621:fake", "somepass")
                assert False, "e621: prefix in local registration must be rejected"
            except ValueError:
                pass

            # 5. Test authentication
            auth_ok = db.authenticate_local_user("FenrirFox", "mypassword")
            assert auth_ok is not None
            assert auth_ok[0]["user_id"] == "local:fenrirfox"

            auth_fail = db.authenticate_local_user("FenrirFox", "wrongpass")
            assert auth_fail is None

            # 6. Test token revocation (logout)
            assert db.revoke_token(token_loc) is True
            assert db.validate_token(token_loc) is None, "Revoked token must not validate"

            # 7. Test tester sandbox profiles
            sb1 = db.create_sandbox_profile(tester_id="tester_demo", profile_name="Furry Safe")
            assert sb1["user_id"] == "test:tester_demo:furry_safe"
            assert sb1["owner_id"] == "tester_demo"

            sb_list = db.list_tester_sandbox_profiles("tester_demo")
            assert len(sb_list) == 1
            assert sb_list[0]["user_id"] == "test:tester_demo:furry_safe"

            # Another tester cannot see or delete this profile
            assert db.delete_sandbox_profile(tester_id="other_tester", profile_id=sb1["user_id"]) is False
            assert db.delete_sandbox_profile(tester_id="tester_demo", profile_id=sb1["user_id"]) is True
            assert len(db.list_tester_sandbox_profiles("tester_demo")) == 0

            # 8. Test linking site session to local user
            assert db.link_site_session(user_id=user_loc["user_id"], site_user_id=88888, site_username="FenrirLinked") is True
            info_linked = db.get_user_account_info(user_loc["user_id"])
            assert info_linked["site_user_id"] == 88888
            assert info_linked["site_source"] == "e621"

    print("  [PASS] Database auth, isolation, and sandbox operations OK")


def test_api_rbac_and_endpoints():
    print("Testing FastAPI API RBAC barriers & routes...")
    os.environ["TIRESIAS_DEV_SEEDS"] = "true"
    auth_rate_limiter.reset()

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_api_auth.db"
        cfg = ServerConfig(custom_db_path=db_path)
        engine = Engine(cfg)
        app.state.engine = engine
        client = TestClient(app)
        try:
            # 1. Unauthenticated request to /auth/me -> 401
            res = client.get("/api/v1/auth/me")
            assert res.status_code == 401, f"Expected 401, got {res.status_code}"

            # 2. Login as user_demo
            res_login = client.post("/api/v1/auth/login", json={"username": "user_demo", "password": "user123"})
            assert res_login.status_code == 200, res_login.text
            user_token = res_login.json()["token"]
            user_headers = {"Authorization": f"Bearer {user_token}"}

            # Check /auth/me for user_demo
            me_res = client.get("/api/v1/auth/me", headers=user_headers)
            assert me_res.status_code == 200
            assert me_res.json()["user_id"] == "user_demo"
            assert me_res.json()["role"] == 10

            # 3. Regular user attempting ADMIN operations -> 403 Forbidden!
            res_wipe = client.post("/api/v1/user/reset", json={"reset_all": True}, headers=user_headers)
            assert res_wipe.status_code == 403, f"Expected 403 Forbidden on reset_all, got {res_wipe.status_code}"

            res_maint = client.post("/api/v1/system/maintenance/enable", headers=user_headers)
            assert res_maint.status_code == 403, f"Expected 403 Forbidden on maintenance, got {res_maint.status_code}"

            res_diag = client.post("/api/v1/system/diagnostics/run", headers=user_headers)
            assert res_diag.status_code == 403, f"Expected 403 Forbidden on diagnostics, got {res_diag.status_code}"

            res_supp = client.post("/api/v1/system/suppression/reload", headers=user_headers)
            assert res_supp.status_code == 403, f"Expected 403 Forbidden on suppression reload, got {res_supp.status_code}"

            # 4. Regular user attempting to reset ANOTHER user's profile -> 403 Forbidden!
            res_other = client.post("/api/v1/user/reset", json={"user_id": "admin_demo"}, headers=user_headers)
            assert res_other.status_code == 403, f"Expected 403 Forbidden on resetting another user, got {res_other.status_code}"

            # 5. Regular user resetting their OWN profile -> 200 OK
            res_own = client.post("/api/v1/user/reset", json={"user_id": "user_demo"}, headers=user_headers)
            assert res_own.status_code == 200, f"Expected 200 OK on own profile reset, got {res_own.status_code}"

            # 6. Login as tester_demo
            res_tlogin = client.post("/api/v1/auth/login", json={"username": "tester_demo", "password": "tester123"})
            assert res_tlogin.status_code == 200
            tester_token = res_tlogin.json()["token"]
            tester_headers = {"Authorization": f"Bearer {tester_token}"}

            # Tester creates sandbox profile
            res_sb = client.post("/api/v1/auth/sandbox/create", json={"profile_name": "alpha_check"}, headers=tester_headers)
            assert res_sb.status_code == 200, res_sb.text
            sb_id = res_sb.json()["profile"]["user_id"]
            assert sb_id == "test:tester_demo:alpha_check"

            # Tester can reset their own sandbox profile -> 200 OK
            res_sb_reset = client.post("/api/v1/user/reset", json={"user_id": sb_id}, headers=tester_headers)
            assert res_sb_reset.status_code == 200

            # Tester cannot override archetype of another user's profile -> 403 Forbidden
            res_over_other = client.post("/api/v1/user/archetype/override", json={"user_id": "user_demo", "locked_archetype": 10}, headers=tester_headers)
            assert res_over_other.status_code == 403

            # Tester CAN override archetype of their own sandbox profile -> 200 OK
            res_over_sb = client.post("/api/v1/user/archetype/override", json={"user_id": sb_id, "locked_archetype": 10}, headers=tester_headers)
            assert res_over_sb.status_code == 200
            assert res_over_sb.json()["effective_archetype_id"] == 10

            # 7. Login as admin_demo
            res_alogin = client.post("/api/v1/auth/login", json={"username": "admin_demo", "password": "admin123"})
            assert res_alogin.status_code == 200
            admin_token = res_alogin.json()["token"]
            admin_headers = {"Authorization": f"Bearer {admin_token}"}

            # Admin CAN enable/disable maintenance mode -> 200 OK
            res_m1 = client.post("/api/v1/system/maintenance/enable", headers=admin_headers)
            assert res_m1.status_code == 200
            res_m2 = client.post("/api/v1/system/maintenance/disable", headers=admin_headers)
            assert res_m2.status_code == 200

            # 8. Seamless e621 handshake
            auth_rate_limiter.reset()
            res_e6 = client.post("/api/v1/auth/session-handshake", json={
                "site": "e621",
                "site_user_id": 99999,
                "username": "E621Tester",
            })
            assert res_e6.status_code == 200
            e6_data = res_e6.json()
            assert e6_data["user_id"] == "e621:99999"
            assert e6_data["token"] is not None
            e6_headers = {"Authorization": f"Bearer {e6_data['token']}"}

            # 9. Test set_password with new username on session user
            res_pw = client.post("/api/v1/auth/password", json={"new_password": "supersecretpassword", "new_username": "MyCoolNick"}, headers=e6_headers)
            assert res_pw.status_code == 200, res_pw.text
            # Now can log in using new username and password
            auth_rate_limiter.reset()
            res_e6_login = client.post("/api/v1/auth/login", json={"username": "MyCoolNick", "password": "supersecretpassword"})
            assert res_e6_login.status_code == 200
            assert res_e6_login.json()["user_id"] == "e621:99999"

            # 10. Test link-session on user_demo
            res_link = client.post("/api/v1/auth/link-session", json={"site_user_id": 123456, "site_username": "DemoOnSite"}, headers=user_headers)
            assert res_link.status_code == 200, res_link.text
            assert res_link.json()["site_user_id"] == 123456
            assert res_link.json()["site_source"] == "e621"
        finally:
            engine.db.close()

    print("  [PASS] API RBAC barriers, tester sandbox, and admin authorization OK")


def test_protected_user_profile_and_feedback():
    print("Testing protection on /{user_id}/profile and /{user_id}/feedback...")
    os.environ["TIRESIAS_DEV_SEEDS"] = "true"
    auth_rate_limiter.reset()

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_user_profile_sec.db"
        cfg = ServerConfig(custom_db_path=db_path)
        engine = Engine(cfg)
        app.state.engine = engine
        client = TestClient(app)

        try:
            # 1. Unauthenticated requests to profile and feedback must return 401
            res_prof_anon = client.get("/api/v1/user/user_demo/profile")
            assert res_prof_anon.status_code == 401, f"Expected 401 for anonymous profile, got {res_prof_anon.status_code}"

            res_fb_anon = client.get("/api/v1/user/user_demo/feedback")
            assert res_fb_anon.status_code == 401, f"Expected 401 for anonymous feedback, got {res_fb_anon.status_code}"

            # 2. Login as regular user (user_demo)
            res_login = client.post("/api/v1/auth/login", json={"username": "user_demo", "password": "user123"})
            assert res_login.status_code == 200
            user_token = res_login.json()["token"]
            user_headers = {"Authorization": f"Bearer {user_token}"}

            # Login as admin (admin_demo)
            res_alogin = client.post("/api/v1/auth/login", json={"username": "admin_demo", "password": "admin123"})
            assert res_alogin.status_code == 200
            admin_token = res_alogin.json()["token"]
            admin_headers = {"Authorization": f"Bearer {admin_token}"}

            # 3. User accesses their OWN profile and feedback -> 200 OK
            res_own_prof = client.get("/api/v1/user/user_demo/profile", headers=user_headers)
            assert res_own_prof.status_code == 200, f"Expected 200 on own profile, got {res_own_prof.status_code}"

            res_own_fb = client.get("/api/v1/user/user_demo/feedback", headers=user_headers)
            assert res_own_fb.status_code == 200, f"Expected 200 on own feedback, got {res_own_fb.status_code}"

            # 4. User attempts to access ANOTHER user's profile and feedback -> 403 Forbidden!
            res_other_prof = client.get("/api/v1/user/admin_demo/profile", headers=user_headers)
            assert res_other_prof.status_code == 403, f"Expected 403 on other user's profile, got {res_other_prof.status_code}"
            assert "forbidden" in res_other_prof.json()["detail"].lower()

            res_other_fb = client.get("/api/v1/user/admin_demo/feedback", headers=user_headers)
            assert res_other_fb.status_code == 403, f"Expected 403 on other user's feedback, got {res_other_fb.status_code}"
            assert "forbidden" in res_other_fb.json()["detail"].lower()

            # 5. ADMIN accesses another user's profile and feedback -> 200 OK (Auditing rights)
            res_admin_view_prof = client.get("/api/v1/user/user_demo/profile", headers=admin_headers)
            assert res_admin_view_prof.status_code == 200, f"Admin must have audit access to profile, got {res_admin_view_prof.status_code}"

            res_admin_view_fb = client.get("/api/v1/user/user_demo/feedback", headers=admin_headers)
            assert res_admin_view_fb.status_code == 200, f"Admin must have audit access to feedback, got {res_admin_view_fb.status_code}"
        finally:
            engine.db.close()

    print("  [PASS] Profile & feedback endpoint access control OK")


def test_password_lock_and_whitelist():
    print("Testing session handshake Password Lock and Whitelist closed beta...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_whitelist_lock.db"
        cfg = ServerConfig(custom_db_path=db_path)
        cfg.registration_mode = "whitelist"  # Enable Closed Beta Whitelist
        engine = Engine(cfg)
        app.state.engine = engine
        client = TestClient(app)

        auth_rate_limiter.reset()

        try:
            # 1. Uninvited user tries to handshake under whitelist mode -> 403 Forbidden
            res_uninvited = client.post("/api/v1/auth/session-handshake", json={
                "site": "e621",
                "site_user_id": 11111,
                "username": "StrangerFox",
            })
            assert res_uninvited.status_code == 403, f"Expected 403 on closed registration, got {res_uninvited.status_code}"
            assert "REGISTRATION_CLOSED" in res_uninvited.json()["detail"]

            # 2. Add user to whitelist
            assert engine.db.add_invited_user(site_user_id=11111, username="StrangerFox", note="Beta Tester 1") is True
            assert engine.db.is_site_user_invited(11111) is True

            # Now handshake succeeds -> 200 OK
            auth_rate_limiter.reset()
            res_invited = client.post("/api/v1/auth/session-handshake", json={
                "site": "e621",
                "site_user_id": 11111,
                "username": "StrangerFox",
            })
            assert res_invited.status_code == 200, res_invited.text
            token_str = res_invited.json()["token"]
            headers_fox = {"Authorization": f"Bearer {token_str}"}

            # 3. Protect account with password
            res_set_pw = client.post(
                "/api/v1/auth/password",
                json={"new_password": "MyStrongPassword123"},
                headers=headers_fox,
            )
            assert res_set_pw.status_code == 200

            # 4. Subsequent session-handshake without password -> 401 Unauthorized (PASSWORD_REQUIRED)
            auth_rate_limiter.reset()
            res_lock_nopass = client.post("/api/v1/auth/session-handshake", json={
                "site": "e621",
                "site_user_id": 11111,
                "username": "StrangerFox",
            })
            assert res_lock_nopass.status_code == 401, f"Expected 401, got {res_lock_nopass.status_code}"
            assert "PASSWORD_REQUIRED" in res_lock_nopass.json()["detail"]

            # 5. Session-handshake with WRONG password -> 401 Unauthorized
            res_lock_badpass = client.post("/api/v1/auth/session-handshake", json={
                "site": "e621",
                "site_user_id": 11111,
                "username": "StrangerFox",
                "password": "wrongpassword",
            })
            assert res_lock_badpass.status_code == 401, f"Expected 401, got {res_lock_badpass.status_code}"
            assert "PASSWORD_REQUIRED" in res_lock_badpass.json()["detail"]

            # 6. Session-handshake with CORRECT password -> 200 OK
            res_lock_ok = client.post("/api/v1/auth/session-handshake", json={
                "site": "e621",
                "site_user_id": 11111,
                "username": "StrangerFox",
                "password": "MyStrongPassword123",
            })
            assert res_lock_ok.status_code == 200, res_lock_ok.text
            assert res_lock_ok.json()["user_id"] == "e621:11111"

            # 7. Test removing from whitelist
            assert engine.db.remove_invited_user(11111) is True
            assert engine.db.is_site_user_invited(11111) is False
        finally:
            engine.db.close()

    print("  [PASS] Password Lock & Whitelist closed beta OK")


def test_in_memory_rate_limiter():
    print("Testing In-memory Rate Limiter (brute-force defense)...")
    auth_rate_limiter.reset()

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_rate_limiter.db"
        cfg = ServerConfig(custom_db_path=db_path)
        engine = Engine(cfg)
        app.state.engine = engine
        client = TestClient(app)

        try:
            # 1. 15 requests must succeed without 429
            for i in range(15):
                res = client.post("/api/v1/auth/login", json={"username": "nobody", "password": "bad"})
                assert res.status_code != 429, f"Request {i+1} should not be rate limited"

            # 2. 16th request must be rate-limited (HTTP 429)
            res_limit = client.post("/api/v1/auth/login", json={"username": "nobody", "password": "bad"})
            assert res_limit.status_code == 429, f"Expected 429 on 16th request, got {res_limit.status_code}"
            assert "Too many authentication attempts" in res_limit.json()["detail"]
            assert res_limit.headers.get("retry-after") == "60"

            # 3. Resetting limiter unblocks requests
            auth_rate_limiter.reset()
            res_unblocked = client.post("/api/v1/auth/login", json={"username": "nobody", "password": "bad"})
            assert res_unblocked.status_code != 429, "Limiter reset must allow subsequent requests"
        finally:
            engine.db.close()

    print("  [PASS] In-memory rate limiting and brute force protection OK")


def test_cli_commands():
    print("Testing CLI administration commands (app.cli)...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = str(Path(tmp_dir) / "test_cli.db")

        # 1. create-admin
        rc = cli.main(["--db-path", db_path, "create-admin", "--username", "superadmin", "--password", "admin_secret_pass"])
        assert rc == 0, f"create-admin failed with exit code {rc}"

        with Database(db_path) as db:
            adm = db.get_user_account_info("local:superadmin")
            assert adm is not None
            assert adm["role"] == 100
            assert adm["has_password"] is True

        # 2. set-role
        rc = cli.main(["--db-path", db_path, "set-role", "--user-id", "local:superadmin", "--role", "tester"])
        assert rc == 0
        with Database(db_path) as db:
            adm = db.get_user_account_info("local:superadmin")
            assert adm["role"] == 20

        rc = cli.main(["--db-path", db_path, "set-role", "--user-id", "local:superadmin", "--role", "100"])
        assert rc == 0
        with Database(db_path) as db:
            adm = db.get_user_account_info("local:superadmin")
            assert adm["role"] == 100

        # 3. invite-add
        rc = cli.main(["--db-path", db_path, "invite-add", "--site-user-id", "77777", "--username", "LuckyFox", "--note", "VIP Tester"])
        assert rc == 0
        with Database(db_path) as db:
            assert db.is_site_user_invited(77777) is True

        # 4. invite-list
        rc = cli.main(["--db-path", db_path, "invite-list"])
        assert rc == 0

        # 5. invite-remove
        rc = cli.main(["--db-path", db_path, "invite-remove", "--site-user-id", "77777"])
        assert rc == 0
        with Database(db_path) as db:
            assert db.is_site_user_invited(77777) is False

        # 6. reset-password
        rc = cli.main(["--db-path", db_path, "reset-password", "--user-id", "local:superadmin", "--password", "brand_new_pass_999"])
        assert rc == 0
        with Database(db_path) as db:
            auth_ok = db.authenticate_local_user("superadmin", "brand_new_pass_999")
            assert auth_ok is not None, "Password reset via CLI must allow login with new password"

    print("  [PASS] CLI administrative operations OK")


def run_all():
    print("=" * 65)
    print("RUNNING TIRESIAS AUTHENTICATION & SECURITY AUTOTEST SUITE (STAGE 2)")
    print("=" * 65)
    test_auth_crypto_primitives()
    test_ephemeral_admin_key()
    test_database_auth_layer()
    test_api_rbac_and_endpoints()
    test_protected_user_profile_and_feedback()
    test_password_lock_and_whitelist()
    test_in_memory_rate_limiter()
    test_cli_commands()
    print("=" * 65)
    print("ALL AUTHENTICATION, RBAC & SECURITY TESTS PASSED! (100% OK)")
    print("=" * 65)


if __name__ == "__main__":
    run_all()
