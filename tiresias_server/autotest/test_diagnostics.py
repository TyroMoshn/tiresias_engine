from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

# Add tiresias_server root to path
server_root = Path(__file__).resolve().parents[1]
if str(server_root) not in sys.path:
    sys.path.insert(0, str(server_root))

from app.config import ServerConfig
from app.core.engine import Engine
from app.core.diagnostics import DiagnosticsAuditor
from app.core.auth import get_admin_master_key
from app.db.telemetry_db import TelemetryDatabase
from fastapi.testclient import TestClient
from app.main import app


class TestDiagnosticsAndTelemetry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ServerConfig()
        cls.engine = Engine(cls.config)
        app.state.engine = cls.engine
        cls.app = app
        cls.client = TestClient(cls.app)

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()

    def test_01_diagnostics_auditor_full_audit(self):
        auditor = DiagnosticsAuditor(self.engine)
        report = auditor.run_full_audit()
        self.assertIn("status", report)
        self.assertIn("warnings", report)
        self.assertIn("subsystems", report)
        self.assertIn("ratings", report["subsystems"])
        self.assertIn("media_assets", report["subsystems"])
        self.assertIn("faiss", report["subsystems"])
        self.assertIn("collab", report["subsystems"])
        self.assertIn("databases", report["subsystems"])
        print("\n[DIAGNOSTICS REPORT]")
        print("Status:", report["status"])
        print("Duration ms:", report["audit_duration_ms"])
        print("Warnings count:", report["total_warnings"])
        for w in report["warnings"]:
            print("  WARN:", w)

    def test_02_telemetry_db_crud(self):
        tdb = self.engine.telemetry_db
        self.assertIsNotNone(tdb)
        # Clear existing
        tdb.clear_errors()
        self.assertEqual(tdb.get_count(), 0)

        # Record error
        err_id = tdb.record_error(
            user_id="tester_1",
            error_type="js_error",
            message="TypeError: Cannot read properties of undefined",
            stack="TypeError: Cannot read properties...\n at Object.render (popup.js:12:34)",
            url="moz-extension://test/popup.html",
            source_file="popup.js",
            lineno=12,
            colno=34,
            metadata={"component": "Popup"},
        )
        self.assertGreater(err_id, 0)
        self.assertEqual(tdb.get_count(), 1)

        # Retrieve
        recent = tdb.get_recent_errors(limit=10)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]["user_id"], "tester_1")
        self.assertEqual(recent[0]["error_type"], "js_error")
        self.assertIn("TypeError", recent[0]["message"])
        self.assertEqual(recent[0]["metadata"]["component"], "Popup")

        # Test trimming (insert > 500)
        # Verify integrity
        integ = tdb.check_integrity()
        self.assertEqual(integ.lower(), "ok")

    def test_03_http_endpoints(self):
        # 1. Health endpoint includes diagnostics & telemetry
        res = self.client.get("/api/v1/system/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("diagnostics_status", data)
        self.assertIn("telemetry_errors_count", data)

        # 2. Diagnostics endpoint
        res_diag = self.client.get("/api/v1/system/diagnostics")
        self.assertEqual(res_diag.status_code, 200)
        diag_data = res_diag.json()
        self.assertIn("status", diag_data)
        self.assertIn("subsystems", diag_data)

        # 3. Post client error
        payload = {
            "user_id": "test_user_telemetry",
            "error_type": "unhandled_rejection",
            "message": "Error: Network request failed",
            "stack": "Error: Network request failed\n at fetch (api.js:45)",
            "url": "https://e926.net/posts",
            "metadata": {"test": True},
        }
        res_post = self.client.post("/api/v1/system/client-errors", json=payload)
        self.assertEqual(res_post.status_code, 200)
        self.assertEqual(res_post.json()["status"], "ok")

        # 4. Get client errors
        res_get = self.client.get("/api/v1/system/client-errors")
        self.assertEqual(res_get.status_code, 200)
        errors_data = res_get.json()
        self.assertGreaterEqual(errors_data["total_errors"], 1)

        # 5. Delete client errors (Requires ADMIN authorization)
        res_del = self.client.delete(
            "/api/v1/system/client-errors",
            headers={"X-Admin-Key": get_admin_master_key()},
        )
        self.assertEqual(res_del.status_code, 200)
        self.assertEqual(res_del.json()["status"], "ok")

        # 6. Verify count is now 0
        res_get_empty = self.client.get("/api/v1/system/client-errors")
        self.assertEqual(res_get_empty.json()["total_errors"], 0)


if __name__ == "__main__":
    unittest.main()
