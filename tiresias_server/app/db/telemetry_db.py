from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("tiresias.telemetry")


def _synchronized(method):
    def wrapper(self, *args, **kwargs):
        lock = getattr(self, "_lock", None)
        if lock is not None:
            with lock:
                return method(self, *args, **kwargs)
        return method(self, *args, **kwargs)
    wrapper.__name__ = method.__name__
    wrapper.__doc__ = method.__doc__
    return wrapper


def thread_safe(cls):
    for attr_name, attr_value in list(cls.__dict__.items()):
        if callable(attr_value) and not attr_name.startswith("__"):
            setattr(cls, attr_name, _synchronized(attr_value))
    return cls


@thread_safe
class TelemetryDatabase:
    """
    Dedicated local SQLite storage for client-side JavaScript error reports and crash telemetry.
    Keeps telemetry isolated from user profile data (tiresias_user.db).
    Automatically trims old records to prevent unbounded database growth.
    """
    MAX_RECORDS = 500

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = threading.RLock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = sqlite3.connect(
            self.db_path.as_posix(),
            timeout=30.0,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_schema()

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def check_integrity(self) -> str:
        if self._conn is None:
            return "Connection closed"
        cursor = self._conn.execute("PRAGMA integrity_check;")
        row = cursor.fetchone()
        return str(row[0]) if row else "unknown"

    def _init_schema(self) -> None:
        if self._conn is None:
            return
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS client_errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT,
                error_type TEXT,
                message TEXT,
                stack TEXT,
                url TEXT,
                source_file TEXT,
                lineno INTEGER,
                colno INTEGER,
                metadata_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_client_errors_ts ON client_errors(timestamp DESC);
        """)
        self._conn.commit()

    def record_error(
        self,
        user_id: str = "default_user",
        error_type: str = "js_error",
        message: str = "",
        stack: Optional[str] = None,
        url: Optional[str] = None,
        source_file: Optional[str] = None,
        lineno: Optional[int] = None,
        colno: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        if self._conn is None:
            return 0
        metadata_str = json.dumps(metadata or {}, ensure_ascii=False)
        cursor = self._conn.execute(
            """
            INSERT INTO client_errors (
                user_id, error_type, message, stack, url, source_file, lineno, colno, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, error_type, message, stack, url, source_file, lineno, colno, metadata_str),
        )
        new_id = cursor.lastrowid
        self._conn.commit()

        # Trim old records beyond MAX_RECORDS
        self._conn.execute(
            """
            DELETE FROM client_errors
            WHERE id NOT IN (
                SELECT id FROM client_errors ORDER BY id DESC LIMIT ?
            )
            """,
            (self.MAX_RECORDS,),
        )
        self._conn.commit()
        return int(new_id)

    def get_recent_errors(self, limit: int = 50) -> List[Dict[str, Any]]:
        if self._conn is None:
            return []
        cursor = self._conn.execute(
            """
            SELECT id, timestamp, user_id, error_type, message, stack, url, source_file, lineno, colno, metadata_json
            FROM client_errors
            ORDER BY id DESC
            LIMIT ?
            """,
            (min(limit, 200),),
        )
        rows = cursor.fetchall()
        results = []
        for r in rows:
            meta = {}
            if r["metadata_json"]:
                try:
                    meta = json.loads(r["metadata_json"])
                except Exception:
                    pass
            results.append({
                "id": r["id"],
                "timestamp": str(r["timestamp"]),
                "user_id": r["user_id"],
                "error_type": r["error_type"],
                "message": r["message"],
                "stack": r["stack"],
                "url": r["url"],
                "source_file": r["source_file"],
                "lineno": r["lineno"],
                "colno": r["colno"],
                "metadata": meta,
            })
        return results

    def clear_errors(self) -> int:
        if self._conn is None:
            return 0
        cursor = self._conn.execute("SELECT COUNT(*) FROM client_errors")
        count = cursor.fetchone()[0]
        self._conn.execute("DELETE FROM client_errors")
        self._conn.commit()
        return int(count)

    def get_count(self) -> int:
        if self._conn is None:
            return 0
        cursor = self._conn.execute("SELECT COUNT(*) FROM client_errors")
        return int(cursor.fetchone()[0])
