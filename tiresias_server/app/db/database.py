from __future__ import annotations

import sqlite3
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ..core.auth import (
    UserRole,
    hash_password,
    verify_password,
    generate_secure_token,
    hash_token,
)


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
class Database:
    """
    Lightweight SQLite storage for local user profiles, feedback, and settings.
    Utilizes WAL journal mode for concurrent read/write and persistent connection.
    Thread-safe across multiple ASGI worker threads.
    """

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.RLock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            self.db_path.as_posix(),
            timeout=30.0,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._init_schema()

    def close(self) -> None:
        """Explicitly closes connection and releases file handle (essential on Windows)."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _init_schema(self) -> None:
        assert self._conn is not None
        with self._conn:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    taste_archetype_id INTEGER DEFAULT 27,
                    locked_archetype INTEGER DEFAULT NULL,
                    forced_archetype_weight REAL DEFAULT NULL
                );

                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    post_id INTEGER NOT NULL,
                    signal_type TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_fb_user_signal ON user_feedback(user_id, signal_type);
                CREATE INDEX IF NOT EXISTS idx_fb_user_post ON user_feedback(user_id, post_id);

                CREATE TABLE IF NOT EXISTS user_seen (
                    user_id TEXT NOT NULL,
                    post_id INTEGER NOT NULL,
                    seen_at REAL NOT NULL,
                    PRIMARY KEY (user_id, post_id)
                );
                CREATE INDEX IF NOT EXISTS idx_seen_user ON user_seen(user_id);

                CREATE TABLE IF NOT EXISTS user_tag_blacklist (
                    user_id TEXT NOT NULL,
                    tag_id INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (user_id, tag_id)
                );

                CREATE TABLE IF NOT EXISTS user_settings (
                    user_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    PRIMARY KEY (user_id, key)
                );

                CREATE TABLE IF NOT EXISTS boards (
                    board_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    cover_post_id INTEGER,
                    is_public INTEGER DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_boards_user ON boards(user_id);

                CREATE TABLE IF NOT EXISTS board_posts (
                    board_id TEXT NOT NULL,
                    post_id INTEGER NOT NULL,
                    added_at REAL NOT NULL,
                    PRIMARY KEY (board_id, post_id),
                    FOREIGN KEY (board_id) REFERENCES boards(board_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_board_posts_bid ON board_posts(board_id);
                CREATE INDEX IF NOT EXISTS idx_board_posts_added ON board_posts(board_id, added_at);

                CREATE TABLE IF NOT EXISTS user_tag_likes (
                    user_id TEXT NOT NULL,
                    tag_id INTEGER NOT NULL,
                    like_count INTEGER DEFAULT 1,
                    PRIMARY KEY (user_id, tag_id)
                );
                CREATE INDEX IF NOT EXISTS idx_user_tag_likes_user ON user_tag_likes(user_id);

                CREATE TABLE IF NOT EXISTS auth_tokens (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_used_at REAL NOT NULL,
                    device_info TEXT DEFAULT '',
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_tokens_user ON auth_tokens(user_id);

                CREATE TABLE IF NOT EXISTS allowed_invites (
                    site_user_id INTEGER PRIMARY KEY,
                    username TEXT,
                    note TEXT,
                    created_at REAL
                );
            """)

            # Ensure newly added columns exist on already-created databases
            cur = self._conn.execute("PRAGMA table_info(boards)")
            existing_cols = {row[1] for row in cur.fetchall()}
            if "description" not in existing_cols:
                self._conn.execute("ALTER TABLE boards ADD COLUMN description TEXT DEFAULT ''")
            if "cover_post_id" not in existing_cols:
                self._conn.execute("ALTER TABLE boards ADD COLUMN cover_post_id INTEGER")
            if "is_public" not in existing_cols:
                self._conn.execute("ALTER TABLE boards ADD COLUMN is_public INTEGER DEFAULT 0")

            cur_u = self._conn.execute("PRAGMA table_info(users)")
            user_cols = {row[1] for row in cur_u.fetchall()}
            if "locked_archetype" not in user_cols:
                self._conn.execute("ALTER TABLE users ADD COLUMN locked_archetype INTEGER DEFAULT NULL")
            if "forced_archetype_weight" not in user_cols:
                self._conn.execute("ALTER TABLE users ADD COLUMN forced_archetype_weight REAL DEFAULT NULL")

            # Auth & Account columns
            if "username" not in user_cols:
                self._conn.execute("ALTER TABLE users ADD COLUMN username TEXT")
            if "display_name" not in user_cols:
                self._conn.execute("ALTER TABLE users ADD COLUMN display_name TEXT")
            if "password_hash" not in user_cols:
                self._conn.execute("ALTER TABLE users ADD COLUMN password_hash TEXT DEFAULT NULL")
            if "password_salt" not in user_cols:
                self._conn.execute("ALTER TABLE users ADD COLUMN password_salt TEXT DEFAULT NULL")
            if "role" not in user_cols:
                self._conn.execute("ALTER TABLE users ADD COLUMN role INTEGER DEFAULT 10")
            if "owner_id" not in user_cols:
                self._conn.execute("ALTER TABLE users ADD COLUMN owner_id TEXT DEFAULT NULL")
            if "site_source" not in user_cols:
                self._conn.execute("ALTER TABLE users ADD COLUMN site_source TEXT DEFAULT 'direct'")
            if "site_user_id" not in user_cols:
                self._conn.execute("ALTER TABLE users ADD COLUMN site_user_id INTEGER DEFAULT NULL")
            if "updated_at" not in user_cols:
                self._conn.execute("ALTER TABLE users ADD COLUMN updated_at REAL")

            # Indices
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_users_owner ON users(owner_id);")

            # Migrate legacy default archetype 0 or NULL to 27
            self._conn.execute(
                "UPDATE users SET taste_archetype_id = 27 WHERE (taste_archetype_id = 0 OR taste_archetype_id IS NULL) AND locked_archetype IS NULL"
            )

            # Seed demo accounts for role & permissions testing
            self._seed_demo_accounts()

    def _seed_demo_accounts(self) -> None:
        assert self._conn is not None
        now = time.time()

        dev_seeds = os.environ.get("TIRESIAS_DEV_SEEDS", "").strip().lower() in ("true", "1", "yes")

        if dev_seeds:
            # 1. user_demo (UserRole.USER = 10)
            cur = self._conn.execute("SELECT user_id FROM users WHERE user_id = 'user_demo'")
            if not cur.fetchone():
                p_hash, p_salt = hash_password("user123")
                self._conn.execute(
                    """
                    INSERT INTO users (user_id, created_at, username, display_name, password_hash, password_salt, role, site_source, taste_archetype_id)
                    VALUES ('user_demo', ?, 'user_demo', 'User Demo', ?, ?, 10, 'direct', 27)
                    """,
                    (now, p_hash, p_salt),
                )

            # 2. tester_demo (UserRole.TESTER = 20)
            cur = self._conn.execute("SELECT user_id FROM users WHERE user_id = 'tester_demo'")
            if not cur.fetchone():
                p_hash, p_salt = hash_password("tester123")
                self._conn.execute(
                    """
                    INSERT INTO users (user_id, created_at, username, display_name, password_hash, password_salt, role, site_source, taste_archetype_id)
                    VALUES ('tester_demo', ?, 'tester_demo', 'Tester Demo', ?, ?, 20, 'direct', 27)
                    """,
                    (now, p_hash, p_salt),
                )

            # 3. admin_demo (UserRole.ADMIN = 100)
            cur = self._conn.execute("SELECT user_id FROM users WHERE user_id = 'admin_demo'")
            if not cur.fetchone():
                p_hash, p_salt = hash_password("admin123")
                self._conn.execute(
                    """
                    INSERT INTO users (user_id, created_at, username, display_name, password_hash, password_salt, role, site_source, taste_archetype_id)
                    VALUES ('admin_demo', ?, 'admin_demo', 'Admin Demo', ?, ?, 100, 'direct', 27)
                    """,
                    (now, p_hash, p_salt),
                )

        # 4. default_user
        cur = self._conn.execute("SELECT username, role FROM users WHERE user_id = 'default_user'")
        row = cur.fetchone()
        if row:
            if not row["username"]:
                self._conn.execute(
                    "UPDATE users SET username = 'default_user', display_name = 'Default User', role = 10, site_source = 'direct' WHERE user_id = 'default_user'"
                )
        else:
            self._conn.execute(
                """
                INSERT INTO users (user_id, created_at, username, display_name, role, site_source, taste_archetype_id)
                VALUES ('default_user', ?, 'default_user', 'Default User', 10, 'direct', 27)
                """,
                (now,),
            )

    def ensure_user(self, user_id: str) -> None:
        if self._conn is None:
            return
        now = time.time()
        with self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO users (user_id, created_at, taste_archetype_id) VALUES (?, ?, 27)",
                (user_id, now),
            )

    def record_feedback(self, user_id: str, post_id: int, signal_type: str) -> None:
        if self._conn is None:
            return
        self.ensure_user(user_id)
        now = time.time()
        with self._conn:
            self._conn.execute(
                "INSERT INTO user_feedback (user_id, post_id, signal_type, created_at) VALUES (?, ?, ?, ?)",
                (user_id, int(post_id), signal_type.lower().strip(), now),
            )

    def remove_feedback(self, user_id: str, post_id: int, signal_type: Optional[str] = None) -> bool:
        """Removes a feedback interaction (e.g. un-dislike or un-like)."""
        if self._conn is None:
            return False
        with self._conn:
            if signal_type:
                st = signal_type.lower().strip()
                if st.startswith("undo_"):
                    st = st[5:]
                elif st == "unlike":
                    st = "like"
                elif st in ("undislike", "unhide"):
                    st = "hide"
                cur = self._conn.execute(
                    "DELETE FROM user_feedback WHERE user_id = ? AND post_id = ? AND signal_type = ?",
                    (user_id, int(post_id), st),
                )
            else:
                cur = self._conn.execute(
                    "DELETE FROM user_feedback WHERE user_id = ? AND post_id = ?",
                    (user_id, int(post_id)),
                )
            return cur.rowcount > 0

    def record_seen_batch(self, user_id: str, post_ids: Iterable[int]) -> int:
        if self._conn is None:
            return 0
        self.ensure_user(user_id)
        now = time.time()
        rows = [(user_id, int(pid), now) for pid in post_ids]
        if not rows:
            return 0
        with self._conn:
            cur = self._conn.executemany(
                "INSERT OR REPLACE INTO user_seen (user_id, post_id, seen_at) VALUES (?, ?, ?)",
                rows,
            )
            return cur.rowcount

    def get_user_hard_rejected_posts(self, user_id: str) -> Set[int]:
        """Returns set of post IDs explicitly disliked or hidden."""
        if self._conn is None:
            return set()
        cur = self._conn.execute(
            "SELECT DISTINCT post_id FROM user_feedback WHERE user_id = ? AND signal_type IN ('dislike', 'hide')",
            (user_id,),
        )
        return {int(row["post_id"]) for row in cur.fetchall()}

    def get_user_seen_posts(self, user_id: str) -> Set[int]:
        """Returns set of post IDs seen by the user (for soft scoring decay)."""
        if self._conn is None:
            return set()
        cur = self._conn.execute(
            "SELECT post_id FROM user_seen WHERE user_id = ?",
            (user_id,),
        )
        return {int(row["post_id"]) for row in cur.fetchall()}

    def get_user_liked_posts(self, user_id: str, limit: int = 100) -> List[int]:
        """Returns most recently liked post IDs."""
        if self._conn is None:
            return []
        cur = self._conn.execute(
            "SELECT post_id FROM user_feedback WHERE user_id = ? AND signal_type = 'like' ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        )
        return [int(row["post_id"]) for row in cur.fetchall()]

    def get_user_feedback_history(
        self,
        user_id: str,
        signal_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Returns paginated feedback history for a user (likes, dislikes/hides)."""
        if self._conn is None:
            return []
        self.ensure_user(user_id)
        limit = max(1, min(200, limit))
        offset = max(0, offset)
        if signal_type:
            st = signal_type.lower().strip()
            if st in ("dislike", "hide"):
                cur = self._conn.execute(
                    "SELECT post_id, signal_type, created_at FROM user_feedback WHERE user_id = ? AND signal_type IN ('dislike', 'hide') ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (user_id, limit, offset),
                )
            else:
                cur = self._conn.execute(
                    "SELECT post_id, signal_type, created_at FROM user_feedback WHERE user_id = ? AND signal_type = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (user_id, st, limit, offset),
                )
        else:
            cur = self._conn.execute(
                "SELECT post_id, signal_type, created_at FROM user_feedback WHERE user_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (user_id, limit, offset),
            )
        return [dict(row) for row in cur.fetchall()]

    def get_user_feedback_count(self, user_id: str, signal_type: Optional[str] = None) -> int:
        if self._conn is None:
            return 0
        if signal_type:
            st = signal_type.lower().strip()
            if st in ("dislike", "hide"):
                cur = self._conn.execute(
                    "SELECT count(*) as cnt FROM user_feedback WHERE user_id = ? AND signal_type IN ('dislike', 'hide')",
                    (user_id,),
                )
            else:
                cur = self._conn.execute(
                    "SELECT count(*) as cnt FROM user_feedback WHERE user_id = ? AND signal_type = ?",
                    (user_id, st),
                )
        else:
            cur = self._conn.execute(
                "SELECT count(*) as cnt FROM user_feedback WHERE user_id = ?",
                (user_id,),
            )
        row = cur.fetchone()
        return int(row["cnt"]) if row else 0

    def get_user_tag_blacklist(self, user_id: str) -> Set[int]:
        if self._conn is None:
            return set()
        cur = self._conn.execute(
            "SELECT tag_id FROM user_tag_blacklist WHERE user_id = ?",
            (user_id,),
        )
        return {int(row["tag_id"]) for row in cur.fetchall()}

    def add_tag_blacklist(self, user_id: str, tag_ids: Iterable[int]) -> int:
        if self._conn is None:
            return 0
        self.ensure_user(user_id)
        now = time.time()
        rows = [(user_id, int(t), now) for t in tag_ids]
        if not rows:
            return 0
        with self._conn:
            cur = self._conn.executemany(
                "INSERT OR IGNORE INTO user_tag_blacklist (user_id, tag_id, created_at) VALUES (?, ?, ?)",
                rows,
            )
            return cur.rowcount

    def remove_tag_blacklist(self, user_id: str, tag_ids: Iterable[int]) -> int:
        if self._conn is None:
            return 0
        t_list = [int(t) for t in tag_ids]
        if not t_list:
            return 0
        with self._conn:
            placeholders = ",".join("?" * len(t_list))
            cur = self._conn.execute(
                f"DELETE FROM user_tag_blacklist WHERE user_id = ? AND tag_id IN ({placeholders})",
                [user_id] + t_list,
            )
            return cur.rowcount

    def get_user_tag_likes(self, user_id: str) -> Dict[int, int]:
        """Returns mapping of tag_id -> like_count for the user."""
        if self._conn is None:
            return {}
        cur = self._conn.execute(
            "SELECT tag_id, like_count FROM user_tag_likes WHERE user_id = ?",
            (user_id,),
        )
        return {int(row["tag_id"]): int(row["like_count"]) for row in cur.fetchall()}

    def increment_user_tag_likes(self, user_id: str, tag_ids: Iterable[int]) -> None:
        """Increments like_count for each tag_id associated with a liked post."""
        if self._conn is None:
            return
        self.ensure_user(user_id)
        t_ids = [int(t) for t in tag_ids]
        if not t_ids:
            return
        with self._conn:
            for tid in t_ids:
                self._conn.execute(
                    """
                    INSERT INTO user_tag_likes (user_id, tag_id, like_count)
                    VALUES (?, ?, 1)
                    ON CONFLICT(user_id, tag_id) DO UPDATE SET like_count = like_count + 1
                    """,
                    (user_id, tid),
                )

    def get_user_archetype(self, user_id: str) -> int:
        if self._conn is None:
            return 27
        cur = self._conn.execute(
            "SELECT taste_archetype_id, locked_archetype FROM users WHERE user_id = ?",
            (user_id,),
        )
        row = cur.fetchone()
        if not row:
            return 27
        if row["locked_archetype"] is not None:
            return int(row["locked_archetype"])
        return int(row["taste_archetype_id"]) if row["taste_archetype_id"] is not None else 27

    def set_user_archetype(self, user_id: str, archetype_id: int) -> None:
        if self._conn is None:
            return
        self.ensure_user(user_id)
        # If user has locked archetype, do not overwrite with dynamic archetype
        cur = self._conn.execute(
            "SELECT locked_archetype FROM users WHERE user_id = ?",
            (user_id,),
        )
        row = cur.fetchone()
        if row and row["locked_archetype"] is not None:
            return
        with self._conn:
            self._conn.execute(
                "UPDATE users SET taste_archetype_id = ? WHERE user_id = ?",
                (int(archetype_id), user_id),
            )

    def get_user_archetype_info(self, user_id: str) -> Dict[str, Any]:
        if self._conn is None:
            return {"archetype_id": 27, "locked_archetype": None, "forced_weight": None}
        self.ensure_user(user_id)
        cur = self._conn.execute(
            "SELECT taste_archetype_id, locked_archetype, forced_archetype_weight FROM users WHERE user_id = ?",
            (user_id,),
        )
        row = cur.fetchone()
        if not row:
            return {"archetype_id": 27, "locked_archetype": None, "forced_weight": None}
        arch = row["taste_archetype_id"]
        locked = row["locked_archetype"]
        weight = row["forced_archetype_weight"]
        eff_arch = int(locked) if locked is not None else (int(arch) if arch is not None else 27)
        return {
            "archetype_id": eff_arch,
            "taste_archetype_id": int(arch) if arch is not None else 27,
            "locked_archetype": int(locked) if locked is not None else None,
            "forced_weight": float(weight) if weight is not None else None,
            "forced_archetype_weight": float(weight) if weight is not None else None,
        }

    def set_user_archetype_override(
        self, user_id: str, locked_archetype: Optional[int], forced_weight: Optional[float]
    ) -> None:
        if self._conn is None:
            return
        self.ensure_user(user_id)
        with self._conn:
            self._conn.execute(
                "UPDATE users SET locked_archetype = ?, forced_archetype_weight = ? WHERE user_id = ?",
                (locked_archetype, forced_weight, user_id),
            )

    def get_user_settings(self, user_id: str) -> Dict[str, str]:
        if self._conn is None:
            return {}
        cur = self._conn.execute(
            "SELECT key, value FROM user_settings WHERE user_id = ?",
            (user_id,),
        )
        return {row["key"]: row["value"] for row in cur.fetchall()}

    def set_user_settings(self, user_id: str, settings: Dict[str, str]) -> None:
        if self._conn is None:
            return
        self.ensure_user(user_id)
        rows = [(user_id, k, str(v)) for k, v in settings.items()]
        if not rows:
            return
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO user_settings (user_id, key, value) VALUES (?, ?, ?)",
                rows,
            )

    # -------------------------------------------------------------------------
    # Boards CRUD & Post Management
    # -------------------------------------------------------------------------

    def create_board(
        self,
        user_id: str,
        name: str,
        description: str = "",
        cover_post_id: Optional[int] = None,
        board_id: Optional[str] = None,
        is_public: bool = False,
    ) -> Dict[str, Any]:
        if self._conn is None:
            raise RuntimeError("Database connection is closed")
        self.ensure_user(user_id)
        now = time.time()
        import uuid
        bid = board_id or uuid.uuid4().hex[:12]
        pub_val = 1 if is_public else 0
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO boards (board_id, user_id, name, description, cover_post_id, is_public, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (bid, user_id, name.strip(), description.strip(), cover_post_id, pub_val, now, now),
            )
        return {
            "board_id": bid,
            "user_id": user_id,
            "name": name.strip(),
            "description": description.strip(),
            "cover_post_id": cover_post_id,
            "is_public": bool(is_public),
            "post_count": 0,
            "created_at": now,
            "updated_at": now,
        }

    def get_board(self, board_id: str) -> Optional[Dict[str, Any]]:
        if self._conn is None:
            return None
        cur = self._conn.execute(
            """
            SELECT b.board_id, b.user_id, b.name, b.description, b.cover_post_id, b.is_public, b.created_at, b.updated_at,
                   COUNT(bp.post_id) as post_count
            FROM boards b
            LEFT JOIN board_posts bp ON b.board_id = bp.board_id
            WHERE b.board_id = ?
            GROUP BY b.board_id
            """,
            (board_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        keys = row.keys()
        return {
            "board_id": row["board_id"],
            "user_id": row["user_id"],
            "name": row["name"],
            "description": row["description"] or "",
            "cover_post_id": row["cover_post_id"],
            "is_public": bool(row["is_public"]) if "is_public" in keys else False,
            "post_count": int(row["post_count"] or 0),
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
        }

    def list_user_boards(self, user_id: str) -> List[Dict[str, Any]]:
        if self._conn is None:
            return []
        cur = self._conn.execute(
            """
            SELECT b.board_id, b.user_id, b.name, b.description, b.cover_post_id, b.is_public, b.created_at, b.updated_at,
                   COUNT(bp.post_id) as post_count
            FROM boards b
            LEFT JOIN board_posts bp ON b.board_id = bp.board_id
            WHERE b.user_id = ?
            GROUP BY b.board_id
            ORDER BY b.updated_at DESC
            """,
            (user_id,),
        )
        res: List[Dict[str, Any]] = []
        for row in cur.fetchall():
            keys = row.keys()
            res.append({
                "board_id": row["board_id"],
                "user_id": row["user_id"],
                "name": row["name"],
                "description": row["description"] or "",
                "cover_post_id": row["cover_post_id"],
                "is_public": bool(row["is_public"]) if "is_public" in keys else False,
                "post_count": int(row["post_count"] or 0),
                "created_at": float(row["created_at"]),
                "updated_at": float(row["updated_at"]),
            })
        return res

    def update_board(
        self,
        board_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        cover_post_id: Optional[int] = None,
        is_public: Optional[bool] = None,
    ) -> bool:
        if self._conn is None:
            return False
        updates = []
        params: List[Any] = []
        if name is not None:
            updates.append("name = ?")
            params.append(name.strip())
        if description is not None:
            updates.append("description = ?")
            params.append(description.strip())
        if cover_post_id is not None:
            updates.append("cover_post_id = ?")
            params.append(int(cover_post_id))
        if is_public is not None:
            updates.append("is_public = ?")
            params.append(1 if is_public else 0)

        if not updates:
            return False

        updates.append("updated_at = ?")
        now = time.time()
        params.append(now)
        params.append(board_id)

        with self._conn:
            cur = self._conn.execute(
                f"UPDATE boards SET {', '.join(updates)} WHERE board_id = ?",
                params,
            )
            return cur.rowcount > 0

    def delete_board(self, board_id: str) -> bool:
        if self._conn is None:
            return False
        with self._conn:
            self._conn.execute("DELETE FROM board_posts WHERE board_id = ?", (board_id,))
            cur = self._conn.execute("DELETE FROM boards WHERE board_id = ?", (board_id,))
            return cur.rowcount > 0

    def add_posts_to_board(self, board_id: str, post_ids: Iterable[int]) -> int:
        if self._conn is None:
            return 0
        now = time.time()
        rows = [(board_id, int(pid), now) for pid in post_ids]
        if not rows:
            return 0
        with self._conn:
            cur = self._conn.executemany(
                "INSERT OR IGNORE INTO board_posts (board_id, post_id, added_at) VALUES (?, ?, ?)",
                rows,
            )
            added_count = cur.rowcount
            if added_count > 0:
                self._conn.execute(
                    "UPDATE boards SET updated_at = ? WHERE board_id = ?",
                    (now, board_id),
                )
            return added_count

    def remove_post_from_board(self, board_id: str, post_id: int) -> bool:
        if self._conn is None:
            return False
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM board_posts WHERE board_id = ? AND post_id = ?",
                (board_id, int(post_id)),
            )
            deleted = cur.rowcount > 0
            if deleted:
                self._conn.execute(
                    "UPDATE boards SET updated_at = ? WHERE board_id = ?",
                    (time.time(), board_id),
                )
            return deleted

    def get_board_post_records(self, board_id: str) -> List[Dict[str, Any]]:
        """Returns list of dicts with post_id and added_at for all posts in a board."""
        if self._conn is None:
            return []
        cur = self._conn.execute(
            "SELECT post_id, added_at FROM board_posts WHERE board_id = ? ORDER BY added_at DESC",
            (board_id,),
        )
        return [{"post_id": int(r["post_id"]), "added_at": float(r["added_at"])} for r in cur.fetchall()]

    def get_all_board_post_ids(self, board_id: str) -> Set[int]:
        """Returns set of all post IDs present in the board."""
        if self._conn is None:
            return set()
        cur = self._conn.execute(
            "SELECT post_id FROM board_posts WHERE board_id = ?",
            (board_id,),
        )
        return {int(r["post_id"]) for r in cur.fetchall()}

    def get_board_post_count(self, board_id: str) -> int:
        if self._conn is None:
            return 0
        cur = self._conn.execute(
            "SELECT COUNT(post_id) as cnt FROM board_posts WHERE board_id = ?",
            (board_id,),
        )
        row = cur.fetchone()
        return int(row["cnt"]) if row else 0

    def reset_user_history(
        self,
        user_id: Optional[str] = None,
        user_ids: Optional[List[str]] = None,
        reset_all: bool = False,
    ) -> Dict[str, Any]:
        """Completely clears feedback, seen records, and tag likes for specified user(s) or all users."""
        if self._conn is None:
            return {"feedback_deleted": 0, "seen_deleted": 0, "tag_likes_deleted": 0, "users_affected": 0}

        targets: List[str] = []
        if reset_all:
            cur = self._conn.execute("SELECT user_id FROM users")
            targets = [r["user_id"] for r in cur.fetchall()]
        elif user_ids:
            targets = [u.strip() for u in user_ids if u and u.strip()]
        elif user_id:
            targets = [user_id.strip()]

        for u in targets:
            self.ensure_user(u)

        with self._conn:
            if reset_all:
                c_fb = self._conn.execute("DELETE FROM user_feedback").rowcount
                c_seen = self._conn.execute("DELETE FROM user_seen").rowcount
                c_tags = self._conn.execute("DELETE FROM user_tag_likes").rowcount
                self._conn.execute("UPDATE users SET taste_archetype_id = 27, locked_archetype = NULL, forced_archetype_weight = NULL")
            elif targets:
                placeholders = ",".join("?" for _ in targets)
                c_fb = self._conn.execute(f"DELETE FROM user_feedback WHERE user_id IN ({placeholders})", targets).rowcount
                c_seen = self._conn.execute(f"DELETE FROM user_seen WHERE user_id IN ({placeholders})", targets).rowcount
                c_tags = self._conn.execute(f"DELETE FROM user_tag_likes WHERE user_id IN ({placeholders})", targets).rowcount
                self._conn.execute(f"UPDATE users SET taste_archetype_id = 27, locked_archetype = NULL, forced_archetype_weight = NULL WHERE user_id IN ({placeholders})", targets)
            else:
                c_fb = c_seen = c_tags = 0

            return {
                "feedback_deleted": max(0, c_fb),
                "seen_deleted": max(0, c_seen),
                "tag_likes_deleted": max(0, c_tags),
                "users_affected": len(targets),
                "targets": targets,
            }

    def get_user_profile_stats(self, user_id: str) -> Dict[str, Any]:
        """Returns statistical profile info, recent interactions, and archetype for tester inspection."""
        if self._conn is None:
            return {
                "user_id": user_id,
                "archetype_id": 27,
                "locked_archetype": None,
                "forced_archetype_weight": None,
                "likes_count": 0,
                "hides_count": 0,
                "seen_count": 0,
                "tag_likes": [],
                "recent_history": [],
            }
        self.ensure_user(user_id)
        cur = self._conn.cursor()

        # Archetype & overrides
        arch_info = self.get_user_archetype_info(user_id)
        arch_id = arch_info["archetype_id"]
        locked_arch = arch_info["locked_archetype"]
        forced_weight = arch_info["forced_weight"]

        # Likes & Hides count
        cur.execute("SELECT count(*) as cnt FROM user_feedback WHERE user_id = ? AND signal_type = 'like'", (user_id,))
        likes_cnt = int(cur.fetchone()["cnt"])

        cur.execute("SELECT count(*) as cnt FROM user_feedback WHERE user_id = ? AND signal_type IN ('hide', 'dislike')", (user_id,))
        hides_cnt = int(cur.fetchone()["cnt"])

        # Seen count
        cur.execute("SELECT count(*) as cnt FROM user_seen WHERE user_id = ?", (user_id,))
        seen_cnt = int(cur.fetchone()["cnt"])

        # Tag likes
        cur.execute("SELECT tag_id, like_count FROM user_tag_likes WHERE user_id = ? ORDER BY like_count DESC LIMIT 20", (user_id,))
        tag_likes = [{"tag_id": int(r["tag_id"]), "likes": int(r["like_count"])} for r in cur.fetchall()]

        # Recent history
        cur.execute("SELECT post_id, signal_type, created_at FROM user_feedback WHERE user_id = ? ORDER BY id DESC LIMIT 15", (user_id,))
        recent = [{"post_id": int(r["post_id"]), "signal": str(r["signal_type"]), "created_at": float(r["created_at"])} for r in cur.fetchall()]

        return {
            "user_id": user_id,
            "archetype_id": arch_id,
            "locked_archetype": locked_arch,
            "forced_archetype_weight": forced_weight,
            "likes_count": likes_cnt,
            "hides_count": hides_cnt,
            "seen_count": seen_cnt,
            "tag_likes": tag_likes,
            "recent_history": recent,
        }

    def check_integrity(self) -> str:
        if self._conn is None:
            return "Connection closed"
        cursor = self._conn.execute("PRAGMA integrity_check;")
        row = cursor.fetchone()
        return str(row[0]) if row else "unknown"

    # -------------------------------------------------------------------------
    # Whitelist / Closed Beta Management
    # -------------------------------------------------------------------------

    def is_site_user_invited(self, site_user_id: int) -> bool:
        """Checks if site_user_id is in allowed_invites whitelist."""
        if self._conn is None:
            return False
        cur = self._conn.execute("SELECT 1 FROM allowed_invites WHERE site_user_id = ?", (int(site_user_id),))
        return cur.fetchone() is not None

    def add_invited_user(self, site_user_id: int, username: str = "", note: str = "") -> bool:
        """Adds a site user to the allowed whitelist."""
        if self._conn is None:
            raise RuntimeError("Database connection is closed")
        now = time.time()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO allowed_invites (site_user_id, username, note, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(site_user_id) DO UPDATE SET
                    username = excluded.username,
                    note = excluded.note
                """,
                (int(site_user_id), username.strip(), note.strip(), now),
            )
            return True

    def remove_invited_user(self, site_user_id: int) -> bool:
        """Removes a site user from the allowed whitelist."""
        if self._conn is None:
            raise RuntimeError("Database connection is closed")
        with self._conn:
            cur = self._conn.execute("DELETE FROM allowed_invites WHERE site_user_id = ?", (int(site_user_id),))
            return cur.rowcount > 0

    def list_invited_users(self) -> List[Dict[str, Any]]:
        """Returns list of all invited users."""
        if self._conn is None:
            return []
        cur = self._conn.execute(
            "SELECT site_user_id, username, note, created_at FROM allowed_invites ORDER BY created_at DESC"
        )
        return [dict(r) for r in cur.fetchall()]

    # -------------------------------------------------------------------------
    # Authentication & User Account Management
    # -------------------------------------------------------------------------

    def create_or_get_e621_user(
        self,
        site_user_id: int,
        username: str,
        device_info: str = "",
        password: Optional[str] = None,
        registration_mode: str = "open",
    ) -> Tuple[Dict[str, Any], str]:
        """
        Seamless onboarding: resolves e621 site user into Tiresias account 'e621:<id>'.
        Enforces password lock for password-protected accounts and whitelist check in closed beta.
        Issues a cryptographically secure device Bearer token.
        """
        if self._conn is None:
            raise RuntimeError("Database connection is closed")
        now = time.time()
        canonical_id = f"e621:{int(site_user_id)}"
        clean_name = username.strip()

        with self._conn:
            cur = self._conn.execute("SELECT * FROM users WHERE user_id = ?", (canonical_id,))
            user_row = cur.fetchone()
            if user_row:
                # Password Lock check
                p_hash = user_row["password_hash"]
                p_salt = user_row["password_salt"]
                if p_hash:
                    if not password or not verify_password(password, p_hash, p_salt):
                        raise ValueError("PASSWORD_REQUIRED: Profile is protected by password. Provide password to authorize a new device.")

                self._conn.execute(
                    "UPDATE users SET username = ?, display_name = ?, updated_at = ? WHERE user_id = ?",
                    (clean_name, f"{clean_name} (e621)", now, canonical_id),
                )
            else:
                # Whitelist Check for new registration
                if registration_mode == "whitelist":
                    if not self.is_site_user_invited(int(site_user_id)):
                        raise ValueError("REGISTRATION_CLOSED: Registration is restricted (closed beta). Your ID is not on the whitelist.")

                self._conn.execute(
                    """
                    INSERT INTO users (user_id, created_at, username, display_name, role, site_source, site_user_id, taste_archetype_id, updated_at)
                    VALUES (?, ?, ?, ?, 10, 'e621', ?, 27, ?)
                    """,
                    (canonical_id, now, clean_name, f"{clean_name} (e621)", int(site_user_id), now),
                )

            # Issue bearer token
            raw_token = generate_secure_token()
            token_h = hash_token(raw_token)
            self._conn.execute(
                """
                INSERT INTO auth_tokens (token_hash, user_id, created_at, last_used_at, device_info)
                VALUES (?, ?, ?, ?, ?)
                """,
                (token_h, canonical_id, now, now, device_info),
            )

            cur_fresh = self._conn.execute("SELECT * FROM users WHERE user_id = ?", (canonical_id,))
            user_dict = dict(cur_fresh.fetchone())
            has_pw = bool(user_dict.get("password_hash"))
            user_dict.pop("password_hash", None)
            user_dict.pop("password_salt", None)
            user_dict["has_password"] = has_pw
            return user_dict, raw_token

    def set_user_role(self, user_id: str, role: int) -> bool:
        """Updates the authorization role level for a user."""
        if self._conn is None:
            raise RuntimeError("Database connection is closed")
        now = time.time()
        with self._conn:
            cur = self._conn.execute(
                "UPDATE users SET role = ?, updated_at = ? WHERE user_id = ? OR username = ? COLLATE NOCASE",
                (int(role), now, user_id, user_id),
            )
            return cur.rowcount > 0

    def create_or_update_admin(
        self,
        username: str,
        password: str,
        display_name: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], str]:
        """Creates a new admin account or updates and elevates an existing user to admin."""
        if self._conn is None:
            raise RuntimeError("Database connection is closed")
        clean_name = username.strip()
        if not clean_name:
            raise ValueError("Administrator username cannot be empty.")
        if len(password) < 4:
            raise ValueError("Password must be at least 4 characters long.")

        canonical_id = f"local:{clean_name.lower()}"
        now = time.time()
        p_hash, p_salt = hash_password(password)
        d_name = display_name.strip() if display_name else clean_name

        with self._conn:
            cur = self._conn.execute(
                "SELECT user_id FROM users WHERE user_id = ? OR (username = ? COLLATE NOCASE AND site_source = 'direct')",
                (canonical_id, clean_name),
            )
            row = cur.fetchone()
            if row:
                uid = row["user_id"]
                self._conn.execute(
                    """
                    UPDATE users
                    SET password_hash = ?, password_salt = ?, role = 100, updated_at = ?
                    WHERE user_id = ?
                    """,
                    (p_hash, p_salt, now, uid),
                )
                raw_token = generate_secure_token()
                token_h = hash_token(raw_token)
                self._conn.execute(
                    """
                    INSERT INTO auth_tokens (token_hash, user_id, created_at, last_used_at, device_info)
                    VALUES (?, ?, ?, ?, 'CLI')
                    """,
                    (token_h, uid, now, now),
                )
                cur_fresh = self._conn.execute("SELECT * FROM users WHERE user_id = ?", (uid,))
                user_dict = dict(cur_fresh.fetchone())
                user_dict.pop("password_hash", None)
                user_dict.pop("password_salt", None)
                user_dict["has_password"] = True
                return user_dict, raw_token
            else:
                return self.register_local_user(
                    username=clean_name,
                    password=password,
                    role=100,
                    display_name=d_name,
                    device_info="CLI",
                )

    def register_local_user(
        self,
        username: str,
        password: str,
        role: int = 10,
        display_name: Optional[str] = None,
        device_info: str = "",
    ) -> Tuple[Dict[str, Any], str]:
        """
        Direct registration of standalone Tiresias profile 'local:<username>'.
        Guarantees no collision with e621 profiles.
        """
        if self._conn is None:
            raise RuntimeError("Database connection is closed")
        clean_name = username.strip()
        if not clean_name:
            raise ValueError("Username cannot be empty.")
        if clean_name.lower().startswith("e621:"):
            raise ValueError("Prefix 'e621:' is reserved for site sessions.")
        if len(password) < 4:
            raise ValueError("Password must be at least 4 characters long.")

        canonical_id = f"local:{clean_name.lower()}"
        now = time.time()
        p_hash, p_salt = hash_password(password)
        d_name = display_name.strip() if display_name else clean_name

        with self._conn:
            cur = self._conn.execute(
                "SELECT user_id FROM users WHERE user_id = ? OR (username = ? COLLATE NOCASE AND site_source = 'direct')",
                (canonical_id, clean_name),
            )
            if cur.fetchone():
                raise ValueError(f"User '{clean_name}' is already registered.")

            self._conn.execute(
                """
                INSERT INTO users (user_id, created_at, username, display_name, password_hash, password_salt, role, site_source, taste_archetype_id, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'direct', 27, ?)
                """,
                (canonical_id, now, clean_name, d_name, p_hash, p_salt, int(role), now),
            )

            raw_token = generate_secure_token()
            token_h = hash_token(raw_token)
            self._conn.execute(
                """
                INSERT INTO auth_tokens (token_hash, user_id, created_at, last_used_at, device_info)
                VALUES (?, ?, ?, ?, ?)
                """,
                (token_h, canonical_id, now, now, device_info),
            )

            cur_fresh = self._conn.execute("SELECT * FROM users WHERE user_id = ?", (canonical_id,))
            user_dict = dict(cur_fresh.fetchone())
            user_dict.pop("password_hash", None)
            user_dict.pop("password_salt", None)
            user_dict["has_password"] = True
            return user_dict, raw_token

    def authenticate_local_user(
        self,
        username: str,
        password: str,
        device_info: str = "",
    ) -> Optional[Tuple[Dict[str, Any], str]]:
        """
        Verifies login credentials and issues a fresh Bearer token.
        Supports lookup by exact user_id or by case-insensitive username.
        """
        if self._conn is None:
            return None
        clean_name = username.strip()
        with self._conn:
            cur = self._conn.execute(
                """
                SELECT * FROM users
                WHERE (user_id = ? OR username = ? COLLATE NOCASE)
                AND password_hash IS NOT NULL
                """,
                (clean_name, clean_name),
            )
            row = cur.fetchone()
            if not row:
                return None

            p_hash = row["password_hash"]
            p_salt = row["password_salt"]
            if not p_hash or not p_salt:
                return None

            if not verify_password(password, p_hash, p_salt):
                return None

            user_id = row["user_id"]
            now = time.time()
            raw_token = generate_secure_token()
            token_h = hash_token(raw_token)
            self._conn.execute(
                """
                INSERT INTO auth_tokens (token_hash, user_id, created_at, last_used_at, device_info)
                VALUES (?, ?, ?, ?, ?)
                """,
                (token_h, user_id, now, now, device_info),
            )

            user_dict = dict(row)
            user_dict.pop("password_hash", None)
            user_dict.pop("password_salt", None)
            user_dict["has_password"] = True
            return user_dict, raw_token

    def validate_token(self, raw_token: str) -> Optional[Dict[str, Any]]:
        """Validates Bearer token, refreshes last_used_at timestamp, and returns user dict."""
        if self._conn is None or not raw_token:
            return None
        token_h = hash_token(raw_token)
        now = time.time()
        with self._conn:
            cur = self._conn.execute(
                """
                SELECT u.*
                FROM auth_tokens t
                JOIN users u ON t.user_id = u.user_id
                WHERE t.token_hash = ?
                """,
                (token_h,),
            )
            row = cur.fetchone()
            if not row:
                return None

            self._conn.execute(
                "UPDATE auth_tokens SET last_used_at = ? WHERE token_hash = ?",
                (now, token_h),
            )

            user_dict = dict(row)
            has_pw = bool(user_dict.get("password_hash"))
            user_dict.pop("password_hash", None)
            user_dict.pop("password_salt", None)
            user_dict["has_password"] = has_pw
            return user_dict

    def revoke_token(self, raw_token: str) -> bool:
        """Revokes an active token (logout)."""
        if self._conn is None or not raw_token:
            return False
        token_h = hash_token(raw_token)
        with self._conn:
            cur = self._conn.execute("DELETE FROM auth_tokens WHERE token_hash = ?", (token_h,))
            return cur.rowcount > 0

    def set_user_password(
        self,
        user_id: str,
        new_password: str,
        new_username: Optional[str] = None,
    ) -> bool:
        """Sets or updates password for any account. Optionally sets or updates username."""
        if self._conn is None:
            return False
        if len(new_password) < 4:
            raise ValueError("Password must be at least 4 characters long.")
        p_hash, p_salt = hash_password(new_password)
        now = time.time()
        with self._conn:
            if new_username and new_username.strip():
                clean_name = new_username.strip()
                cur_u = self._conn.execute(
                    "SELECT user_id FROM users WHERE username = ? COLLATE NOCASE AND user_id != ?",
                    (clean_name, user_id),
                )
                if cur_u.fetchone():
                    raise ValueError(f"Username '{clean_name}' is already taken.")
                cur = self._conn.execute(
                    "UPDATE users SET password_hash = ?, password_salt = ?, username = ?, updated_at = ? WHERE user_id = ?",
                    (p_hash, p_salt, clean_name, now, user_id),
                )
            else:
                cur = self._conn.execute(
                    "UPDATE users SET password_hash = ?, password_salt = ?, updated_at = ? WHERE user_id = ?",
                    (p_hash, p_salt, now, user_id),
                )
            return cur.rowcount > 0

    def link_site_session(
        self,
        user_id: str,
        site_user_id: int,
        site_username: str,
    ) -> bool:
        """
        Links active e621 session (site_user_id) to existing user account.
        Enables seamless site login to resolve into this Tiresias account.
        """
        if self._conn is None:
            raise RuntimeError("Database connection is closed")
        now = time.time()
        clean_name = site_username.strip()
        with self._conn:
            cur = self._conn.execute(
                """
                UPDATE users
                SET site_user_id = ?,
                    display_name = CASE WHEN display_name IS NULL OR display_name = '' THEN ? ELSE display_name END,
                    site_source = 'e621',
                    updated_at = ?
                WHERE user_id = ?
                """,
                (int(site_user_id), clean_name, now, user_id),
            )
            return cur.rowcount > 0

    def get_user_account_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Returns safe user account info without passwords."""
        if self._conn is None:
            return None
        cur = self._conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        row = cur.fetchone()
        if not row:
            return None
        d = dict(row)
        has_pw = bool(d.get("password_hash"))
        d.pop("password_hash", None)
        d.pop("password_salt", None)
        d["has_password"] = has_pw
        return d

    # -------------------------------------------------------------------------
    # Tester Profile Sandbox (Multi-profile management)
    # -------------------------------------------------------------------------

    def create_sandbox_profile(self, tester_id: str, profile_name: str) -> Dict[str, Any]:
        """
        Creates an isolated test sub-profile for a tester ('test:<tester_id>:<name>').
        The profile is owned by tester_id and cannot be touched by other users.
        """
        if self._conn is None:
            raise RuntimeError("Database connection is closed")
        clean_name = profile_name.strip()
        if not clean_name:
            raise ValueError("Profile name cannot be empty.")
        now = time.time()
        safe_suffix = clean_name.lower().replace(" ", "_")
        canonical_id = f"test:{tester_id}:{safe_suffix}"

        with self._conn:
            cur = self._conn.execute("SELECT * FROM users WHERE user_id = ?", (canonical_id,))
            existing = cur.fetchone()
            if existing:
                d = dict(existing)
                d.pop("password_hash", None)
                d.pop("password_salt", None)
                return d

            self._conn.execute(
                """
                INSERT INTO users (user_id, created_at, username, display_name, role, owner_id, site_source, taste_archetype_id, updated_at)
                VALUES (?, ?, ?, ?, 10, ?, 'sandbox', 27, ?)
                """,
                (canonical_id, now, safe_suffix, f"[Test] {clean_name}", tester_id, now),
            )
            cur_fresh = self._conn.execute("SELECT * FROM users WHERE user_id = ?", (canonical_id,))
            d = dict(cur_fresh.fetchone())
            d.pop("password_hash", None)
            d.pop("password_salt", None)
            d["has_password"] = False
            return d

    def list_tester_sandbox_profiles(self, tester_id: str) -> List[Dict[str, Any]]:
        """Returns list of sub-profiles owned by this tester."""
        if self._conn is None:
            return []
        cur = self._conn.execute(
            """
            SELECT user_id, username, display_name, created_at, taste_archetype_id, locked_archetype
            FROM users
            WHERE owner_id = ?
            ORDER BY created_at ASC
            """,
            (tester_id,),
        )
        return [dict(r) for r in cur.fetchall()]

    def delete_sandbox_profile(self, tester_id: str, profile_id: str) -> bool:
        """Deletes a sandbox profile owned by tester, cascading associated data."""
        if self._conn is None:
            return False
        with self._conn:
            cur = self._conn.execute("SELECT owner_id FROM users WHERE user_id = ?", (profile_id,))
            row = cur.fetchone()
            if not row or row["owner_id"] != tester_id:
                return False

            self._conn.execute("DELETE FROM user_feedback WHERE user_id = ?", (profile_id,))
            self._conn.execute("DELETE FROM user_seen WHERE user_id = ?", (profile_id,))
            self._conn.execute("DELETE FROM user_tag_likes WHERE user_id = ?", (profile_id,))
            self._conn.execute("DELETE FROM user_tag_blacklist WHERE user_id = ?", (profile_id,))
            self._conn.execute("DELETE FROM user_settings WHERE user_id = ?", (profile_id,))
            self._conn.execute("DELETE FROM boards WHERE user_id = ?", (profile_id,))
            self._conn.execute("DELETE FROM auth_tokens WHERE user_id = ?", (profile_id,))
            self._conn.execute("DELETE FROM users WHERE user_id = ?", (profile_id,))
            return True

    # -------------------------------------------------------------------------
    # Administrative Queries & Database Operations
    # -------------------------------------------------------------------------

    def list_users(
        self,
        page: int = 1,
        page_size: int = 50,
        search: Optional[str] = None,
        role: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Safely queries the users table with optional search across user_id, username, and display_name.
        Supports optional role filtering (WHERE role = ?).
        Returns a tuple of (user_dicts, total_matched_count).
        Does not expose password_hash or password_salt. Includes has_password boolean.
        """
        if self._conn is None:
            return [], 0

        page = max(1, int(page))
        page_size = max(1, min(int(page_size), 200))
        offset = (page - 1) * page_size

        conditions: List[str] = []
        params: List[Any] = []

        if search and search.strip():
            s = f"%{search.strip()}%"
            conditions.append("(user_id LIKE ? OR username LIKE ? OR display_name LIKE ?)")
            params.extend([s, s, s])

        if role is not None:
            conditions.append("role = ?")
            params.append(int(role))

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        with self._conn:
            count_sql = f"SELECT COUNT(*) FROM users {where_clause}"
            cur_count = self._conn.execute(count_sql, params)
            total_count = cur_count.fetchone()[0]

            query_sql = f"""
                SELECT * FROM users
                {where_clause}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """
            cur_users = self._conn.execute(query_sql, params + [page_size, offset])
            results: List[Dict[str, Any]] = []
            for r in cur_users.fetchall():
                d = dict(r)
                has_pw = bool(d.get("password_hash"))
                d.pop("password_hash", None)
                d.pop("password_salt", None)
                d["has_password"] = has_pw
                results.append(d)

            return results, total_count

    def delete_user(self, user_id: str) -> bool:
        """
        In an atomic transaction, deletes associated records for user_id:
        auth_tokens, user_feedback, user_seen, user_tag_likes, user_tag_blacklist,
        user_settings, boards and board_posts, owned sandbox sub-profiles, and the user record.
        Returns True if deleted, False if user didn't exist.
        """
        if self._conn is None:
            return False

        with self._conn:
            cur = self._conn.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
            if not cur.fetchone():
                return False

            tables_cur = self._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {r[0] for r in tables_cur.fetchall()}

            # Cascade boards and board posts
            if "boards" in tables and "board_posts" in tables:
                self._conn.execute(
                    "DELETE FROM board_posts WHERE board_id IN (SELECT board_id FROM boards WHERE user_id = ?)",
                    (user_id,),
                )
            if "boards" in tables:
                self._conn.execute("DELETE FROM boards WHERE user_id = ?", (user_id,))

            if "user_boards" in tables and "user_board_posts" in tables:
                self._conn.execute(
                    "DELETE FROM user_board_posts WHERE board_id IN (SELECT board_id FROM user_boards WHERE user_id = ?)",
                    (user_id,),
                )
            if "user_boards" in tables:
                self._conn.execute("DELETE FROM user_boards WHERE user_id = ?", (user_id,))

            # User data tables
            for tbl in [
                "auth_tokens",
                "user_feedback",
                "user_seen",
                "user_tag_likes",
                "user_tag_blacklist",
                "user_settings",
                "user_archetype_overrides",
            ]:
                if tbl in tables:
                    self._conn.execute(f"DELETE FROM {tbl} WHERE user_id = ?", (user_id,))

            # Cascade any sandbox profiles owned by this tester
            cur_sub = self._conn.execute("SELECT user_id FROM users WHERE owner_id = ?", (user_id,))
            sub_ids = [r[0] for r in cur_sub.fetchall()]
            for sid in sub_ids:
                if "boards" in tables and "board_posts" in tables:
                    self._conn.execute(
                        "DELETE FROM board_posts WHERE board_id IN (SELECT board_id FROM boards WHERE user_id = ?)",
                        (sid,),
                    )
                if "boards" in tables:
                    self._conn.execute("DELETE FROM boards WHERE user_id = ?", (sid,))
                for tbl in [
                    "auth_tokens",
                    "user_feedback",
                    "user_seen",
                    "user_tag_likes",
                    "user_tag_blacklist",
                    "user_settings",
                    "user_archetype_overrides",
                ]:
                    if tbl in tables:
                        self._conn.execute(f"DELETE FROM {tbl} WHERE user_id = ?", (sid,))
                self._conn.execute("DELETE FROM users WHERE user_id = ?", (sid,))

            # Finally delete the user row itself
            self._conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
            return True

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Returns row counts for users, user_feedback, user_boards, allowed_invites, and auth_tokens.
        """
        if self._conn is None:
            return {
                "users": 0,
                "user_feedback": 0,
                "user_boards": 0,
                "boards": 0,
                "allowed_invites": 0,
                "auth_tokens": 0,
            }

        with self._conn:
            tables_cur = self._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {r[0] for r in tables_cur.fetchall()}

            def _cnt(tbl_name: str) -> int:
                if tbl_name in tables:
                    return self._conn.execute(f"SELECT COUNT(*) FROM {tbl_name}").fetchone()[0]
                return 0

            users_count = _cnt("users")
            feedback_count = _cnt("user_feedback")
            boards_count = _cnt("boards") if "boards" in tables else _cnt("user_boards")
            invites_count = _cnt("allowed_invites")
            tokens_count = _cnt("auth_tokens")

            return {
                "users": users_count,
                "user_feedback": feedback_count,
                "user_boards": boards_count,
                "boards": boards_count,
                "allowed_invites": invites_count,
                "auth_tokens": tokens_count,
            }

