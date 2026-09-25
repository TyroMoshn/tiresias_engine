#!/usr/bin/env python3
"""
Test suite for Dynamic Archetype Migration, Adaptive Bayesian Decay, and Tester Overrides.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

# Set paths
HERE = Path(__file__).resolve().parent
SERVER_DIR = HERE.parent
ROOT_DIR = SERVER_DIR.parent
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import ServerConfig
from app.core.engine import Engine
from app.db.database import Database
from fastapi.testclient import TestClient
from app.main import app


def test_database_archetype_methods() -> None:
    print("Testing Database archetype persistence and lock behavior...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_arch.db"
        with Database(db_path) as db:
            user = "alice"
            # 1. Cold start default
            info = db.get_user_archetype_info(user)
            assert info["taste_archetype_id"] == 27, f"Expected 27, got {info['taste_archetype_id']}"
            assert info["locked_archetype"] is None
            assert info["forced_weight"] is None
            assert db.get_user_archetype(user) == 27

            # 2. Dynamic update
            db.set_user_archetype(user, 14)
            assert db.get_user_archetype(user) == 14

            # 3. Lock archetype and force weight
            db.set_user_archetype_override(user, locked_archetype=42, forced_weight=0.65)
            info = db.get_user_archetype_info(user)
            assert info["locked_archetype"] == 42
            assert info["forced_weight"] == 0.65
            assert info["archetype_id"] == 42
            assert db.get_user_archetype(user) == 42

            # 4. Attempt dynamic update while locked
            db.set_user_archetype(user, 5)
            # Must remain 42 because it is locked
            assert db.get_user_archetype(user) == 42
            info = db.get_user_archetype_info(user)
            assert info["archetype_id"] == 42

            # 5. Clear override
            db.set_user_archetype_override(user, locked_archetype=None, forced_weight=None)
            info = db.get_user_archetype_info(user)
            assert info["locked_archetype"] is None
            assert info["forced_weight"] is None

            # 6. Reset history
            db.reset_user_history(user_id=user)
            info = db.get_user_archetype_info(user)
            assert info["taste_archetype_id"] == 27
            assert info["locked_archetype"] is None
            assert info["forced_weight"] is None
    print("Database archetype tests passed!")


def test_engine_dynamic_decay_and_migration() -> None:
    print("Testing Engine dynamic archetype decay, coherence, and migration...")
    config = ServerConfig()
    engine = Engine(config)
    assert engine.faiss.is_ready(), "FAISS must be ready"

    test_user = "test_decay_user_999"
    # Clean any leftover
    engine.db.reset_user_history(user_id=test_user)

    # 1. Cold start state
    state0 = engine.get_user_archetype_state(test_user)
    assert state0["effective_archetype_id"] == 27, f"Expected 27, got {state0['effective_archetype_id']}"
    assert state0["taste_coherence"] == 0.0
    assert state0["archetype_influence"] == 1.0
    assert state0["liked_vectors_count"] == 0

    # 2. Pick top posts from archetype 10 and like them
    arch10_posts = engine.collab.get_archetype_posts(10, limit=5)
    assert len(arch10_posts) > 0, "Archetype 10 should have posts"

    # Like first post
    first_pid = arch10_posts[0]
    engine.record_feedback(test_user, first_pid, "like")

    state1 = engine.get_user_archetype_state(test_user)
    assert state1["liked_vectors_count"] == 1
    # 1 post: coherence R = 1.0, alpha = 1 / (1 + 1*1/3) = 0.75
    assert abs(state1["taste_coherence"] - 1.0) < 1e-3
    assert abs(state1["archetype_influence"] - 0.75) < 1e-2

    # Like 4 more posts from archetype 10
    for pid in arch10_posts[1:]:
        engine.record_feedback(test_user, pid, "like")

    state5 = engine.get_user_archetype_state(test_user)
    assert state5["liked_vectors_count"] == len(arch10_posts)
    # Influence should decay significantly (e.g. < 0.45)
    assert state5["archetype_influence"] < 0.50
    # Dynamic archetype should now have migrated toward archetype 10!
    assert state5["taste_archetype_id"] == 10, f"Expected taste_archetype_id to be 10, got {state5['taste_archetype_id']}"

    # 3. Test tester forced weight override
    engine.db.set_user_archetype_override(test_user, locked_archetype=None, forced_weight=0.90)
    state_forced = engine.get_user_archetype_state(test_user)
    assert abs(state_forced["archetype_influence"] - 0.90) < 1e-3
    # auto influence is still computed in background
    assert state_forced["auto_archetype_influence"] < 0.50

    # 4. Test tester locked archetype
    engine.db.set_user_archetype_override(test_user, locked_archetype=55, forced_weight=None)
    state_locked = engine.get_user_archetype_state(test_user)
    assert state_locked["effective_archetype_id"] == 55
    assert state_locked["locked_archetype"] == 55

    # 5. Clean up
    engine.db.reset_user_history(user_id=test_user)
    state_cleaned = engine.get_user_archetype_state(test_user)
    assert state_cleaned["effective_archetype_id"] == 27
    assert state_cleaned["archetype_influence"] == 1.0
    print("Engine dynamic decay and migration tests passed!")


def test_api_archetype_endpoints() -> None:
    print("Testing API archetype endpoints (/override, /profile, /archetypes)...")
    with TestClient(app) as client:
        # System archetypes
        resp = client.get("/api/v1/system/archetypes")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_archetypes"] == 64
        assert len(data["archetypes"]) == 64

        # Archetype 27 detail
        resp27 = client.get("/api/v1/system/archetypes/27")
        assert resp27.status_code == 200
        d27 = resp27.json()
        assert d27["archetype_id"] == 27
        assert len(d27["items"]) > 0

        user = "tester_api_user_77"
        # Reset user first
        client.post("/api/v1/user/reset", json={"user_id": user})

        # Check profile cold start
        prof = client.get(f"/api/v1/user/{user}/profile").json()
        assert prof["taste_archetype_id"] == 27
        assert prof["effective_archetype_id"] == 27
        assert prof["archetype_influence"] == 1.0
        assert prof["taste_coherence"] == 0.0

        # Override archetype
        over_resp = client.post(
            "/api/v1/user/archetype/override",
            json={"user_id": user, "locked_archetype": 33, "forced_weight": 0.42},
        )
        assert over_resp.status_code == 200
        over_data = over_resp.json()
        assert over_data["locked_archetype"] == 33
        assert over_data["forced_weight"] == 0.42
        assert over_data["effective_archetype_id"] == 33
        assert over_data["archetype_influence"] == 0.42

        # Check profile reflects override
        prof_over = client.get(f"/api/v1/user/{user}/profile").json()
        assert prof_over["locked_archetype"] == 33
        assert prof_over["forced_archetype_weight"] == 0.42
        assert prof_over["effective_archetype_id"] == 33
        assert prof_over["archetype_influence"] == 0.42

        # Reset override to auto
        clear_resp = client.post(
            "/api/v1/user/archetype/override",
            json={"user_id": user, "locked_archetype": None, "forced_weight": None},
        )
        assert clear_resp.status_code == 200
        clear_data = clear_resp.json()
        assert clear_data["locked_archetype"] is None
        assert clear_data["forced_weight"] is None

        # Clean up
        client.post("/api/v1/user/reset", json={"user_id": user})
    print("API archetype endpoints passed!")


def test_feedback_history_and_feed_pagination() -> None:
    print("Testing feedback history API and feed has_more pagination...")
    with TestClient(app) as client:
        user = "test_feedback_user_88"
        client.post("/api/v1/user/reset", json={"user_id": user})

        # 1. Test feed has_more
        feed_resp = client.post("/api/v1/recommend/feed", json={"user_id": user, "limit": 30, "cursor": 0})
        assert feed_resp.status_code == 200, f"Feed failed: {feed_resp.text}"
        feed_data = feed_resp.json()
        assert len(feed_data["items"]) > 0, "Feed returned 0 items"
        assert feed_data["has_more"] is True, "Feed has_more should be True for first page with deep pool"

        # 2. Record likes and dislikes
        pids = [item["id"] for item in feed_data["items"][:5]]
        client.post("/api/v1/feedback", json={"user_id": user, "post_id": pids[0], "signal_type": "like"})
        client.post("/api/v1/feedback", json={"user_id": user, "post_id": pids[1], "signal_type": "like"})
        client.post("/api/v1/feedback", json={"user_id": user, "post_id": pids[2], "signal_type": "dislike"})

        # 3. Test GET /feedback
        all_fb = client.get(f"/api/v1/user/{user}/feedback").json()
        assert all_fb["total"] == 3, f"Expected 3 feedback entries, got {all_fb['total']}"
        assert len(all_fb["items"]) == 3

        likes_fb = client.get(f"/api/v1/user/{user}/feedback?signal_type=like").json()
        assert likes_fb["total"] == 2
        assert len(likes_fb["items"]) == 2
        assert all(item["signal_type"] == "like" for item in likes_fb["items"])
        # Check enriched metadata presence
        assert "score" in likes_fb["items"][0]
        assert "rating" in likes_fb["items"][0]

        dislikes_fb = client.get(f"/api/v1/user/{user}/feedback?signal_type=dislike").json()
        assert dislikes_fb["total"] == 1
        assert dislikes_fb["items"][0]["post_id"] == pids[2]

        # Clean up
        client.post("/api/v1/user/reset", json={"user_id": user})
    print("Feedback history and feed pagination tests passed!")


if __name__ == "__main__":
    test_database_archetype_methods()
    test_engine_dynamic_decay_and_migration()
    test_api_archetype_endpoints()
    test_feedback_history_and_feed_pagination()
    print("ALL DYNAMIC ARCHETYPE TESTS PASSED SUCCESSFULLY!")
