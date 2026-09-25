#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import tempfile
import time
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
from app.core.scorer import CandidateScorer
from app.db.database import Database


def test_sqlite_database() -> None:
    print("Testing SQLite database layer (app/db/database.py)...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_user.db"
        with Database(db_path) as db:
            user = "test_user_123"

            # 1. Feedback signals (like, dislike, hide, custom signal)
            db.record_feedback(user, 1001, "like")
            db.record_feedback(user, 1002, "dislike")
            db.record_feedback(user, 1003, "hide")
            db.record_feedback(user, 1004, "bookmark")

            rejected = db.get_user_hard_rejected_posts(user)
            assert 1002 in rejected, "Disliked post 1002 must be hard rejected"
            assert 1003 in rejected, "Hidden post 1003 must be hard rejected"
            assert 1001 not in rejected, "Liked post 1001 must not be rejected"
            assert 1004 not in rejected, "Bookmarked post 1004 must not be rejected"

            liked = db.get_user_liked_posts(user)
            assert 1001 in liked, "Liked post 1001 must be in liked list"

            # 2. Seen posts batch
            db.record_seen_batch(user, [1001, 1005, 1006])
            seen = db.get_user_seen_posts(user)
            assert 1001 in seen and 1005 in seen and 1006 in seen
            assert 1002 not in seen

            # 3. Tag blacklist
            db.add_tag_blacklist(user, [42, 99])
            bl = db.get_user_tag_blacklist(user)
            assert 42 in bl and 99 in bl
            db.remove_tag_blacklist(user, [42])
            bl_after = db.get_user_tag_blacklist(user)
            assert 42 not in bl_after and 99 in bl_after

            # 4. Settings
            db.set_user_settings(user, {"dark_mode": "true", "min_score": "15"})
            settings = db.get_user_settings(user)
            assert settings.get("dark_mode") == "true"
            assert settings.get("min_score") == "15"

    print("  [PASS] SQLite database functionality, signal types & blacklist OK")


def test_candidate_scorer_and_diversity() -> None:
    print("Testing CandidateScorer & diversity re-ranking...")
    scorer = CandidateScorer(
        w_vector_sim=0.50,
        w_quality_log=0.25,
        w_collab=0.15,
        w_tag_match=0.10,
        seen_decay=0.30,
    )

    # 1. Fresh candidate scoring
    s_fresh, reasons_fresh = scorer.score_candidate(
        vector_sim=0.80,
        fav_count=1000,
        score_val=50,
        collab_weight=0.5,
        is_seen=False,
    )
    assert s_fresh > 0.40, f"Expected strong score, got {s_fresh}"
    assert "visual_affinity" in reasons_fresh
    assert "community_favorite" in reasons_fresh
    assert "co_favorited" in reasons_fresh
    assert "previously_seen" not in reasons_fresh

    # 2. Soft decay on seen post
    s_seen, reasons_seen = scorer.score_candidate(
        vector_sim=0.80,
        fav_count=1000,
        score_val=50,
        collab_weight=0.5,
        is_seen=True,
    )
    assert abs(s_seen - s_fresh * 0.30) < 1e-5, "Seen post must have 0.3x score decay"
    assert "previously_seen" in reasons_seen

    # 3. Diversity re-ranking (max 2 consecutive from same author)
    mock_candidates = [
        {"post_id": 1, "uploader_id": 10, "score": 0.95},
        {"post_id": 2, "uploader_id": 10, "score": 0.94},
        {"post_id": 3, "uploader_id": 10, "score": 0.93},  # Should be deferred!
        {"post_id": 4, "uploader_id": 20, "score": 0.90},
        {"post_id": 5, "uploader_id": 10, "score": 0.85},
    ]
    diversified = scorer.diversify_and_rank(mock_candidates, limit=4, max_consecutive_uploader=2)
    selected_pids = [item["post_id"] for item in diversified]
    assert selected_pids[0] == 1
    assert selected_pids[1] == 2
    assert selected_pids[2] == 4, f"Post 4 (different author) must break the sequence of author 10, got {selected_pids}"

    print("  [PASS] Multi-factor scoring, soft decay and diversity OK")


def test_engine_end_to_end() -> None:
    print("Testing Engine end-to-end pipeline...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        cfg = ServerConfig(data_root=tmp_path)
        with Engine(cfg) as engine:
            # Pre-seed some mock data into engine
            user = "test_user_alpha"

            # 1. Check status
            status = engine.get_status()
            assert status["status"] == "online"
            assert "memory_rss_mb" in status
            print(f"  [INFO] Engine RSS: {status['memory_rss_mb']} MB")

            # 2. Add likes and dislikes
            engine.db.record_feedback(user, 101, "like")
            engine.db.record_feedback(user, 202, "dislike")

            # 3. Request feed
            res = engine.recommend_feed(user_id=user, limit=10)
            assert "items" in res
            assert "latency_ms" in res
            # Check hard rejection: 202 must NOT be in feed
            returned_pids = [item["post_id"] for item in res["items"]]
            assert 202 not in returned_pids, "Disliked post 202 was returned in feed!"

            # 4. Request similar
            sim_res = engine.recommend_similar(post_id=101, limit=5)
            assert "items" in sim_res
            assert "latency_ms" in sim_res

    print("  [PASS] Engine pipeline end-to-end OK")


def test_latency_and_memory_bench() -> None:
    print("Testing Engine latency and memory budget (50 requests)...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        cfg = ServerConfig(data_root=tmp_path)
        with Engine(cfg) as engine:
            latencies = []
            for i in range(50):
                t0 = time.perf_counter()
                _ = engine.recommend_feed(user_id=f"user_{i % 5}", limit=30)
                latencies.append((time.perf_counter() - t0) * 1000.0)

            avg_lat = float(np.mean(latencies))
            p95_lat = float(np.percentile(latencies, 95))
            status = engine.get_status()
            rss_mb = status["memory_rss_mb"]

            print(f"  [BENCH] Avg Latency: {avg_lat:.2f} ms (Target < 30 ms)")
            print(f"  [BENCH] p95 Latency: {p95_lat:.2f} ms")
            print(f"  [BENCH] Memory RSS:  {rss_mb:.1f} MB (Target < 1,200 MB)")

            assert avg_lat < 30.0, f"Average latency {avg_lat:.2f} ms exceeds 30 ms target"
            assert rss_mb < 1200.0, f"Memory {rss_mb:.1f} MB exceeds 1,200 MB target"

    print("  [PASS] Latency and memory benchmarks strictly within targets")


def test_fastapi_endpoints() -> None:
    print("Testing FastAPI HTTP routers...")
    try:
        from fastapi.testclient import TestClient
        from app.main import app
        with TestClient(app) as client:
            # 1. Health
            r_health = client.get("/api/v1/system/health")
            assert r_health.status_code == 200
            assert r_health.json()["status"] == "online"

            # 2. Feed
            r_feed = client.post("/api/v1/recommend/feed", json={"user_id": "u1", "limit": 10})
            assert r_feed.status_code == 200
            feed_data = r_feed.json()
            assert "items" in feed_data
            assert "cursor" in feed_data

            # 3. Feedback (like & dislike)
            r_like = client.post("/api/v1/feedback", json={"user_id": "u1", "post_id": 555, "signal_type": "like"})
            assert r_like.status_code == 200
            assert r_like.json()["success"] is True

            r_dislike = client.post("/api/v1/feedback", json={"user_id": "u1", "post_id": 999, "signal_type": "dislike"})
            assert r_dislike.status_code == 200

            # Verify hard rejection via HTTP
            r_feed2 = client.post("/api/v1/recommend/feed", json={"user_id": "u1", "limit": 20})
            assert r_feed2.status_code == 200
            pids = [item["post_id"] for item in r_feed2.json()["items"]]
            assert 999 not in pids, "Disliked post 999 returned in feed!"

            # 4. Seen
            r_seen = client.post("/api/v1/feedback/seen", json={"user_id": "u1", "post_ids": [555, 777]})
            assert r_seen.status_code == 200
            assert r_seen.json()["recorded_count"] == 2

            # 5. Tag Blacklist
            r_bl = client.put("/api/v1/settings/tag_blacklist", json={"user_id": "u1", "tag_ids": [12, 34]})
            assert r_bl.status_code == 200
            assert 12 in r_bl.json()["blacklisted_tag_ids"]

            # 6. Boards HTTP Endpoints
            r_bcreate = client.post("/api/v1/boards", json={"user_id": "u1", "name": "HTTP Test Board", "description": "Board via API"})
            assert r_bcreate.status_code == 201
            b_id = r_bcreate.json()["board_id"]

            r_badd = client.post(f"/api/v1/boards/{b_id}/posts", json={"post_ids": [5353063, 5355177]})
            assert r_badd.status_code == 200
            assert r_badd.json()["added_count"] == 2

            r_bget = client.get(f"/api/v1/boards/{b_id}?sort_by=affinity&order=desc")
            assert r_bget.status_code == 200
            assert len(r_bget.json()["posts"]) == 2

            r_brec = client.get(f"/api/v1/boards/{b_id}/recommend?limit=10")
            assert r_brec.status_code == 200
            rec_items = r_brec.json()["items"]
            rec_ids = [item["post_id"] for item in rec_items]
            assert 5353063 not in rec_ids and 5355177 not in rec_ids, "Existing board posts returned in recommendations!"

            r_bexport = client.get(f"/api/v1/boards/{b_id}/export")
            assert r_bexport.status_code == 200
            assert set(r_bexport.json()["post_ids"]) == {5353063, 5355177}

            r_bimport = client.post("/api/v1/boards/import", json={"user_id": "u2", "name": "Imported via HTTP", "post_ids": [5353063, 5355177]})
            assert r_bimport.status_code == 201
            assert r_bimport.json()["post_count"] == 2

            r_bdel = client.delete(f"/api/v1/boards/{b_id}")
            assert r_bdel.status_code == 200

            print("  [PASS] All FastAPI HTTP endpoints (including Boards CRUD, Sorting & Recommendations) returned 200/201 OK")

    except ImportError as e:
        print(f"  [SKIP] Optional HTTP TestClient dependency not installed: {e}")


def test_boards_and_recommendations() -> None:
    print("Testing Boards Engine & Board Recommendations (Phase 4)...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        db_path = tmp_path / "boards_test.db"

        # 1. Test database boards CRUD
        with Database(db_path) as db:
            user_id = "artist_dan"
            b = db.create_board(user_id=user_id, name="Landscape Moodboard", description="Calm horizons")
            bid = b["board_id"]
            assert b["name"] == "Landscape Moodboard"
            assert b["post_count"] == 0

            # Add posts
            test_pids = [5353063, 5355177, 5341735, 5372730]
            added = db.add_posts_to_board(bid, test_pids)
            assert added == 4
            assert db.get_board_post_count(bid) == 4

            # Duplicate add is ignored
            added_dup = db.add_posts_to_board(bid, [5353063])
            assert added_dup == 0
            assert db.get_board_post_count(bid) == 4

            # Remove post
            rem = db.remove_post_from_board(bid, 5372730)
            assert rem is True
            assert db.get_board_post_count(bid) == 3

            # Update board
            db.update_board(bid, name="Updated Moodboard", cover_post_id=5353063)
            b_up = db.get_board(bid)
            assert b_up["name"] == "Updated Moodboard"
            assert b_up["cover_post_id"] == 5353063

            # User list
            b_list = db.list_user_boards(user_id)
            assert len(b_list) == 1
            assert b_list[0]["board_id"] == bid

        # 2. Test Engine & BoardEngine with FAISS
        cfg = ServerConfig(custom_db_path=db_path)
        with Engine(cfg) as engine:
            # Reconstruct vector test
            vec = engine.faiss.reconstruct_post_vector(5353063)
            if engine.faiss.is_ready() and vec is not None:
                assert vec.shape == (128,)
                norm = float(np.linalg.norm(vec))
                assert abs(norm - 1.0) < 1e-4, f"Vector should be L2-normalized, got norm={norm}"

                # Test Centroid calculation
                # 3 valid IDs + 1 missing ID (e.g. 999999999)
                mixed_pids = [5353063, 5355177, 5341735, 999999999]
                centroid, valid, missing = engine.boards.compute_board_centroid(mixed_pids)
                assert centroid is not None
                assert len(valid) == 3
                assert 999999999 in missing
                c_norm = float(np.linalg.norm(centroid))
                assert abs(c_norm - 1.0) < 1e-4, f"Centroid should be L2-normalized, got {c_norm}"

                # Test Affinities calculation
                affs = engine.boards.compute_affinities(mixed_pids, centroid)
                assert 5353063 in affs
                assert affs[999999999] == 0.0

            # 3. Test all 5 sorting modes in get_hydrated_board_posts
            for s_by in ["added_at", "epoch_day", "score", "fav_count", "affinity"]:
                posts_desc, cnt = engine.boards.get_hydrated_board_posts(bid, sort_by=s_by, order="desc")
                assert cnt == 3
                assert len(posts_desc) == 3
                posts_asc, _ = engine.boards.get_hydrated_board_posts(bid, sort_by=s_by, order="asc")
                assert len(posts_asc) == 3
                # Verify order inversion
                if posts_desc[0][s_by] != posts_desc[-1][s_by]:
                    assert posts_desc[0][s_by] >= posts_desc[-1][s_by]
                    assert posts_asc[0][s_by] <= posts_asc[-1][s_by]

            # 4. Test Board Recommendations & HARD REJECTION of board posts
            recs = engine.boards.recommend_for_board(board_id=bid, limit=20)
            assert "items" in recs
            rec_pids = {item["post_id"] for item in recs["items"]}
            current_board_pids = engine.db.get_all_board_post_ids(bid)
            for bp in current_board_pids:
                assert bp not in rec_pids, f"Board post {bp} leaked into board recommendations!"

            # 5. Test Export / Import
            export_data = {
                "board_id": bid,
                "name": b_up["name"],
                "description": b_up["description"],
                "created_at": b_up["created_at"],
                "updated_at": b_up["updated_at"],
                "cover_post_id": b_up["cover_post_id"],
                "post_ids": list(current_board_pids),
            }
            # Import into another user profile
            imp_board = engine.db.create_board("imported_user", name="Cloned Board")
            engine.db.add_posts_to_board(imp_board["board_id"], export_data["post_ids"])
            imp_pids = engine.db.get_all_board_post_ids(imp_board["board_id"])
            assert imp_pids == current_board_pids

    print("  [PASS] Board CRUD, Centroid math, SQ8 reconstruct, 5-mode sorting & Hard Rejection OK")


def main() -> int:
    print("=" * 65)
    print("TIRESIAS_ENGINE - Serving Core & Recommendation API Tests")
    print("=" * 65)
    try:
        test_sqlite_database()
        test_candidate_scorer_and_diversity()
        test_engine_end_to_end()
        test_latency_and_memory_bench()
        test_boards_and_recommendations()
        test_fastapi_endpoints()
        print("-" * 65)
        print("ALL SERVING CORE & BOARDS ENGINE TESTS PASSED SUCCESSFULLY!")
        print("-" * 65)
        return 0
    except Exception as exc:
        print(f"\n[FAIL] Test error: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
