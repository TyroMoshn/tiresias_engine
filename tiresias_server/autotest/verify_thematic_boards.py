#!/usr/bin/env python3
"""
Semantic verification script for Phase 4 Boards Engine.
Evaluates user-curated thematic collections (Food & Elaborate Lighting).
Demonstrates:
  1. Missing ID handling with WARNING.
  2. Board Centroid vector calculation via in-memory FAISS SQ8 reconstruct.
  3. Post Affinity ranking (core defining images vs outliers).
  4. Salient Tags extraction via TF-IDF.
  5. 'More Like This Board' recommendations with 100% hard rejection of seed posts.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVER_DIR = HERE.parent
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from app.config import ServerConfig
from app.core.engine import Engine
from app.db.database import Database


def run_thematic_verification() -> None:
    samples_file = HERE / "thematic_samples.json"
    if not samples_file.exists():
        print(f"Error: {samples_file} not found")
        return

    with open(samples_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_db = Path(tmp_dir) / "thematic_test.db"
        cfg = ServerConfig(custom_db_path=test_db)

        print("=" * 70)
        print("TIRESIAS_ENGINE: Thematic Boards Semantic Verification")
        print("=" * 70)

        with Engine(cfg) as engine:
            if not engine.faiss.is_ready():
                print("[WARN] FAISS index is not loaded, skipping vector evaluation.")
                return

            for key, info in samples.items():
                title = info["title"]
                desc = info["description"]
                pids = info["post_ids"]

                print(f"\n>>> THEME: {title.upper()}")
                print(f"    Description: {desc}")
                print(f"    Input Seed IDs ({len(pids)} total): {pids}")

                # 1. Create board
                board = engine.db.create_board("thematic_tester", name=title, description=desc)
                bid = board["board_id"]
                engine.db.add_posts_to_board(bid, pids)

                # 2. Compute Centroid & check presence
                centroid, valid_pids, missing_pids = engine.boards.compute_board_centroid(pids)
                print(f"\n  [Data Coverage]")
                print(f"    - Valid in FAISS Index: {len(valid_pids)} posts {valid_pids}")
                print(f"    - Missing (Post-cutoff): {len(missing_pids)} posts {missing_pids}")

                # 3. Post Affinity Ranking (Core vs Outliers)
                posts_affinity, _ = engine.boards.get_hydrated_board_posts(
                    bid, sort_by="affinity", order="desc", limit=50
                )
                print(f"\n  [Posts Ranked by Affinity to Board Centroid (Core -> Outliers)]")
                for rank, p in enumerate(posts_affinity, 1):
                    print(
                        f"    #{rank:02d} Post {p['post_id']} | "
                        f"Affinity: {p['affinity']:+.4f} | "
                        f"Score: {p['score']:4d} | Favs: {p['fav_count']:4d} | "
                        f"Rating: {p['rating']}"
                    )

                # 4. Salient Tags Extraction
                salient_tags = engine.boards.extract_salient_tags(valid_pids, top_k=8)
                print(f"\n  [Extracted Salient Tags (TF-IDF)]")
                if salient_tags:
                    for t in salient_tags:
                        print(
                            f"    * {t['tag']:<24} (score: {t['score']:.2f}, tf: {t['tf']:.2f}, idf: {t['idf']:.2f}, in {t['post_count']} posts)"
                        )
                else:
                    print("    (Tags parquet partitions not found or empty)")

                # 5. Recommendations 'More Like This Board'
                t0 = time.perf_counter()
                rec_result = engine.boards.recommend_for_board(
                    board_id=bid, user_id="thematic_tester", limit=8
                )
                rec_time = (time.perf_counter() - t0) * 1000

                print(f"\n  [Recommendations 'More Like This Board' (Latency: {rec_time:.2f} ms)]")
                recs = rec_result.get("items", [])
                seed_set = set(pids)
                for rank, item in enumerate(recs, 1):
                    assert item["post_id"] not in seed_set, "Seed post leaked into recommendations!"
                    print(
                        f"    #{rank:02d} Post {item['post_id']} | "
                        f"Similarity: {item['similarity']:.4f} | "
                        f"Score: {item['score_val']:4d} | Favs: {item['fav_count']:4d} | "
                        f"Rating: {item['rating']}"
                    )

                print("-" * 70)


if __name__ == "__main__":
    run_thematic_verification()
