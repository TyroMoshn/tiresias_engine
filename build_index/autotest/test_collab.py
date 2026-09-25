#!/usr/bin/env python3
"""
Unit tests for Phase 2: Collaborative Filtering & Taste Personas.
Tests:
1. Postgres array string parsing (_parse_post_ids_field)
2. Co-favorited graph mathematical formula W_cofav(a, b) = sum 1 / sqrt(|F_u|)
3. User taste vector calculation and Spherical K-Means clustering
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

# Add project root to sys.path
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_index.config import Config
from build_index.collab_stage import _parse_post_ids_field, step_collab_favs


def test_parse_post_ids() -> None:
    print("Testing _parse_post_ids_field...")
    assert _parse_post_ids_field("") == []
    assert _parse_post_ids_field("{}") == []
    assert _parse_post_ids_field("  {}  ") == []
    assert _parse_post_ids_field("{1,2,3}") == [1, 2, 3]
    assert _parse_post_ids_field("{3, 1, 2, 2}") == [1, 2, 3]
    assert _parse_post_ids_field("{100, bad, 200}") == [100, 200]
    print("  [PASS] Post IDs parsing OK")


def test_cofav_math() -> None:
    print("Testing co-favorited mathematical weights...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        csv_path = tmp_path / "user_favorites.csv"
        out_parquet = tmp_path / "post_cofav.parquet"

        # Synthetic user favorites:
        # User 1: {10, 20, 30} (|F_1| = 3, w = 1 / sqrt(3) ~= 0.57735)
        # User 2: {20, 30, 40} (|F_2| = 3, w = 1 / sqrt(3) ~= 0.57735)
        # User 3: {10, 20}     (|F_3| = 2, w = 1 / sqrt(2) ~= 0.70711)
        csv_content = (
            "user_id,favorite_count,favorites\n"
            '1,3,"{10,20,30}"\n'
            '2,3,"{20,30,40}"\n'
            '3,2,"{10,20}"\n'
        )
        csv_path.write_text(csv_content, encoding="utf-8")

        cfg = Config(
            root=tmp_path,
            csv=tmp_path / "posts.csv",
            posts_parquet=tmp_path / "posts_parquet",
            post_tags_parquet=tmp_path / "post_tags_parquet",
            bitmaps_dir=tmp_path / "bitmaps",
            topk_dir=tmp_path / "topk",
            features_dir=tmp_path,
            tags_parquet=tmp_path / "tags.parquet",
            tags_csv=tmp_path / "tags.csv",
            tag_aliases_csv=tmp_path / "tag_aliases.csv",
            tag_implications_csv=tmp_path / "tag_implications.csv",
            pools_csv=tmp_path / "pools.csv",
            pools_parquet=tmp_path / "pools_parquet",
            pools_meta_parquet=tmp_path / "pools_meta.parquet",
            pools_entropy_parquet=tmp_path / "pool_entropy.parquet",
            pools_edges_parquet=tmp_path / "pool_edges.parquet",
            tag_co_from_pools_parquet=tmp_path / "tag_co_from_pools.parquet",
            mmaps_dir=tmp_path / "mmaps",
            user_favorites_csv=csv_path,
            post_cofav_parquet=out_parquet,
            fav_user_min_posts=2,
            cofav_min_weight=0.1,
            cofav_max_edges_per_post=10,
        )

        step_collab_favs(cfg)
        assert out_parquet.exists(), "post_cofav.parquet was not created"

        df = pl.read_parquet(out_parquet)
        pairs = {}
        for rec in df.iter_rows(named=True):
            a, b, w = int(rec["a_post_id"]), int(rec["b_post_id"]), float(rec["weight"])
            pairs[(a, b)] = w

        # Expected weights:
        # Pair (20, 30): user 1 + user 2 = 1/sqrt(3) + 1/sqrt(3) ~= 1.1547
        # Pair (10, 20): user 1 + user 3 = 1/sqrt(3) + 1/sqrt(2) ~= 1.2845
        # Pair (10, 30): user 1 = 1/sqrt(3) ~= 0.5774
        # Pair (20, 40): user 2 = 1/sqrt(3) ~= 0.5774
        # Pair (30, 40): user 2 = 1/sqrt(3) ~= 0.5774
        expected_20_30 = 1.0 / np.sqrt(3) + 1.0 / np.sqrt(3)
        expected_10_20 = 1.0 / np.sqrt(3) + 1.0 / np.sqrt(2)

        assert (20, 30) in pairs, "Pair (20, 30) missing"
        assert abs(pairs[(20, 30)] - expected_20_30) < 1e-4, f"Weight mismatch for (20, 30): {pairs[(20, 30)]} vs {expected_20_30}"

        assert (10, 20) in pairs, "Pair (10, 20) missing"
        assert abs(pairs[(10, 20)] - expected_10_20) < 1e-4, f"Weight mismatch for (10, 20): {pairs[(10, 20)]} vs {expected_10_20}"

        print(f"  [PASS] Co-favorited weights match formula exactly: W(20,30)={pairs[(20,30)]:.4f}, W(10,20)={pairs[(10,20)]:.4f}")


def test_spherical_kmeans() -> None:
    print("Testing Spherical K-Means clustering...")
    from sklearn.cluster import MiniBatchKMeans

    # Generate 500 synthetic user vectors in 128 dimensions clustered around 4 modes
    np.random.seed(42)
    dim = 128
    true_centers = np.random.randn(4, dim).astype(np.float32)
    true_centers /= np.linalg.norm(true_centers, axis=1, keepdims=True)

    users = []
    for _ in range(500):
        c = true_centers[np.random.randint(0, 4)]
        noise = np.random.randn(dim).astype(np.float32) * 0.1
        v = c + noise
        v /= np.linalg.norm(v)
        users.append(v)

    U_matrix = np.asarray(users, dtype=np.float32)

    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=256, random_state=42, n_init=3)
    kmeans.fit(U_matrix)
    centroids = kmeans.cluster_centers_.astype(np.float32)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / norms

    # Verify unit sphere normalization
    centroid_norms = np.linalg.norm(centroids, axis=1)
    for i, norm in enumerate(centroid_norms):
        assert abs(norm - 1.0) < 1e-5, f"Centroid {i} is not unit norm: {norm}"

    print(f"  [PASS] K-Means clustered into 4 centroids, all unit norm (norms={centroid_norms[:2]}...)")


def test_taste_archetypes_end_to_end() -> None:
    print("Testing step_taste_archetypes end-to-end...")
    from build_index.collab_stage import step_taste_archetypes

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        features_path = tmp_path / "features"
        features_path.mkdir(parents=True, exist_ok=True)

        fav_csv = tmp_path / "user_favorites.csv"
        vec_parquet = features_path / "post2vec.parquet"
        out_archetypes = features_path / "taste_archetypes.parquet"
        out_centroids = features_path / "taste_centroids.npy"

        # Create 50 synthetic posts with 128-d vectors
        dim = 128
        np.random.seed(42)
        pids = list(range(101, 151))
        vecs = np.random.randn(len(pids), dim).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

        pl.DataFrame({
            "post_id": pids,
            "vec": [v.tolist() for v in vecs],
        }).write_parquet(vec_parquet, compression="zstd")

        # Create 20 synthetic users
        lines = ["user_id,favorite_count,favorites"]
        for uid in range(1, 21):
            sample_pids = np.random.choice(pids, size=8, replace=False).tolist()
            lines.append(f'{uid},{len(sample_pids)},"{{{",".join(map(str, sample_pids))}}}"')
        fav_csv.write_text("\n".join(lines), encoding="utf-8")

        cfg = Config(
            root=tmp_path,
            csv=tmp_path / "posts.csv",
            posts_parquet=tmp_path / "posts_parquet",
            post_tags_parquet=tmp_path / "post_tags_parquet",
            bitmaps_dir=tmp_path / "bitmaps",
            topk_dir=tmp_path / "topk",
            features_dir=features_path,
            tags_parquet=tmp_path / "tags.parquet",
            tags_csv=tmp_path / "tags.csv",
            tag_aliases_csv=tmp_path / "tag_aliases.csv",
            tag_implications_csv=tmp_path / "tag_implications.csv",
            pools_csv=tmp_path / "pools.csv",
            pools_parquet=tmp_path / "pools_parquet",
            pools_meta_parquet=tmp_path / "pools_meta.parquet",
            pools_entropy_parquet=tmp_path / "pool_entropy.parquet",
            pools_edges_parquet=tmp_path / "pool_edges.parquet",
            tag_co_from_pools_parquet=tmp_path / "tag_co_from_pools.parquet",
            mmaps_dir=tmp_path / "mmaps",
            user_favorites_csv=fav_csv,
            taste_archetypes_parquet=out_archetypes,
            taste_centroids_npy=out_centroids,
            taste_archetypes_k=4,
            taste_archetypes_top_posts=10,
            fav_user_min_posts=3,
        )

        step_taste_archetypes(cfg)

        assert out_centroids.exists(), "taste_centroids.npy was not created"
        centroids = np.load(out_centroids)
        assert centroids.shape == (4, dim), f"Centroids shape mismatch: {centroids.shape}"

        assert out_archetypes.exists(), "taste_archetypes.parquet was not created"
        arch_df = pl.read_parquet(out_archetypes)
        assert arch_df.height == 40, f"Expected 4 * 10 = 40 rows, got {arch_df.height}"
        assert set(arch_df.columns) == {"archetype_id", "post_id", "rank", "score"}

        print(f"  [PASS] step_taste_archetypes generated {centroids.shape} centroids and {arch_df.height} archetype top posts")


def main() -> int:
    print("=" * 60)
    print("TIRESIAS_ENGINE - Collaborative Filtering Unit Tests")
    print("=" * 60)
    try:
        test_parse_post_ids()
        test_cofav_math()
        test_spherical_kmeans()
        test_taste_archetypes_end_to_end()
        print("-" * 60)
        print("ALL COLLAB TESTS PASSED SUCCESSFULLY!")
        print("-" * 60)
        return 0
    except Exception as exc:
        print(f"\n[FAIL] Test error: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
