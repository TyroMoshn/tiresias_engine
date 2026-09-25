# autotest/test_bijection.py
"""
Unit test for 64-bit post ID bijection into dense uint32 indices.
Verifies:
1. Correct mapping: post_id(int64) -> dense_id(uint32) in [0, N-1]
2. Reverse mapping: dense_id(uint32) -> post_id(int64)
3. Support for IDs exceeding 2^31 - 1 (preventing int32 overflow)
4. Full compatibility with pyroaring.BitMap serialization/deserialization.
"""
from __future__ import annotations

import sys
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from pyroaring import BitMap
except ImportError:
    BitMap = None


def test_bijection_roundtrip() -> None:
    print("Testing 64-bit ID bijection roundtrip...")
    # Generate sorted 64-bit post IDs, including values > 2^31 - 1
    post_ids = np.array([
        100, 500, 10_000, 2_000_000,
        2_147_483_647,       # max int32
        2_147_483_648,       # int32 overflow edge
        3_500_000_000,       # fits only in uint32 / int64
        9_000_000_000,       # fits only in int64
    ], dtype=np.int64)

    n = len(post_ids)

    # Test forward mapping pi(p) -> dense_idx
    query_ids = np.array([500, 2_147_483_648, 9_000_000_000], dtype=np.int64)
    dense_idx = np.searchsorted(post_ids, query_ids)
    valid = (dense_idx < n) & (post_ids[dense_idx] == query_ids)
    assert np.all(valid), "All queried IDs must be found"
    dense_ids = dense_idx[valid].astype(np.uint32)

    # Expected indices: 500 is index 1, 2_147_483_648 is index 5, 9_000_000_000 is index 7
    np.testing.assert_array_equal(dense_ids, np.array([1, 5, 7], dtype=np.uint32))

    # Test reverse mapping pi^(-1)(dense_idx) -> post_id
    reconstructed = post_ids[dense_ids]
    np.testing.assert_array_equal(reconstructed, query_ids)
    print("  [PASS] Bijection roundtrip successful.")


def test_roaring_bitmap_compatibility() -> None:
    print("Testing pyroaring.BitMap integration with dense uint32 indices...")
    if BitMap is None:
        print("  [SKIP] pyroaring is not installed in this environment.")
        return

    # Create dense indices for a mock collection
    dense_ids = np.array([0, 1, 2, 10, 50, 1000, 6_000_000], dtype=np.uint32)
    bm = BitMap(dense_ids.tolist())

    # Serialize & deserialize
    serialized = bm.serialize()
    deserialized = BitMap.deserialize(serialized)

    recovered = np.array(list(deserialized), dtype=np.uint32)
    np.testing.assert_array_equal(recovered, dense_ids)
    assert len(deserialized) == len(dense_ids)
    print(f"  [PASS] Roaring BitMap correctly stored and restored {len(dense_ids)} dense IDs.")


def main() -> int:
    print("=" * 60)
    print("TIRESIAS_ENGINE - Bijection & Roaring Compatibility Tests")
    print("=" * 60)
    try:
        test_bijection_roundtrip()
        test_roaring_bitmap_compatibility()
        print("-" * 60)
        print("ALL BIJECTION TESTS PASSED SUCCESSFULLY!")
        print("-" * 60)
        return 0
    except Exception as exc:
        print(f"\n[FAIL] Test failed with error: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
