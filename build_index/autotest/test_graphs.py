# autotest/test_graphs.py
"""
Unit test for iterative Tarjan SCC implementation in graphs.py.
Verifies that graphs with deep recursion depths (> 100,000 vertices) execute
safely on Windows without stack overflow or recursion errors.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to sys.path so build_index can be imported
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_index.graphs import tarjan_scc


def test_basic_scc() -> None:
    print("Testing basic SCC functionality...")
    nodes = ["a", "b", "c", "d", "e"]
    edges = {
        "a": ["b"],
        "b": ["c"],
        "c": ["a", "d"],
        "d": ["e"],
        "e": []
    }
    sccs = tarjan_scc(nodes, edges)
    sccs_sorted = sorted([sorted(c) for c in sccs])
    assert sccs_sorted == [["a", "b", "c"], ["d"], ["e"]], f"Unexpected SCCs: {sccs_sorted}"
    print("  [PASS] Basic SCC test passed.")


def test_disconnected_nodes() -> None:
    print("Testing disconnected vertices...")
    nodes = ["x", "y", "z"]
    edges = {}
    sccs = tarjan_scc(nodes, edges)
    sccs_sorted = sorted([sorted(c) for c in sccs])
    assert sccs_sorted == [["x"], ["y"], ["z"]], f"Unexpected SCCs: {sccs_sorted}"
    print("  [PASS] Disconnected vertices test passed.")


def test_deep_chain() -> None:
    n = 150_000
    print(f"Testing deep linear dependency chain of {n:,} vertices (anti-recursion test)...")
    t0 = time.perf_counter()
    nodes = [f"v_{i}" for i in range(n)]
    edges = {f"v_{i}": [f"v_{i+1}"] for i in range(n - 1)}

    sccs = tarjan_scc(nodes, edges)
    duration = time.perf_counter() - t0

    assert len(sccs) == n, f"Expected {n} components, got {len(sccs)}"
    print(f"  [PASS] Deep chain processed in {duration:.2f}s without recursion error.")


def test_deep_chain_with_cycles() -> None:
    n = 100_000
    cycle_size = 5_000
    print(f"Testing deep chain ({n:,} vertices) containing a {cycle_size:,}-vertex cycle...")
    t0 = time.perf_counter()
    nodes = [f"u_{i}" for i in range(n)]
    edges = {f"u_{i}": [f"u_{i+1}"] for i in range(n - 1)}

    # Close a cycle between u_10000 and u_(10000 + cycle_size - 1)
    start_c = 10_000
    end_c = start_c + cycle_size - 1
    edges[f"u_{end_c}"].append(f"u_{start_c}")

    sccs = tarjan_scc(nodes, edges)
    duration = time.perf_counter() - t0

    sizes = sorted(len(c) for c in sccs)
    assert sizes[-1] == cycle_size, f"Expected largest SCC of size {cycle_size}, got {sizes[-1]}"
    print(f"  [PASS] Deep chain with cycle processed in {duration:.2f}s.")


def main() -> int:
    print("=" * 60)
    print("TIRESIAS_ENGINE - Graph Algorithm Unit Tests")
    print("=" * 60)
    try:
        test_basic_scc()
        test_disconnected_nodes()
        test_deep_chain()
        test_deep_chain_with_cycles()
        print("-" * 60)
        print("ALL GRAPH TESTS PASSED SUCCESSFULLY!")
        print("-" * 60)
        return 0
    except Exception as exc:
        print(f"\n[FAIL] Test failed with error: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
