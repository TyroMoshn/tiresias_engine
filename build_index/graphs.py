# build_index/graphs.py
from __future__ import annotations
from typing import Dict, List, Set

def tarjan_scc(nodes: List[str], edges: Dict[str, List[str]]) -> List[List[str]]:
    """SCC Tarjan на строковых вершинах."""
    import sys
    index = 0
    stack: List[str] = []
    onstack: Set[str] = set()
    idx: Dict[str, int] = {}
    low: Dict[str, int] = {}
    sccs: List[List[str]] = []

    sys.setrecursionlimit(max(10_000, len(nodes) * 2))

    def strongconnect(v: str) -> None:
        nonlocal index
        idx[v] = index
        low[v] = index
        index += 1
        stack.append(v)
        onstack.add(v)
        for w in edges.get(v, []):
            if w not in idx:
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif w in onstack:
                low[v] = min(low[v], idx[w])
        if low[v] == idx[v]:
            comp = []
            while True:
                w = stack.pop()
                onstack.discard(w)
                comp.append(w)
                if w == v:
                    break
            sccs.append(comp)

    for v in nodes:
        if v not in idx:
            strongconnect(v)
    return sccs