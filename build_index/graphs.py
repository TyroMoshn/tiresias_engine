# build_index/graphs.py
from __future__ import annotations
from typing import Dict, List, Set

def tarjan_scc(nodes: List[str], edges: Dict[str, List[str]]) -> List[List[str]]:
    """
    Iterative Tarjan's Strongly Connected Components algorithm.
    Runs in strictly O(|V| + |E|) time and O(|V|) memory on the heap.
    Eliminates recursion depth limits and prevents stack overflows on deep chains.
    """
    index = 0
    idx: Dict[str, int] = {}
    low: Dict[str, int] = {}
    onstack: Set[str] = set()
    stack: List[str] = []
    sccs: List[List[str]] = []

    for root in nodes:
        if root in idx:
            continue

        # Call stack emulation frame: (vertex, outgoing_neighbors, current_neighbor_index)
        call_stack = [(root, edges.get(root, []), 0)]
        idx[root] = low[root] = index
        index += 1
        stack.append(root)
        onstack.add(root)

        while call_stack:
            u, neighbors, i = call_stack[-1]
            if i < len(neighbors):
                v = neighbors[i]
                call_stack[-1] = (u, neighbors, i + 1)
                if v not in idx:
                    idx[v] = low[v] = index
                    index += 1
                    stack.append(v)
                    onstack.add(v)
                    call_stack.append((v, edges.get(v, []), 0))
                elif v in onstack:
                    low[u] = min(low[u], idx[v])
            else:
                # Post-visit return
                call_stack.pop()
                if call_stack:
                    parent = call_stack[-1][0]
                    low[parent] = min(low[parent], low[u])

                if low[u] == idx[u]:
                    comp: List[str] = []
                    while True:
                        w = stack.pop()
                        onstack.discard(w)
                        comp.append(w)
                        if w == u:
                            break
                    sccs.append(comp)

    return sccs