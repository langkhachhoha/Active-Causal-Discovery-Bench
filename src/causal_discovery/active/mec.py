"""Markov-equivalence-class enumeration over a (C)PDAG.

Two DAGs are Markov equivalent iff they share a skeleton and a set of unshielded
colliders (Verma & Pearl, 1990). We therefore enumerate acyclic orientations of the
undirected part that (a) keep the already-directed edges and (b) reproduce the
reference v-structure set.

For the graph sizes used in these studies (d <= 10) the exhaustive path is cheap.
A constructive sampler is used as a fallback when the undirected part is large.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from causal_discovery.core import DAG
from causal_discovery.equivalence import CPDAG
from causal_discovery.active.pdag import dag_v_structures, meek_closure, v_structures

EXHAUSTIVE_LIMIT_BITS = 16  # 2**16 = 65536 orientations


def _is_acyclic(num_nodes: int, edges: set[tuple[int, int]]) -> bool:
    indegree = [0] * num_nodes
    children: list[list[int]] = [[] for _ in range(num_nodes)]
    for src, dst in edges:
        indegree[dst] += 1
        children[src].append(dst)
    stack = [n for n in range(num_nodes) if indegree[n] == 0]
    visited = 0
    while stack:
        node = stack.pop()
        visited += 1
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                stack.append(child)
    return visited == num_nodes


def _enumerate_exhaustive(
    pdag: CPDAG, reference: frozenset[tuple[int, int, int]], max_members: int
) -> list[DAG]:
    fixed = set(pdag.directed_edges)
    undirected = sorted(pdag.undirected_edges)
    members: list[DAG] = []
    for choice in product((0, 1), repeat=len(undirected)):
        edges = set(fixed)
        for bit, (a, b) in zip(choice, undirected):
            edges.add((a, b) if bit == 0 else (b, a))
        if not _is_acyclic(pdag.num_nodes, edges):
            continue
        dag = DAG.from_edges(pdag.num_nodes, edges)
        if dag_v_structures(dag) != reference:
            continue
        members.append(dag)
        if len(members) >= max_members:
            break
    return members


def _sample_one(pdag: CPDAG, rng: np.random.Generator) -> DAG | None:
    current = pdag
    guard = 0
    while current.undirected_edges:
        guard += 1
        if guard > 4 * pdag.num_nodes * pdag.num_nodes:
            return None
        edges = sorted(current.undirected_edges)
        a, b = edges[int(rng.integers(len(edges)))]
        options = [(a, b), (b, a)]
        if rng.random() < 0.5:
            options.reverse()
        advanced = False
        for src, dst in options:
            directed = set(current.directed_edges) | {(src, dst)}
            if not _is_acyclic(current.num_nodes, directed):
                continue
            undirected = set(current.undirected_edges) - {(a, b)}
            try:
                candidate = CPDAG(
                    num_nodes=current.num_nodes,
                    directed_edges=frozenset(directed),
                    undirected_edges=frozenset(undirected),
                )
            except ValueError:
                continue
            current = meek_closure(candidate, include_rule4=False)
            advanced = True
            break
        if not advanced:
            return None
    try:
        return DAG.from_edges(current.num_nodes, current.directed_edges)
    except ValueError:
        return None


def _enumerate_sampled(
    pdag: CPDAG,
    reference: frozenset[tuple[int, int, int]],
    max_members: int,
    rng: np.random.Generator,
    attempts_per_member: int = 12,
) -> list[DAG]:
    seen: dict[frozenset[tuple[int, int]], DAG] = {}
    for _ in range(max_members * attempts_per_member):
        dag = _sample_one(pdag, rng)
        if dag is None:
            continue
        if dag_v_structures(dag) != reference:
            continue
        seen.setdefault(dag.edges, dag)
        if len(seen) >= max_members:
            break
    return list(seen.values())


def enumerate_mec(
    pdag: CPDAG,
    *,
    max_members: int = 512,
    rng: np.random.Generator | None = None,
) -> tuple[tuple[DAG, ...], bool]:
    """Enumerate DAG members of the equivalence class described by `pdag`.

    Returns `(members, exhaustive)` where `exhaustive` says whether the enumeration
    is provably complete (no sampling and no truncation).
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    reference = v_structures(pdag)
    n_undirected = pdag.num_undirected_edges

    if n_undirected == 0:
        dag = DAG.from_edges(pdag.num_nodes, pdag.directed_edges)
        return (dag,), True

    if n_undirected <= EXHAUSTIVE_LIMIT_BITS:
        members = _enumerate_exhaustive(pdag, reference, max_members)
        exhaustive = len(members) < max_members
        if members:
            return tuple(members), exhaustive

    members = _enumerate_sampled(pdag, reference, max_members, rng)
    if not members:
        # last resort: any acyclic completion, ignoring the v-structure constraint
        fallback = _sample_one(pdag, rng)
        if fallback is not None:
            return (fallback,), False
        return tuple(), False
    return tuple(members), False


def mec_entropy(weights: np.ndarray) -> float:
    """Shannon entropy (nats) of a normalised weight vector."""
    w = np.asarray(weights, dtype=float)
    w = w[w > 0.0]
    if w.size == 0:
        return 0.0
    return float(-np.sum(w * np.log(w)))


def edge_marginals(members: "tuple[DAG, ...]", weights: np.ndarray, num_nodes: int) -> np.ndarray:
    """`M[i, j]` = posterior probability that edge i -> j is present."""
    marginals = np.zeros((num_nodes, num_nodes), dtype=float)
    for dag, weight in zip(members, np.asarray(weights, dtype=float)):
        for src, dst in dag.edges:
            marginals[src, dst] += weight
    return marginals
