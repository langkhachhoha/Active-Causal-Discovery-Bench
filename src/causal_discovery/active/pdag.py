"""Public partially-directed-graph utilities for the active-experiment studies.

Everything here is a thin, *public* wrapper over the benchmark's own theory layer
(`causal_discovery.equivalence.theory`) so that the studies and the benchmark share
exactly the same Meek-rule semantics.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from causal_discovery.core import DAG
from causal_discovery.equivalence import CPDAG
from causal_discovery.equivalence.cpdag import canonical_undirected_edge
from causal_discovery.equivalence.theory import _apply_meek_closure, _MutablePDAG

MEAN_SHIFT_Z_975 = 1.959963984540054
DEFAULT_INTERVENTION_OFFSET = 3.0


# --------------------------------------------------------------------------- #
# basic accessors
# --------------------------------------------------------------------------- #
def undirected_degree(pdag: CPDAG) -> dict[int, int]:
    """Number of undirected edges incident to each node (nodes with 0 omitted)."""
    degree: dict[int, int] = {}
    for a, b in pdag.undirected_edges:
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
    return degree


def open_targets(pdag: CPDAG) -> tuple[int, ...]:
    """Nodes incident to at least one undirected edge — the only useful targets."""
    return tuple(sorted(undirected_degree(pdag).keys()))


def skeleton_pairs(pdag: CPDAG) -> frozenset[tuple[int, int]]:
    directed = {canonical_undirected_edge(a, b) for a, b in pdag.directed_edges}
    return frozenset(directed | set(pdag.undirected_edges))


def v_structures(pdag: CPDAG) -> frozenset[tuple[int, int, int]]:
    """Unshielded colliders `(a, c, b)` with a < b, both a->c and b->c directed."""
    skeleton = skeleton_pairs(pdag)
    parents: dict[int, set[int]] = {}
    for src, dst in pdag.directed_edges:
        parents.setdefault(dst, set()).add(src)
    out: set[tuple[int, int, int]] = set()
    for child, ps in parents.items():
        for a, b in combinations(sorted(ps), 2):
            if canonical_undirected_edge(a, b) not in skeleton:
                out.add((a, child, b))
    return frozenset(out)


def dag_v_structures(dag: DAG) -> frozenset[tuple[int, int, int]]:
    skeleton = {canonical_undirected_edge(a, b) for a, b in dag.edges}
    out: set[tuple[int, int, int]] = set()
    for child in dag.nodes:
        for a, b in combinations(dag.parents(child), 2):
            if canonical_undirected_edge(a, b) not in skeleton:
                out.add((a, child, b))
    return frozenset(out)


def cpdag_of_dag_edges(num_nodes: int, edges: frozenset[tuple[int, int]]) -> CPDAG:
    from causal_discovery.equivalence import dag_to_cpdag

    return dag_to_cpdag(DAG.from_edges(num_nodes, edges))


def dag_from_pdag(pdag: CPDAG) -> DAG | None:
    """Return the DAG if the PDAG is fully oriented, else None."""
    if pdag.undirected_edges:
        return None
    try:
        return DAG.from_edges(pdag.num_nodes, pdag.directed_edges)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# closure / orientation
# --------------------------------------------------------------------------- #
def meek_closure(pdag: CPDAG, *, include_rule4: bool = True) -> CPDAG:
    """Apply Meek rules until no change. `include_rule4=True` is the post-intervention regime."""
    mutable = _MutablePDAG.from_cpdag(pdag)
    _apply_meek_closure(mutable, include_rule4=include_rule4)
    try:
        return mutable.to_cpdag()
    except ValueError:
        return pdag


def _would_create_cycle(directed: set[tuple[int, int]], src: int, dst: int) -> bool:
    adjacency: dict[int, list[int]] = {}
    for a, b in directed:
        adjacency.setdefault(a, []).append(b)
    stack = [dst]
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if node == src:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(adjacency.get(node, []))
    return False


def _orient_incident(
    pdag: CPDAG,
    target: int,
    decide: "callable[[int, int], tuple[int, int] | None]",
) -> tuple[CPDAG, int]:
    """Orient every undirected edge incident to `target` via `decide`, then close.

    `decide(target, other)` returns the oriented pair or None to leave it undirected.
    Returns the new PDAG and the number of *undirected edges removed* (incl. closure).
    """
    before = pdag.num_undirected_edges
    directed = set(pdag.directed_edges)
    undirected = set(pdag.undirected_edges)

    incident = sorted(edge for edge in undirected if target in edge)
    for edge in incident:
        other = edge[1] if edge[0] == target else edge[0]
        oriented = decide(target, other)
        if oriented is None:
            continue
        if _would_create_cycle(directed, *oriented):
            oriented = (oriented[1], oriented[0])
            if _would_create_cycle(directed, *oriented):
                continue
        directed.add(oriented)
        undirected.discard(edge)

    try:
        candidate = CPDAG(
            num_nodes=pdag.num_nodes,
            directed_edges=frozenset(directed),
            undirected_edges=frozenset(undirected),
        )
    except ValueError:
        return pdag, 0
    closed = meek_closure(candidate, include_rule4=True)
    return closed, before - closed.num_undirected_edges


def orient_from_truth(pdag: CPDAG, target: int, true_dag: DAG) -> tuple[CPDAG, int]:
    """Benchmark-owned oracle orientation — used only for regret bookkeeping."""

    def decide(a: int, b: int) -> tuple[int, int] | None:
        if true_dag.has_edge(a, b):
            return (a, b)
        if true_dag.has_edge(b, a):
            return (b, a)
        return None

    return _orient_incident(pdag, target, decide)


def mean_shift_threshold(obs_var: float, n_obs: int, int_var: float, n_int: int) -> float:
    return MEAN_SHIFT_Z_975 * float(np.sqrt(obs_var / max(n_obs, 1) + int_var / max(n_int, 1)))


def orient_from_intervention(
    pdag: CPDAG,
    target: int,
    obs_data: np.ndarray,
    int_data: np.ndarray,
) -> tuple[CPDAG, int]:
    """Orient edges incident to `target` with the calibrated mean-shift test, then close.

    This is the benchmark's `z_calibrated_mean_shift` rule **plus** Meek closure.
    """
    mu_obs = obs_data.mean(axis=0)
    var_obs = obs_data.var(axis=0, ddof=1)
    n_obs = int(obs_data.shape[0])
    mu_int = int_data.mean(axis=0)
    var_int = int_data.var(axis=0, ddof=1)
    n_int = int(int_data.shape[0])

    def decide(a: int, b: int) -> tuple[int, int] | None:
        threshold = mean_shift_threshold(float(var_obs[b]), n_obs, float(var_int[b]), n_int)
        shifted = abs(float(mu_int[b]) - float(mu_obs[b])) > threshold
        return (a, b) if shifted else (b, a)

    return _orient_incident(pdag, target, decide)


def force_orient_remaining(pdag: CPDAG, rng: np.random.Generator) -> CPDAG:
    """Break any leftover undirected edges arbitrarily (acyclic-safe). For MAP-style submits."""
    directed = set(pdag.directed_edges)
    undirected = sorted(pdag.undirected_edges)
    rng.shuffle(undirected)
    for a, b in undirected:
        candidate = (a, b) if rng.random() < 0.5 else (b, a)
        if _would_create_cycle(directed, *candidate):
            candidate = (candidate[1], candidate[0])
            if _would_create_cycle(directed, *candidate):
                continue
        directed.add(candidate)
    return CPDAG(
        num_nodes=pdag.num_nodes,
        directed_edges=frozenset(directed),
        undirected_edges=frozenset(),
    )


def intervention_value(obs_data: np.ndarray, target: int, offset: float = DEFAULT_INTERVENTION_OFFSET) -> float:
    """Standard intervention value: push `target` `offset` observational sds above its mean."""
    mu = float(obs_data[:, target].mean())
    sd = float(obs_data[:, target].std(ddof=1))
    if not np.isfinite(sd) or sd <= 0.0:
        sd = 1.0
    return mu + offset * sd
