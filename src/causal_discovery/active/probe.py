"""PROBE — Proposal-based Optimal Bayesian Experimentation.

The agent is split into three parts with clearly separated responsibilities:

1. **Propose**  the LLM reads sufficient statistics of the observational panel and
   emits K candidate DAGs. This is the only place the LLM is used.
2. **Score**    each candidate is scored by exact linear-Gaussian BIC on the observed
   data, giving a posterior `w` over the hypothesis set. Bad proposals are demoted
   mechanically rather than trusted.
3. **Choose / Update**  the next intervention maximises the exact expected
   information gain about the hypothesis index, and its outcome updates `w` by Bayes
   using the closed-form interventional likelihood of a mutilated linear-Gaussian SCM.

The LLM never reasons numerically about the interventional data; it only supplies the
hypothesis space. Study 1 motivates exactly this division of labour.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from causal_discovery import BenchmarkEnv, GraphSubmission, score_submission
from causal_discovery.core import DAG
from causal_discovery.equivalence import CPDAG
from causal_discovery.equivalence.cpdag import canonical_undirected_edge
from causal_discovery.active.episode import (
    EpisodeResult,
    run_pc,
    score_fields,
    skeleton_ceiling_f1,
)
from causal_discovery.active.gaussian import (
    GaussianParams,
    LocalBicCache,
    bic_score,
    fit_linear_gaussian,
    implied_mean_cov,
    interventional_loglik,
    normalise_log_weights,
    sample_interventional,
    _gaussian_loglik,
)
from causal_discovery.active.inference import sanitize_graph
from causal_discovery.active.llm_client import OpenRouterClient
from causal_discovery.active.mec import edge_marginals, enumerate_mec, mec_entropy
from causal_discovery.active.pdag import intervention_value

HYPOTHESIS_SOURCES = (
    "hybrid", "llm_repair", "llm_graphs", "hybrid_graphs", "pc_skeleton", "pc_mec", "random",
    "oracle_repair", "noise_repair",
)
# sources whose skeleton edits come from somewhere other than an LLM; they share the
# entire downstream pipeline with `llm_repair`, so they isolate the *content* of the edits.
EDIT_SOURCES = ("llm_repair", "hybrid", "oracle_repair", "noise_repair")
SELECT_RULES = ("eig", "random", "maxdeg")


# --------------------------------------------------------------------------- #
# statistics handed to the proposer
# --------------------------------------------------------------------------- #
def observational_summary(obs: np.ndarray) -> dict[str, Any]:
    """Correlation and full-order partial correlation — sufficient for a Gaussian model."""
    corr = np.corrcoef(obs, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    d = corr.shape[0]
    try:
        precision = np.linalg.inv(corr + 1e-8 * np.eye(d))
        denominator = np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
        partial = -precision / np.where(denominator == 0.0, 1.0, denominator)
        np.fill_diagonal(partial, 1.0)
    except np.linalg.LinAlgError:
        partial = np.eye(d)
    return {
        "n_rows": int(obs.shape[0]),
        "means": [round(float(v), 3) for v in obs.mean(axis=0)],
        "stds": [round(float(v), 3) for v in obs.std(axis=0, ddof=1)],
        "correlation": np.round(corr, 3).tolist(),
        "partial_correlation_given_all_others": np.round(partial, 3).tolist(),
    }


# --------------------------------------------------------------------------- #
# hypothesis proposal
# --------------------------------------------------------------------------- #
PROPOSE_TOOL = {
    "type": "function",
    "function": {
        "name": "propose_graphs",
        "description": "Propose several distinct candidate causal DAGs.",
        "parameters": {
            "type": "object",
            "properties": {
                "graphs": {
                    "type": "array",
                    "description": (
                        "Each element is one candidate DAG: a list of [source, target] "
                        "0-based index pairs."
                    ),
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                    },
                },
                "rationale": {"type": "string", "description": "How the candidates differ."},
            },
            "required": ["graphs", "rationale"],
            "additionalProperties": False,
        },
    },
}

PROPOSE_SYSTEM_PROMPT = (
    "You are the hypothesis-generation module of a causal discovery system.\n"
    "The world is an unknown linear-Gaussian DAG with no hidden confounders and full observability.\n"
    "You are shown sufficient statistics of an observational sample: the correlation matrix and the "
    "partial correlation of each pair given all other variables.\n"
    "Read them like this: a partial correlation near zero given all others means the two variables are "
    "very likely NOT directly connected; a large partial correlation means they very likely ARE directly "
    "connected. A large marginal correlation with a near-zero partial correlation indicates an indirect "
    "path, not an edge. With a small sample these statistics are noisy, so treat borderline values as "
    "genuinely uncertain.\n"
    "Your job is NOT to pick one answer. Emit a diverse SET of plausible acyclic candidates that "
    "together cover the uncertainty: vary the orientation of edges the data cannot orient, and vary "
    "the borderline adjacencies. A downstream module will score each candidate by likelihood and run "
    "experiments to tell them apart, so a set that contains the truth is far more valuable than a set "
    "of near-identical safe guesses.\n"
    "Every candidate must be acyclic and use 0-based integer variable indices.\n"
    "Call propose_graphs exactly once."
)


def _dag_from_pairs(num_nodes: int, pairs: Any) -> DAG | None:
    submission, _ = sanitize_graph(num_nodes, pairs if isinstance(pairs, list) else [], [])
    if not submission.directed_edges:
        return None
    try:
        return DAG.from_edges(num_nodes, submission.directed_edges)
    except ValueError:
        return None


def propose_hypotheses_llm(
    client: OpenRouterClient,
    obs: np.ndarray,
    num_nodes: int,
    num_candidates: int,
    *,
    work_key: str,
    skeleton_hint: CPDAG | None = None,
    rounds: int = 1,
) -> tuple[list[DAG], dict[str, Any]]:
    """Ask the LLM for candidate DAGs; return the valid, de-duplicated set."""
    payload: dict[str, Any] = {
        "variables": [f"X{i}" for i in range(num_nodes)],
        "statistics": observational_summary(obs),
        "num_candidates_requested": num_candidates,
    }
    if skeleton_hint is not None:
        payload["skeleton_hint"] = {
            "source": "PC algorithm on the same observational sample",
            "directed_edges": [list(e) for e in sorted(skeleton_hint.directed_edges)],
            "undirected_edges": [list(e) for e in sorted(skeleton_hint.undirected_edges)],
        }

    def validate(data: dict[str, Any]) -> None:
        graphs = data.get("graphs")
        if not isinstance(graphs, list) or not graphs:
            raise ValueError("graphs must be a non-empty list of edge lists")
        for graph in graphs:
            if not isinstance(graph, list):
                raise ValueError("each graph must be a list of [source, target] pairs")

    unique: dict[frozenset[tuple[int, int]], DAG] = {}
    proposed = 0
    repairs = 0
    for round_index in range(max(rounds, 1)):
        prompt = (
            "Observational evidence JSON:\n"
            + json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
            + f"\n\nPropose {num_candidates} distinct acyclic candidate graphs."
        )
        if round_index > 0 and unique:
            prompt += (
                "\nYou already proposed these candidates; propose DIFFERENT ones that disagree "
                "with them on edge directions or on borderline adjacencies:\n"
                + json.dumps([sorted(map(list, dag.edges)) for dag in unique.values()][:12])
            )
        response = client.call_tool(
            system_prompt=PROPOSE_SYSTEM_PROMPT,
            user_prompt=prompt,
            tool=PROPOSE_TOOL,
            validate=validate,
            tag="propose",
            context={"work_key": work_key, "round": round_index},
        )
        repairs += response.repairs
        graphs = response.payload.get("graphs") or []
        proposed += len(graphs)
        for graph in graphs:
            dag = _dag_from_pairs(num_nodes, graph)
            if dag is not None:
                unique.setdefault(dag.edges, dag)

    stats = {
        "proposed_raw": proposed,
        "proposed_valid_unique": len(unique),
        "propose_repairs": repairs,
        "propose_rounds": max(rounds, 1),
    }
    return list(unique.values()), stats


# --------------------------------------------------------------------------- #
# skeleton repair: the LLM edits PC's adjacency set instead of inventing whole graphs
# --------------------------------------------------------------------------- #
REPAIR_TOOL = {
    "type": "function",
    "function": {
        "name": "propose_skeleton_edits",
        "description": "Flag adjacencies the PC algorithm probably got wrong.",
        "parameters": {
            "type": "object",
            "properties": {
                "remove": {
                    "type": "array",
                    "description": "Pairs [i, j] currently adjacent that are probably NOT directly connected.",
                    "items": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                },
                "add": {
                    "type": "array",
                    "description": "Pairs [i, j] currently non-adjacent that probably ARE directly connected.",
                    "items": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                },
                "rationale": {"type": "string"},
            },
            "required": ["remove", "add", "rationale"],
            "additionalProperties": False,
        },
    },
}

REPAIR_SYSTEM_PROMPT = (
    "You audit the adjacency structure of a causal graph estimated from a small sample.\n"
    "The world is an unknown linear-Gaussian DAG with no hidden confounders and full observability.\n"
    "You are given the correlation matrix, the partial correlation of every pair given all other "
    "variables, and the adjacency set the PC algorithm returned from the same sample.\n"
    "Read the statistics like this: for a linear-Gaussian DAG, two variables are directly connected if "
    "and only if their partial correlation given all others is non-zero. So a pair with a sizeable "
    "partial correlation that PC left non-adjacent is a likely FALSE NEGATIVE, and a pair PC made "
    "adjacent whose partial correlation is near zero is a likely FALSE POSITIVE. With a small sample "
    "PC's independence tests are underpowered, so borderline values matter.\n"
    "List only the pairs you would genuinely bet on — an empty list is a valid and often correct answer. "
    "Do not restate the whole graph; report edits only. At most 4 removals and 4 additions.\n"
    "Call propose_skeleton_edits exactly once."
)

SEMANTIC_SUFFIX = (
    "\nThe variables are named, and the names are meaningful: they come from {domain}. "
    "Use what you know about this domain alongside the statistics — a pair that is "
    "implausible on domain grounds is a good removal candidate even when the statistics "
    "are borderline, and vice versa."
)


def _pairs_from(raw: Any, num_nodes: int, limit: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        try:
            a, b = int(item[0]), int(item[1])
        except (TypeError, ValueError):
            continue
        if a == b or not (0 <= a < num_nodes and 0 <= b < num_nodes):
            continue
        edge = canonical_undirected_edge(a, b)
        if edge not in out:
            out.append(edge)
        if len(out) >= limit:
            break
    return out


def propose_skeleton_edits_llm(
    client: OpenRouterClient,
    obs: np.ndarray,
    pc_pdag: CPDAG,
    *,
    work_key: str,
    max_edits: int = 4,
    var_names: tuple[str, ...] | None = None,
    domain: str = "",
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], dict[str, Any]]:
    """Ask the LLM which PC adjacencies to drop and which missing pairs to add."""
    num_nodes = int(obs.shape[1])
    skeleton = sorted(
        {canonical_undirected_edge(a, b) for a, b in pc_pdag.directed_edges} | set(pc_pdag.undirected_edges)
    )
    labels = list(var_names) if var_names else [f"X{i}" for i in range(num_nodes)]
    payload = {
        "variables": [{"index": i, "name": labels[i]} for i in range(num_nodes)] if var_names
                     else [f"X{i}" for i in range(num_nodes)],
        "statistics": observational_summary(obs),
        "pc_adjacencies": [list(e) for e in skeleton],
        "non_adjacent_pairs": [
            [i, j] for i in range(num_nodes) for j in range(i + 1, num_nodes) if (i, j) not in set(skeleton)
        ],
    }

    def validate(data: dict[str, Any]) -> None:
        for field in ("remove", "add"):
            if not isinstance(data.get(field), list):
                raise ValueError(f"{field} must be a list of [i, j] pairs (use [] for none)")

    system_prompt = REPAIR_SYSTEM_PROMPT
    if var_names and domain:
        system_prompt += SEMANTIC_SUFFIX.format(domain=domain)
    response = client.call_tool(
        system_prompt=system_prompt,
        user_prompt=(
            "Evidence JSON:\n"
            + json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
            + "\n\nWhich adjacencies did PC probably get wrong?"
        ),
        tool=REPAIR_TOOL,
        validate=validate,
        tag="repair",
        context={"work_key": work_key},
    )
    skeleton_set = set(skeleton)
    remove = [e for e in _pairs_from(response.payload.get("remove"), num_nodes, max_edits) if e in skeleton_set]
    add = [e for e in _pairs_from(response.payload.get("add"), num_nodes, max_edits) if e not in skeleton_set]
    stats = {
        "repair_remove": len(remove),
        "repair_add": len(add),
        "propose_repairs": response.repairs,
    }
    return remove, add, stats


class ProposalCache:
    """Share one skeleton-repair call across every ablation arm of the same instance.

    Ablation arms differ only in the component under test; letting each one draw its own
    proposal would confound the comparison with provider non-determinism. The first arm
    to ask pays for the call, and the recorded token/cost usage is replayed onto every
    arm that reuses it, so per-arm cost figures stay meaningful.
    """

    def __init__(self) -> None:
        import threading

        self._lock = threading.Lock()
        self._value: tuple[list, list, dict[str, Any]] | None = None
        self._usage: dict[str, int | float] | None = None

    def get_or_call(self, factory, *, client) -> tuple[list, list, dict[str, Any]]:
        with self._lock:
            if self._value is None:
                before = client.usage.as_row()
                self._value = factory()
                after = client.usage.as_row()
                self._usage = {
                    key: after[key] - before[key]
                    for key in ("llm_calls", "prompt_tokens", "completion_tokens", "total_tokens", "cost_usd")
                }
                remove, add, stats = self._value
                return list(remove), list(add), {**stats, "propose_cached": 0}

            remove, add, stats = self._value
            if self._usage is not None:
                client.usage.calls += int(self._usage["llm_calls"])
                client.usage.prompt_tokens += int(self._usage["prompt_tokens"])
                client.usage.completion_tokens += int(self._usage["completion_tokens"])
                client.usage.total_tokens += int(self._usage["total_tokens"])
                client.usage.cost_usd += float(self._usage["cost_usd"])
            return list(remove), list(add), {**stats, "propose_cached": 1}


def skeleton_variants(
    base: set[tuple[int, int]],
    remove: list[tuple[int, int]],
    add: list[tuple[int, int]],
    max_variants: int,
) -> list[frozenset[tuple[int, int]]]:
    """Base skeleton, each single edit, and the all-edits variant — de-duplicated."""
    variants: list[frozenset[tuple[int, int]]] = [frozenset(base)]
    seen = {frozenset(base)}
    for edge in remove:
        candidate = frozenset(base - {edge})
        if candidate not in seen:
            seen.add(candidate)
            variants.append(candidate)
    for edge in add:
        candidate = frozenset(base | {edge})
        if candidate not in seen:
            seen.add(candidate)
            variants.append(candidate)
    if remove or add:
        combined = frozenset((base - set(remove)) | set(add))
        if combined not in seen:
            seen.add(combined)
            variants.append(combined)
    return variants[:max_variants]


def dags_from_skeleton(
    skeleton: frozenset[tuple[int, int]],
    num_nodes: int,
    max_dags: int,
    rng: np.random.Generator,
) -> list[DAG]:
    """Acyclic orientations of a fixed skeleton (exhaustive when cheap, else sampled)."""
    from causal_discovery.active.mec import _is_acyclic

    edges = sorted(skeleton)
    if not edges:
        return []
    out: list[DAG] = []
    if len(edges) <= 14 and 2 ** len(edges) <= max_dags * 8:
        from itertools import product

        for choice in product((0, 1), repeat=len(edges)):
            oriented = {(a, b) if bit == 0 else (b, a) for bit, (a, b) in zip(choice, edges)}
            if _is_acyclic(num_nodes, oriented):
                out.append(DAG.from_edges(num_nodes, oriented))
            if len(out) >= max_dags:
                break
        return out
    seen: set[frozenset[tuple[int, int]]] = set()
    for _ in range(max_dags * 6):
        bits = rng.integers(0, 2, size=len(edges))
        oriented = {(a, b) if bit == 0 else (b, a) for bit, (a, b) in zip(bits, edges)}
        key = frozenset(oriented)
        if key in seen or not _is_acyclic(num_nodes, oriented):
            continue
        seen.add(key)
        out.append(DAG.from_edges(num_nodes, oriented))
        if len(out) >= max_dags:
            break
    return out


def top_by_bic(candidates: list[DAG], obs: np.ndarray, keep: int) -> list[DAG]:
    """Keep the `keep` best-scoring candidates under decomposable BIC."""
    if len(candidates) <= keep:
        return candidates
    cache = LocalBicCache(obs)
    scored = sorted(candidates, key=cache.score)
    return scored[:keep]


def hypotheses_from_skeleton_search(
    obs: np.ndarray,
    pc_pdag: CPDAG,
    *,
    remove: list[tuple[int, int]],
    add: list[tuple[int, int]],
    max_variants: int,
    max_dags_per_skeleton: int,
    keep: int,
    rng: np.random.Generator,
    reserve_frac: float = 0.5,
) -> list[DAG]:
    """Enumerate acyclic orientations of PC's skeleton and its proposed variants.

    A `reserve_frac` share of the budget is reserved for orientations of PC's *unedited*
    skeleton, so a proposal can only ever add hypotheses to the space, never crowd out the
    default ones. This guard is what makes a wrong proposal cheap; `reserve_frac=0`
    removes it and lets an over-eager proposer degrade the space instead of enriching it.
    """
    num_nodes = int(obs.shape[1])
    base = {canonical_undirected_edge(a, b) for a, b in pc_pdag.directed_edges} | set(pc_pdag.undirected_edges)

    base_dags = dags_from_skeleton(frozenset(base), num_nodes, max_dags_per_skeleton, rng)
    n_reserved = 0 if reserve_frac <= 0.0 else max(int(round(keep * reserve_frac)), 1)
    reserved = top_by_bic(base_dags, obs, n_reserved) if n_reserved else []
    selected: dict[frozenset[tuple[int, int]], DAG] = {dag.edges: dag for dag in reserved}

    edited: dict[frozenset[tuple[int, int]], DAG] = {}
    for variant in skeleton_variants(base, remove, add, max_variants)[1:]:
        for dag in dags_from_skeleton(variant, num_nodes, max_dags_per_skeleton, rng):
            if dag.edges not in selected:
                edited.setdefault(dag.edges, dag)

    for dag in top_by_bic(list(edited.values()), obs, max(keep - len(selected), 0)):
        selected.setdefault(dag.edges, dag)
    for dag in top_by_bic(base_dags, obs, keep):  # backfill if edits produced few candidates
        if len(selected) >= keep:
            break
        selected.setdefault(dag.edges, dag)
    return list(selected.values())


def oracle_skeleton_edits(
    pc_pdag: CPDAG, true_dag: DAG, max_edits: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """The edits a perfect proposer would make: exactly PC's skeleton errors.

    This is the ceiling of the proposal channel — it says how much accuracy is still on
    the table for a better proposer, holding the rest of the pipeline fixed.
    """
    base = {canonical_undirected_edge(a, b) for a, b in pc_pdag.directed_edges} | set(pc_pdag.undirected_edges)
    truth = {canonical_undirected_edge(a, b) for a, b in true_dag.edges}
    return sorted(base - truth)[:max_edits], sorted(truth - base)[:max_edits]


def noise_skeleton_edits(
    pc_pdag: CPDAG,
    num_nodes: int,
    n_remove: int,
    n_add: int,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Edits drawn uniformly at random, at a rate matched to the LLM's.

    The control for "does the *content* of the proposal matter, or only the fact that the
    space got wider?". Every downstream stage is identical to the LLM arm.
    """
    base = sorted(
        {canonical_undirected_edge(a, b) for a, b in pc_pdag.directed_edges} | set(pc_pdag.undirected_edges)
    )
    base_set = set(base)
    non_adjacent = [
        (i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes) if (i, j) not in base_set
    ]
    remove = [base[i] for i in rng.choice(len(base), size=min(n_remove, len(base)), replace=False)] if base else []
    add = (
        [non_adjacent[i] for i in rng.choice(len(non_adjacent), size=min(n_add, len(non_adjacent)), replace=False)]
        if non_adjacent
        else []
    )
    return sorted(remove), sorted(add)


def hypotheses_from_mec(cpdag: CPDAG, max_members: int, rng: np.random.Generator) -> list[DAG]:
    members, _ = enumerate_mec(cpdag, max_members=max_members, rng=rng)
    return list(members)


def random_hypotheses(num_nodes: int, num_edges: int, count: int, rng: np.random.Generator) -> list[DAG]:
    from causal_discovery.graph_gen import sample_random_dag

    unique: dict[frozenset[tuple[int, int]], DAG] = {}
    for _ in range(count * 8):
        dag = sample_random_dag(num_nodes, max(1, num_edges), rng)
        unique.setdefault(dag.edges, dag)
        if len(unique) >= count:
            break
    return list(unique.values())


# --------------------------------------------------------------------------- #
# posterior over hypotheses
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Posterior:
    hypotheses: list[DAG]
    params: list[GaussianParams]
    log_weights: np.ndarray
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def weights(self) -> np.ndarray:
        return normalise_log_weights(self.log_weights)

    @property
    def entropy(self) -> float:
        return mec_entropy(self.weights)

    def map_index(self) -> int:
        return int(np.argmax(self.weights))

    def rank_of(self, dag: DAG) -> int:
        """1-based posterior rank of `dag`, or -1 if it is not in the set."""
        order = np.argsort(-self.weights)
        for rank, index in enumerate(order, start=1):
            if self.hypotheses[index].edges == dag.edges:
                return rank
        return -1


def build_posterior(hypotheses: list[DAG], obs: np.ndarray, *, use_bic: bool) -> Posterior:
    params = [fit_linear_gaussian(obs, dag) for dag in hypotheses]
    if use_bic:
        bics = np.array([bic_score(obs, dag, params=p) for dag, p in zip(hypotheses, params)], dtype=float)
        log_weights = -0.5 * bics
    else:
        log_weights = np.zeros(len(hypotheses), dtype=float)
    return Posterior(hypotheses=hypotheses, params=params, log_weights=log_weights)


# --------------------------------------------------------------------------- #
# experiment selection
# --------------------------------------------------------------------------- #
def expected_information_gain(
    posterior: Posterior,
    target: int,
    value: float,
    n_int: int,
    rng: np.random.Generator,
    num_outcomes: int = 12,
) -> float:
    """MC estimate of I(hypothesis ; interventional batch | do(target = value)) in nats."""
    weights = posterior.weights
    prior_entropy = mec_entropy(weights)
    if prior_entropy <= 1e-12:
        return 0.0

    d = posterior.params[0].weights.shape[0]
    keep = [j for j in range(d) if j != target]
    cache = []
    for params in posterior.params:
        mean, cov = implied_mean_cov(params, intervene=(target, value))
        cache.append((mean[keep], cov[np.ix_(keep, keep)]))

    log_prior = np.log(np.clip(weights, 1e-300, None))
    posterior_entropies = []
    indices = rng.choice(len(weights), size=num_outcomes, p=weights)
    for index in indices:
        batch = sample_interventional(posterior.params[int(index)], target, value, n_int, rng)
        loglik = np.array([_gaussian_loglik(batch, mean, cov) for mean, cov in cache], dtype=float)
        posterior_entropies.append(mec_entropy(normalise_log_weights(log_prior + loglik)))
    return float(prior_entropy - float(np.mean(posterior_entropies)))


def choose_target(
    posterior: Posterior,
    obs: np.ndarray,
    candidates: list[int],
    n_int: int,
    rng: np.random.Generator,
    rule: str,
    num_outcomes: int = 12,
) -> tuple[int, dict[str, Any]]:
    if not candidates:
        return 0, {"rule": rule, "note": "no candidates"}
    if rule == "random":
        target = int(candidates[int(rng.integers(len(candidates)))])
        return target, {"rule": "random"}
    if rule == "maxdeg":
        marginals = edge_marginals(tuple(posterior.hypotheses), posterior.weights, obs.shape[1])
        ambiguity = np.minimum(marginals, marginals.T)
        scores = {node: float(ambiguity[node].sum() + ambiguity[:, node].sum()) for node in candidates}
        target = max(sorted(scores), key=lambda node: scores[node])
        return target, {"rule": "max_ambiguous_degree", "score": round(scores[target], 4)}
    if rule != "eig":
        raise ValueError(f"unknown selection rule: {rule}")

    gains: dict[int, float] = {}
    for node in candidates:
        value = intervention_value(obs, node)
        gains[node] = expected_information_gain(posterior, node, value, n_int, rng, num_outcomes)
    target = max(sorted(gains), key=lambda node: gains[node])
    return target, {
        "rule": "max_expected_information_gain",
        "eig_nats": round(gains[target], 5),
        "eig_by_target": {str(k): round(v, 5) for k, v in gains.items()},
    }


# --------------------------------------------------------------------------- #
# submission
# --------------------------------------------------------------------------- #
def submission_from_posterior(
    posterior: Posterior,
    num_nodes: int,
    mode: str,
    *,
    presence_threshold: float = 0.5,
    direction_threshold: float = 0.75,
) -> GraphSubmission:
    if mode == "map":
        best = posterior.hypotheses[posterior.map_index()]
        return GraphSubmission(
            num_nodes=num_nodes,
            directed_edges=frozenset(best.edges),
            undirected_edges=frozenset(),
        )
    if mode != "marginal":
        raise ValueError(f"unknown submit mode: {mode}")

    marginals = edge_marginals(tuple(posterior.hypotheses), posterior.weights, num_nodes)
    directed: list[list[int]] = []
    undirected: list[list[int]] = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            forward = float(marginals[i, j])
            backward = float(marginals[j, i])
            presence = forward + backward
            if presence < presence_threshold:
                continue
            share = max(forward, backward) / presence if presence > 0 else 0.0
            if share >= direction_threshold:
                directed.append([i, j] if forward >= backward else [j, i])
            else:
                undirected.append([i, j])
    ordered = sorted(directed, key=lambda e: -max(marginals[e[0], e[1]], 0.0))
    submission, _ = sanitize_graph(num_nodes, ordered, undirected)
    return submission


# --------------------------------------------------------------------------- #
# full episode
# --------------------------------------------------------------------------- #
def run_probe_episode(
    *,
    instance,
    client: OpenRouterClient | None,
    runtime_seed: int,
    work_key: str,
    alpha: float = 0.05,
    hypothesis_source: str = "hybrid",
    num_candidates: int = 12,
    propose_rounds: int = 1,
    max_hypotheses: int = 64,
    select_rule: str = "eig",
    use_bic: bool = True,
    use_update: bool = True,
    submit_mode: str = "map",
    eig_outcomes: int = 12,
    skeleton_hint: bool = True,
    max_skeleton_edits: int = 4,
    max_skeleton_variants: int = 6,
    max_dags_per_skeleton: int = 1024,
    reserve_frac: float = 0.5,
    noise_edits_remove: int = 4,
    noise_edits_add: int = 4,
    var_names: tuple[str, ...] | None = None,
    domain: str = "",
    proposal_cache: "ProposalCache | None" = None,
) -> EpisodeResult:
    """One PROBE (or PROBE-ablation) episode."""
    rng = np.random.default_rng(runtime_seed)
    env = BenchmarkEnv(instance, np.random.default_rng(runtime_seed))
    obs = env.observe()
    num_nodes = int(obs.shape[1])
    n_int = int(instance.config.n_int)
    true_dag = instance.true_dag

    pc_pdag = run_pc(obs, alpha)

    hypotheses: list[DAG] = []
    propose_stats: dict[str, Any] = {}

    # (a) whole-graph proposals from the LLM
    if hypothesis_source in {"llm_graphs", "hybrid_graphs"}:
        if client is None:
            raise ValueError("hypothesis_source requires an OpenRouter client")
        llm_hypotheses, propose_stats = propose_hypotheses_llm(
            client,
            obs,
            num_nodes,
            num_candidates,
            work_key=work_key,
            skeleton_hint=pc_pdag if skeleton_hint else None,
            rounds=propose_rounds,
        )
        hypotheses.extend(llm_hypotheses)

    # (b) skeleton repair: the LLM edits PC's adjacency set, we enumerate orientations
    remove: list[tuple[int, int]] = []
    add: list[tuple[int, int]] = []
    if hypothesis_source in {"llm_repair", "hybrid"}:
        if client is None:
            raise ValueError("hypothesis_source requires an OpenRouter client")
        call = lambda: propose_skeleton_edits_llm(  # noqa: E731
            client,
            obs,
            pc_pdag,
            work_key=work_key,
            max_edits=max_skeleton_edits,
            var_names=var_names,
            domain=domain,
        )
        if proposal_cache is not None:
            remove, add, repair_stats = proposal_cache.get_or_call(call, client=client)
        else:
            remove, add, repair_stats = call()
        propose_stats.update(repair_stats)
    elif hypothesis_source == "oracle_repair":
        remove, add = oracle_skeleton_edits(pc_pdag, true_dag, max_skeleton_edits)
        propose_stats.update({"repair_remove": len(remove), "repair_add": len(add)})
    elif hypothesis_source == "noise_repair":
        remove, add = noise_skeleton_edits(
            pc_pdag, num_nodes, noise_edits_remove, noise_edits_add, rng
        )
        propose_stats.update({"repair_remove": len(remove), "repair_add": len(add)})

    if hypothesis_source in {"llm_repair", "hybrid", "pc_skeleton", "oracle_repair", "noise_repair"}:
        hypotheses.extend(
            hypotheses_from_skeleton_search(
                obs,
                pc_pdag,
                remove=remove,
                add=add,
                max_variants=max_skeleton_variants,
                max_dags_per_skeleton=max_dags_per_skeleton,
                keep=max_hypotheses,
                rng=rng,
                reserve_frac=reserve_frac,
            )
        )
    n_from_llm = len(hypotheses) if hypothesis_source in {"llm_repair", "llm_graphs", "hybrid", "hybrid_graphs"} else 0

    if hypothesis_source in {"pc_mec", "hybrid", "hybrid_graphs"}:
        hypotheses.extend(hypotheses_from_mec(pc_pdag, max_hypotheses, rng))
    if hypothesis_source == "random":
        hypotheses.extend(random_hypotheses(num_nodes, len(true_dag.edges), num_candidates, rng))

    unique: dict[frozenset[tuple[int, int]], DAG] = {}
    for dag in hypotheses:
        unique.setdefault(dag.edges, dag)
    if not unique:
        unique = {dag.edges: dag for dag in hypotheses_from_mec(pc_pdag, max_hypotheses, rng)}
    if not unique:
        fallback = random_hypotheses(num_nodes, max(1, len(true_dag.edges)), 8, rng)
        unique = {dag.edges: dag for dag in fallback}

    # Sources are merged and then pruned by BIC rank. Pruning is a computational
    # necessity and is applied to every arm identically; `use_bic` separately controls
    # whether BIC also sets the posterior *weights*.
    hypothesis_list = top_by_bic(list(unique.values()), obs, max_hypotheses)
    posterior = build_posterior(hypothesis_list, obs, use_bic=use_bic)

    truth_in_set = any(dag.edges == true_dag.edges for dag in hypothesis_list)
    best_possible_f1 = max(_directed_f1(dag, true_dag) for dag in hypothesis_list)
    entropy_initial = posterior.entropy
    rank_initial = posterior.rank_of(true_dag)

    steps: list[dict[str, Any]] = []
    while env.remaining_budget > 0:
        weights = posterior.weights
        if float(np.max(weights)) > 0.999 or len(hypothesis_list) == 1:
            break
        marginals = edge_marginals(tuple(hypothesis_list), weights, num_nodes)
        ambiguity = np.minimum(marginals, marginals.T)
        candidates = [
            node
            for node in range(num_nodes)
            if float(ambiguity[node].sum() + ambiguity[:, node].sum()) > 1e-6
        ]
        if not candidates:
            candidates = [
                node for node in range(num_nodes) if float(marginals[node].sum() + marginals[:, node].sum()) > 1e-6
            ]
        if not candidates:
            break

        step_index = len(steps) + 1
        target, meta = choose_target(
            posterior, obs, candidates, n_int, rng, select_rule, num_outcomes=eig_outcomes
        )
        value = intervention_value(obs, target)
        int_data = env.intervene(var=target, value=value)

        entropy_before = posterior.entropy
        if use_update:
            loglik = np.array(
                [interventional_loglik(int_data, params, target, value) for params in posterior.params],
                dtype=float,
            )
            posterior.log_weights = posterior.log_weights + loglik
        entropy_after = posterior.entropy

        steps.append(
            {
                "step": step_index,
                "target": target,
                "value": round(float(value), 4),
                "n_hypotheses": len(hypothesis_list),
                "entropy_before_nats": round(float(entropy_before), 5),
                "entropy_after_nats": round(float(entropy_after), 5),
                "entropy_drop_nats": round(float(entropy_before - entropy_after), 5),
                "map_weight_after": round(float(np.max(posterior.weights)), 5),
                "map_directed_f1_after": round(
                    _directed_f1(posterior.hypotheses[posterior.map_index()], true_dag), 5
                ),
                "truth_rank_after": posterior.rank_of(true_dag),
                "selector_meta": json.dumps(meta, default=str),
            }
        )

    submission = submission_from_posterior(posterior, num_nodes, submit_mode)
    output = env.submit_graph(submission)
    scores = score_submission(instance, output.submission)

    metrics: dict[str, Any] = {
        **score_fields(scores),
        "submit_directed": len(output.submission.directed_edges),
        "submit_undirected": len(output.submission.undirected_edges),
        "steps_taken": len(steps),
        "n_hypotheses": len(hypothesis_list),
        "n_hypotheses_from_llm": n_from_llm,
        "truth_in_hypotheses": int(truth_in_set),
        "best_f1_in_hypotheses": round(float(best_possible_f1), 6),
        "truth_rank_initial": rank_initial,
        "truth_rank_final": posterior.rank_of(true_dag),
        "entropy_initial_nats": round(float(entropy_initial), 5),
        "entropy_final_nats": round(float(posterior.entropy), 5),
        "map_weight_final": round(float(np.max(posterior.weights)), 5),
        "pc_skeleton_f1_ceiling": round(skeleton_ceiling_f1(pc_pdag, true_dag), 6),
        "pc_undirected_edges": pc_pdag.num_undirected_edges,
        "pc_directed_edges": pc_pdag.num_directed_edges,
        "reserve_frac": reserve_frac,
        "edits_correct_remove": sum(
            1 for e in remove if e not in {canonical_undirected_edge(a, b) for a, b in true_dag.edges}
        ),
        "edits_correct_add": sum(
            1 for e in add if e in {canonical_undirected_edge(a, b) for a, b in true_dag.edges}
        ),
    }
    metrics.update(propose_stats)
    return EpisodeResult(metrics=metrics, steps=steps)


def _directed_f1(dag: DAG, true_dag: DAG) -> float:
    tp = len(dag.edges & true_dag.edges)
    fp = len(dag.edges - true_dag.edges)
    fn = len(true_dag.edges - dag.edges)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------- #
# non-LLM reference arms for study 2
# --------------------------------------------------------------------------- #
def run_pc_greedy_episode(*, instance, runtime_seed: int, alpha: float, meek: bool) -> EpisodeResult:
    """Classical active baseline: PC + max-degree targeting + mean-shift orientation.

    `meek=False` reproduces the parent benchmark's `pc_greedy` exactly (no closure);
    `meek=True` adds Meek closure after every intervention.
    """
    from causal_discovery.active.pdag import (
        _orient_incident,
        mean_shift_threshold,
        open_targets,
        undirected_degree,
    )
    from causal_discovery.active.pdag import orient_from_intervention

    env = BenchmarkEnv(instance, np.random.default_rng(runtime_seed))
    obs = env.observe()
    pdag = run_pc(obs, alpha)
    steps: list[dict[str, Any]] = []

    while env.remaining_budget > 0 and pdag.undirected_edges:
        degree = undirected_degree(pdag)
        target = min(degree.keys(), key=lambda node: (-degree[node], node))
        value = intervention_value(obs, target)
        int_data = env.intervene(var=target, value=value)
        before = pdag
        if meek:
            pdag, resolved = orient_from_intervention(before, target, obs, int_data)
        else:
            pdag, resolved = _orient_no_closure(before, target, obs, int_data)
        # score the graph the arm *would* submit right now, so accuracy-per-experiment
        # curves are directly comparable with the PROBE arms
        interim = score_submission(
            instance,
            GraphSubmission(
                num_nodes=pdag.num_nodes,
                directed_edges=pdag.directed_edges,
                undirected_edges=pdag.undirected_edges,
            ),
        )
        steps.append(
            {
                "step": len(steps) + 1,
                "target": target,
                "value": round(float(value), 4),
                "edges_resolved": int(resolved),
                "undirected_after": pdag.num_undirected_edges,
                "map_directed_f1_after": round(float(interim.directed_f1), 5),
            }
        )
        if resolved == 0 and before.num_undirected_edges == pdag.num_undirected_edges:
            break

    submission = GraphSubmission(
        num_nodes=pdag.num_nodes,
        directed_edges=pdag.directed_edges,
        undirected_edges=pdag.undirected_edges,
    )
    output = env.submit_graph(submission)
    scores = score_submission(instance, output.submission)
    metrics = {
        **score_fields(scores),
        "submit_directed": len(output.submission.directed_edges),
        "submit_undirected": len(output.submission.undirected_edges),
        "steps_taken": len(steps),
        "pc_skeleton_f1_ceiling": round(skeleton_ceiling_f1(run_pc(obs, alpha), instance.true_dag), 6),
    }
    return EpisodeResult(metrics=metrics, steps=steps)


def _orient_no_closure(pdag: CPDAG, target: int, obs: np.ndarray, int_data: np.ndarray):
    """Mean-shift orientation without Meek closure (parent benchmark's rule)."""
    from causal_discovery.active.pdag import _would_create_cycle, mean_shift_threshold

    mu_obs = obs.mean(axis=0)
    var_obs = obs.var(axis=0, ddof=1)
    mu_int = int_data.mean(axis=0)
    var_int = int_data.var(axis=0, ddof=1)
    n_obs, n_int = int(obs.shape[0]), int(int_data.shape[0])

    directed = set(pdag.directed_edges)
    undirected = set(pdag.undirected_edges)
    before = len(undirected)
    for edge in sorted(e for e in undirected if target in e):
        other = edge[1] if edge[0] == target else edge[0]
        threshold = mean_shift_threshold(float(var_obs[other]), n_obs, float(var_int[other]), n_int)
        shifted = abs(float(mu_int[other]) - float(mu_obs[other])) > threshold
        candidate = (target, other) if shifted else (other, target)
        if _would_create_cycle(directed, *candidate):
            continue
        directed.add(candidate)
        undirected.discard(edge)
    try:
        out = CPDAG(
            num_nodes=pdag.num_nodes,
            directed_edges=frozenset(directed),
            undirected_edges=frozenset(undirected),
        )
    except ValueError:
        return pdag, 0
    return out, before - len(undirected)


def run_oracle_episode(*, instance, runtime_seed: int) -> EpisodeResult:
    env = BenchmarkEnv(instance, np.random.default_rng(runtime_seed))
    env.observe()
    submission = GraphSubmission.from_dag(instance.true_dag)
    output = env.submit_graph(submission)
    scores = score_submission(instance, output.submission)
    return EpisodeResult(
        metrics={
            **score_fields(scores),
            "submit_directed": len(output.submission.directed_edges),
            "submit_undirected": 0,
            "steps_taken": 0,
        }
    )
