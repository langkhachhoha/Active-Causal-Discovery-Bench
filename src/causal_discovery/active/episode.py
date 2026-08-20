"""Episode runners shared by both studies."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from causal_discovery import BenchmarkEnv, GraphSubmission, score_submission
from causal_discovery.baselines import parse_causallearn_endpoint_matrix
from causal_discovery.core import DAG
from causal_discovery.equivalence import CPDAG
from causal_discovery.equivalence.cpdag import canonical_undirected_edge
from causal_discovery.active.inference import Inferencer, sanitize_graph
from causal_discovery.active.llm_client import OpenRouterClient, coerce_float, coerce_int
from causal_discovery.active.mec import enumerate_mec
from causal_discovery.active.pdag import (
    MEAN_SHIFT_Z_975,
    dag_v_structures,
    intervention_value,
    orient_from_intervention,
    skeleton_pairs,
    v_structures,
)
from causal_discovery.active.selectors import (
    Selector,
    mec_target_entropy,
    true_gain_per_target,
)
from causal_discovery.active.state import BeliefState, Evidence


# --------------------------------------------------------------------------- #
# PC front-end shared by every arm
# --------------------------------------------------------------------------- #
def run_pc(obs: np.ndarray, alpha: float, *, return_sepsets: bool = False):
    """PC (Fisher-Z) on the observational panel, coerced into a valid CPDAG.

    With `return_sepsets`, also returns the separating set PC found for each pair it made
    non-adjacent — the actual evidence behind the decision, which a proposer needs in
    order to second-guess it.
    """
    from causallearn.search.ConstraintBased.PC import pc

    num_nodes = int(obs.shape[1])
    cg = pc(obs, alpha=alpha, indep_test="fisherz", show_progress=False, verbose=False)
    sepsets: dict[tuple[int, int], tuple[int, ...]] = {}
    if return_sepsets:
        raw = getattr(cg, "sepset", None)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                entry = None
                if raw is not None:
                    try:
                        entry = raw[i][j]
                    except (IndexError, TypeError):
                        entry = None
                if entry is None:
                    continue
                # causallearn stores either a set of nodes or a tuple of candidate sets
                flat: tuple[int, ...] = ()
                if isinstance(entry, (set, frozenset, list, tuple)):
                    items = list(entry)
                    if items and isinstance(items[0], (set, frozenset, list, tuple)):
                        items = list(items[0]) if items[0] is not None else []
                    try:
                        flat = tuple(sorted(int(x) for x in items))
                    except (TypeError, ValueError):
                        flat = ()
                sepsets[(i, j)] = flat
    try:
        directed, undirected = parse_causallearn_endpoint_matrix(cg.G.graph)
    except ValueError:
        directed, undirected = frozenset(), frozenset()
    try:
        result = CPDAG(num_nodes=num_nodes, directed_edges=directed, undirected_edges=undirected)
    except ValueError:
        # PC occasionally emits a directed part that is not acyclic; demote it.
        demoted = {canonical_undirected_edge(a, b) for a, b in directed} | set(undirected)
        result = CPDAG(num_nodes=num_nodes, directed_edges=frozenset(), undirected_edges=frozenset(demoted))
    return (result, sepsets) if return_sepsets else result


def truth_in_class(pdag: CPDAG, true_dag: DAG) -> bool:
    """Is the true DAG a member of the equivalence class described by `pdag`?"""
    if skeleton_pairs(pdag) != {canonical_undirected_edge(a, b) for a, b in true_dag.edges}:
        return False
    if v_structures(pdag) != dag_v_structures(true_dag):
        return False
    for src, dst in pdag.directed_edges:
        if not true_dag.has_edge(src, dst):
            return False
    return True


def skeleton_ceiling_f1(pdag: CPDAG, true_dag: DAG) -> float:
    """Best directed-F1 reachable from this skeleton (all orientations correct)."""
    skeleton = skeleton_pairs(pdag)
    truth_pairs = {canonical_undirected_edge(a, b) for a, b in true_dag.edges}
    tp = len(skeleton & truth_pairs)
    fp = len(skeleton - truth_pairs)
    fn = len(truth_pairs - skeleton)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------- #
# result container
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class EpisodeResult:
    metrics: dict[str, Any]
    steps: list[dict[str, Any]] = field(default_factory=list)


def score_fields(scores) -> dict[str, Any]:
    return {
        "skeleton_precision": scores.skeleton_precision,
        "skeleton_recall": scores.skeleton_recall,
        "skeleton_f1": scores.skeleton_f1,
        "compelled_f1": scores.compelled_f1,
        "directed_precision": scores.directed_precision,
        "directed_recall": scores.directed_recall,
        "directed_f1": scores.directed_f1,
        "dag_shd": scores.dag_shd,
        "efficiency": scores.efficiency,
        "interventions_used": scores.interventions_used,
        "optimal_interventions": scores.optimal_interventions,
    }


def _newly_oriented(before: CPDAG, after: CPDAG) -> set[tuple[int, int]]:
    return set(after.directed_edges) - set(before.directed_edges)


def _orientation_accuracy(new_edges: set[tuple[int, int]], true_dag: DAG) -> tuple[int, int]:
    correct = sum(1 for src, dst in new_edges if true_dag.has_edge(src, dst))
    return correct, len(new_edges) - correct


# --------------------------------------------------------------------------- #
# study 1: selector x inferencer decomposition
# --------------------------------------------------------------------------- #
def run_decomposition_episode(
    *,
    instance,
    selector: Selector,
    inferencer: Inferencer,
    alpha: float,
    runtime_seed: int,
    eig_max_members: int = 256,
    track_eig: bool = True,
    z_threshold: float = MEAN_SHIFT_Z_975,
    abstain_below: float | None = None,
) -> EpisodeResult:
    """One `selector x inferencer` arm on one instance.

    Every arm shares the same PC front-end and the same mechanical belief update, so
    all differences between arms are attributable to (a) which experiments were run and
    (b) how the final graph was read off the evidence.
    """
    rng = np.random.default_rng(runtime_seed)
    env = BenchmarkEnv(instance, np.random.default_rng(runtime_seed))
    obs = env.observe()
    true_dag = instance.true_dag

    initial_pdag = run_pc(obs, alpha)
    state = BeliefState.create(initial_pdag, obs, env.remaining_budget)

    _, mec_size0, mec_entropy0, exhaustive0 = mec_target_entropy(
        initial_pdag, max_members=eig_max_members, rng=rng
    ) if track_eig else ({}, 0, 0.0, False)

    steps: list[dict[str, Any]] = []
    total_regret = 0.0
    total_eig_regret = 0.0
    quality_scores: list[float] = []
    correct_orientations = 0
    wrong_orientations = 0
    wasted_steps = 0

    while state.remaining_budget > 0 and state.pdag.undirected_edges:
        state.step += 1
        before = state.pdag

        gains = true_gain_per_target(before, true_dag)
        best_gain = max(gains.values()) if gains else 0
        if track_eig:
            eig_map, mec_size, mec_entropy, exhaustive = mec_target_entropy(
                before, max_members=eig_max_members, rng=rng
            )
        else:
            eig_map, mec_size, mec_entropy, exhaustive = {}, 0, 0.0, False
        best_eig = max(eig_map.values()) if eig_map else 0.0

        selection = selector.choose(state)
        target = int(selection.target)
        if not (0 <= target < state.num_nodes):
            target = int(sorted(gains.keys())[0]) if gains else 0

        chosen_gain = int(gains.get(target, 0))
        chosen_eig = float(eig_map.get(target, 0.0))
        regret = float(best_gain - chosen_gain)
        eig_regret = float(best_eig - chosen_eig)
        total_regret += regret
        total_eig_regret += eig_regret
        if best_gain > 0:
            quality_scores.append(chosen_gain / best_gain)
        if chosen_gain == 0:
            wasted_steps += 1

        value = intervention_value(obs, target)
        int_data = env.intervene(var=target, value=value)
        after, resolved = orient_from_intervention(
            before, target, obs, int_data,
            z_threshold=z_threshold, abstain_below=abstain_below,
        )
        new_edges = _newly_oriented(before, after)
        correct, wrong = _orientation_accuracy(new_edges, true_dag)
        correct_orientations += correct
        wrong_orientations += wrong

        steps.append(
            {
                "step": state.step,
                "target": target,
                "value": round(float(value), 4),
                "undirected_before": before.num_undirected_edges,
                "undirected_after": after.num_undirected_edges,
                "mec_size_before": mec_size,
                "mec_entropy_before": round(mec_entropy, 5),
                "mec_exhaustive": exhaustive,
                "true_gain_chosen": chosen_gain,
                "true_gain_best": best_gain,
                "selection_regret": regret,
                "selection_quality": round(chosen_gain / best_gain, 5) if best_gain > 0 else "",
                "eig_chosen_nats": round(chosen_eig, 5),
                "eig_best_nats": round(best_eig, 5),
                "eig_regret_nats": round(eig_regret, 5),
                "edges_resolved": int(resolved),
                "orientations_correct": correct,
                "orientations_wrong": wrong,
                "selector_meta": json.dumps(selection.meta, default=str),
            }
        )

        state.evidence.append(
            Evidence(
                step=state.step,
                target=target,
                value=float(value),
                n_rows=int(int_data.shape[0]),
                means=tuple(float(v) for v in int_data.mean(axis=0)),
                stds=tuple(float(v) for v in int_data.std(axis=0, ddof=1)),
                data=int_data,
            )
        )
        state.pdag = after
        state.remaining_budget = env.remaining_budget

    inference = inferencer.infer(state, initial_pdag)
    output = env.submit_graph(inference.submission)
    scores = score_submission(instance, output.submission)

    metrics: dict[str, Any] = {
        **score_fields(scores),
        "submit_directed": len(output.submission.directed_edges),
        "submit_undirected": len(output.submission.undirected_edges),
        "steps_taken": len(steps),
        "pc_skeleton_f1_ceiling": round(skeleton_ceiling_f1(initial_pdag, true_dag), 6),
        "pc_truth_in_class": int(truth_in_class(initial_pdag, true_dag)),
        "pc_undirected_edges": initial_pdag.num_undirected_edges,
        "pc_directed_edges": initial_pdag.num_directed_edges,
        "mec_size_initial": mec_size0,
        "mec_entropy_initial": round(float(mec_entropy0), 5),
        "belief_undirected_final": state.pdag.num_undirected_edges,
        "selection_regret_total": round(total_regret, 5),
        "selection_regret_mean": round(total_regret / len(steps), 5) if steps else "",
        "eig_regret_total": round(total_eig_regret, 5),
        "selection_quality_mean": round(float(np.mean(quality_scores)), 5) if quality_scores else "",
        "wasted_steps": wasted_steps,
        "orientations_correct": correct_orientations,
        "orientations_wrong": wrong_orientations,
        "orientation_accuracy": (
            round(correct_orientations / (correct_orientations + wrong_orientations), 5)
            if (correct_orientations + wrong_orientations) > 0
            else ""
        ),
    }
    for key, value in inference.meta.items():
        metrics[f"infer_{key}"] = value
    for attribute in ("wasted_targets", "repeat_targets"):
        if hasattr(selector, attribute):
            metrics[f"select_{attribute}"] = getattr(selector, attribute)
    return EpisodeResult(metrics=metrics, steps=steps)


# --------------------------------------------------------------------------- #
# reference arm: end-to-end LLM agent (the parent benchmark's `llm_raw`)
# --------------------------------------------------------------------------- #
E2E_TOOL = {
    "type": "function",
    "function": {
        "name": "causal_discovery_action",
        "description": "Take exactly one action: run an intervention, or submit the final graph.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["intervene", "submit_graph"]},
                "var": {"type": ["integer", "null"], "description": "intervene: 0-based target index"},
                "value": {"type": ["number", "null"], "description": "intervene: value to force"},
                "directed_edges": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                },
                "undirected_edges": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                },
                "rationale": {"type": "string"},
            },
            "required": ["action", "var", "value", "directed_edges", "undirected_edges", "rationale"],
            "additionalProperties": False,
        },
    },
}

E2E_SYSTEM_PROMPT = (
    "You are performing causal discovery over an unknown linear-Gaussian DAG with full "
    "observability and no hidden confounders.\n"
    "Correlation is not causation: chains and common causes create dependence between variables that "
    "are not directly connected. Do not add an edge merely because two variables are correlated.\n"
    "An intervention do(X_a = v) severs every incoming edge of X_a and forces it to v. Variables "
    "downstream of X_a shift their mean; variables that are not downstream do not. Choose values far "
    "from the observational mean so the shift is detectable.\n"
    "Observational data alone can only identify the graph up to its Markov equivalence class, so use "
    "your intervention budget to resolve directions.\n"
    "The target is a directed acyclic graph. Use undirected_edges only for genuinely unresolved "
    "adjacencies. Take exactly one action per turn."
)


def run_e2e_llm_episode(
    *,
    instance,
    client: OpenRouterClient,
    runtime_seed: int,
    max_steps: int,
    work_key: str,
    evidence_mode: str = "raw",
) -> EpisodeResult:
    """LLM does everything: sees the data, chooses interventions, submits the graph."""
    env = BenchmarkEnv(instance, np.random.default_rng(runtime_seed))
    obs = env.observe()
    num_nodes = int(obs.shape[1])
    true_dag = instance.true_dag

    history: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    intervene_actions = 0

    base_payload: dict[str, Any] = {
        "variables": [f"X{i}" for i in range(num_nodes)],
        "n_rows": int(obs.shape[0]),
        "observational_means": [round(float(v), 3) for v in obs.mean(axis=0)],
        "observational_stds": [round(float(v), 3) for v in obs.std(axis=0, ddof=1)],
    }
    if evidence_mode == "raw":
        base_payload["observational_data"] = np.round(obs, 3).tolist()

    submission: GraphSubmission | None = None
    sanitize_stats: dict[str, int] = {}

    for step in range(1, max_steps + 1):
        final = step == max_steps or env.remaining_budget <= 0
        payload = dict(base_payload)
        payload["remaining_budget"] = env.remaining_budget
        payload["step"] = step
        payload["max_steps"] = max_steps
        payload["experiments"] = history
        instruction = (
            "You must submit_graph now using all evidence collected so far."
            if final
            else "Take your next action."
        )

        def validate(data: dict[str, Any]) -> None:
            action = str(data.get("action", "")).strip()
            if action not in {"intervene", "submit_graph"}:
                raise ValueError(f"action must be 'intervene' or 'submit_graph', got {action!r}")
            data["action"] = action
            if final and action != "submit_graph":
                raise ValueError("no budget or steps remain; you must call submit_graph")
            if action == "intervene":
                var = coerce_int(data.get("var"), "var")
                if not (0 <= var < num_nodes):
                    raise ValueError(f"var must be in 0..{num_nodes - 1}, got {var}")
                data["var"] = var
                data["value"] = coerce_float(data.get("value"), "value")

        response = client.call_tool(
            system_prompt=E2E_SYSTEM_PROMPT,
            user_prompt=(
                "Session JSON:\n"
                + json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
                + f"\n\n{instruction}"
            ),
            tool=E2E_TOOL,
            validate=validate,
            tag="e2e",
            context={"work_key": work_key, "step": step},
        )
        data = response.payload

        if data["action"] == "intervene":
            target = int(data["var"])
            value = float(data["value"])
            int_data = env.intervene(var=target, value=value)
            intervene_actions += 1
            history.append(
                {
                    "step": step,
                    "target": target,
                    "value": round(value, 3),
                    "n_rows": int(int_data.shape[0]),
                    "means": [round(float(v), 3) for v in int_data.mean(axis=0)],
                    "stds": [round(float(v), 3) for v in int_data.std(axis=0, ddof=1)],
                }
            )
            steps.append(
                {
                    "step": step,
                    "target": target,
                    "value": round(value, 4),
                    "selector_meta": json.dumps({"rule": "llm_e2e", "rationale": str(data.get("rationale", ""))[:400]}),
                }
            )
            continue

        submission, sanitize_stats = sanitize_graph(
            num_nodes, data.get("directed_edges", []), data.get("undirected_edges", [])
        )
        break

    if submission is None:
        submission = GraphSubmission(num_nodes=num_nodes, directed_edges=frozenset(), undirected_edges=frozenset())

    output = env.submit_graph(submission)
    scores = score_submission(instance, output.submission)
    metrics: dict[str, Any] = {
        **score_fields(scores),
        "submit_directed": len(output.submission.directed_edges),
        "submit_undirected": len(output.submission.undirected_edges),
        "steps_taken": intervene_actions,
        "pc_skeleton_f1_ceiling": "",
        "pc_truth_in_class": "",
        "true_dag_edges": len(true_dag.edges),
    }
    for key, value in sanitize_stats.items():
        metrics[f"infer_{key}"] = value
    return EpisodeResult(metrics=metrics, steps=steps)


__all__ = [
    "EpisodeResult",
    "run_decomposition_episode",
    "run_e2e_llm_episode",
    "run_pc",
    "score_fields",
    "skeleton_ceiling_f1",
    "truth_in_class",
]
