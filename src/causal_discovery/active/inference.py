"""Structure-inference policies: turn collected evidence into a submitted graph."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from causal_discovery.equivalence import CPDAG
from causal_discovery.equivalence.cpdag import canonical_undirected_edge
from causal_discovery.scoring.submission import GraphSubmission
from causal_discovery.active.llm_client import OpenRouterClient
from causal_discovery.active.state import BeliefState


@dataclass(frozen=True, slots=True)
class Inference:
    submission: GraphSubmission
    meta: dict[str, Any]


class Inferencer(Protocol):
    name: str

    def infer(self, state: BeliefState, initial_pdag: CPDAG | None) -> Inference: ...


# --------------------------------------------------------------------------- #
# edge sanitisation shared by every LLM-produced graph
# --------------------------------------------------------------------------- #
def _has_path(adjacency: dict[int, list[int]], src: int, dst: int) -> bool:
    stack = [src]
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if node == dst:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(adjacency.get(node, []))
    return False


def sanitize_graph(
    num_nodes: int,
    directed_raw: list[Any],
    undirected_raw: list[Any],
) -> tuple[GraphSubmission, dict[str, int]]:
    """Build a valid submission, dropping malformed / contradictory edges and counting them."""
    stats = {
        "dropped_malformed": 0,
        "dropped_self_loop": 0,
        "dropped_out_of_range": 0,
        "dropped_cycle": 0,
        "dropped_overlap": 0,
        "dropped_duplicate": 0,
    }

    def parse_pair(item: Any) -> tuple[int, int] | None:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            stats["dropped_malformed"] += 1
            return None
        try:
            a, b = int(item[0]), int(item[1])
        except (TypeError, ValueError):
            stats["dropped_malformed"] += 1
            return None
        if a == b:
            stats["dropped_self_loop"] += 1
            return None
        if not (0 <= a < num_nodes and 0 <= b < num_nodes):
            stats["dropped_out_of_range"] += 1
            return None
        return (a, b)

    directed: set[tuple[int, int]] = set()
    adjacency: dict[int, list[int]] = {}
    seen_pairs: set[tuple[int, int]] = set()
    for item in directed_raw or []:
        pair = parse_pair(item)
        if pair is None:
            continue
        key = canonical_undirected_edge(*pair)
        if key in seen_pairs:
            stats["dropped_duplicate"] += 1
            continue
        if _has_path(adjacency, pair[1], pair[0]):
            stats["dropped_cycle"] += 1
            continue
        directed.add(pair)
        adjacency.setdefault(pair[0], []).append(pair[1])
        seen_pairs.add(key)

    undirected: set[tuple[int, int]] = set()
    for item in undirected_raw or []:
        pair = parse_pair(item)
        if pair is None:
            continue
        key = canonical_undirected_edge(*pair)
        if key in seen_pairs:
            stats["dropped_overlap"] += 1
            continue
        if key in undirected:
            stats["dropped_duplicate"] += 1
            continue
        undirected.add(key)

    submission = GraphSubmission(
        num_nodes=num_nodes,
        directed_edges=frozenset(directed),
        undirected_edges=frozenset(undirected),
    )
    return submission, stats


# --------------------------------------------------------------------------- #
# mechanical inferencer
# --------------------------------------------------------------------------- #
class MeekInferencer:
    """Submit the belief PDAG maintained by the harness (PC + mean-shift + Meek closure)."""

    name = "meek"

    def infer(self, state: BeliefState, initial_pdag: CPDAG | None) -> Inference:
        submission = GraphSubmission(
            num_nodes=state.num_nodes,
            directed_edges=state.pdag.directed_edges,
            undirected_edges=state.pdag.undirected_edges,
        )
        return Inference(submission=submission, meta={"rule": "pc_meanshift_meek"})


# --------------------------------------------------------------------------- #
# LLM inferencer
# --------------------------------------------------------------------------- #
SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_graph",
        "description": "Submit the final estimated causal graph.",
        "parameters": {
            "type": "object",
            "properties": {
                "directed_edges": {
                    "type": "array",
                    "description": "Edges you are confident about, as [source, target] index pairs.",
                    "items": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                },
                "undirected_edges": {
                    "type": "array",
                    "description": "Adjacencies whose direction remains unresolved, as [a, b] pairs.",
                    "items": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                },
                "rationale": {"type": "string", "description": "Brief justification."},
            },
            "required": ["directed_edges", "undirected_edges", "rationale"],
            "additionalProperties": False,
        },
    },
}

SUBMIT_SYSTEM_PROMPT = (
    "You are the structure-inference module of a causal discovery system.\n"
    "The world is an unknown linear-Gaussian DAG with no hidden confounders and full observability.\n"
    "You are given a partially oriented graph estimated from observational data, plus the results of "
    "one or more interventions.\n"
    "How to read an intervention do(X_a = v): X_a was forced to a constant, severing its incoming edges. "
    "Compare each variable's interventional mean with its observational mean. A variable whose mean moved "
    "is a descendant of X_a; a variable whose mean did not move is not. For an undirected edge {a, b}, "
    "a shift in X_b under do(X_a) means a -> b, and no shift means b -> a.\n"
    "Judge a shift against sampling noise: the standard error of a mean is sd / sqrt(n), so a difference "
    "of roughly two standard errors or more is real.\n"
    "Return a graph that is acyclic. Keep an edge undirected only if the evidence genuinely does not "
    "orient it; leaving an edge out entirely is different from leaving it undirected.\n"
    "Call submit_graph exactly once."
)


class LLMInferencer:
    name = "llm"

    def __init__(
        self,
        client: OpenRouterClient,
        work_key: str = "",
        *,
        evidence_mode: str = "summary",
    ) -> None:
        self._client = client
        self._work_key = work_key
        self._evidence_mode = evidence_mode

    def infer(self, state: BeliefState, initial_pdag: CPDAG | None) -> Inference:
        base = initial_pdag if initial_pdag is not None else state.pdag
        experiments: list[dict[str, Any]] = []
        for item in state.evidence:
            record = item.summary()
            if self._evidence_mode == "raw":
                record["rows"] = np.round(item.data, 3).tolist()
            experiments.append(record)

        payload: dict[str, Any] = {
            "variables": [f"X{i}" for i in range(state.num_nodes)],
            "observational_graph": {
                "directed_edges": [list(e) for e in sorted(base.directed_edges)],
                "undirected_edges": [list(e) for e in sorted(base.undirected_edges)],
                "note": "estimated from observational data by the PC algorithm",
            },
            "observational_stats": {
                "n_rows": int(state.obs_data.shape[0]),
                "means": [round(v, 3) for v in state.obs_means],
                "stds": [round(v, 3) for v in state.obs_stds],
            },
            "experiments": experiments,
        }
        if self._evidence_mode == "raw":
            payload["observational_rows"] = np.round(state.obs_data, 3).tolist()

        user_prompt = (
            "Evidence JSON:\n"
            + json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
            + "\n\nSubmit your final graph."
        )

        def validate(data: dict[str, Any]) -> None:
            for field in ("directed_edges", "undirected_edges"):
                if not isinstance(data.get(field), list):
                    raise ValueError(f"{field} must be a list of [i, j] pairs")

        response = self._client.call_tool(
            system_prompt=SUBMIT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            tool=SUBMIT_TOOL,
            validate=validate,
            tag="infer",
            context={"work_key": self._work_key},
        )
        submission, stats = sanitize_graph(
            state.num_nodes,
            response.payload.get("directed_edges", []),
            response.payload.get("undirected_edges", []),
        )
        meta: dict[str, Any] = {"rule": "llm", "repairs": response.repairs}
        meta.update(stats)
        meta["rationale"] = str(response.payload.get("rationale", ""))[:400]
        return Inference(submission=submission, meta=meta)


def build_inferencer(
    name: str,
    *,
    client: OpenRouterClient | None,
    work_key: str,
    evidence_mode: str = "summary",
) -> Inferencer:
    if name == "meek":
        return MeekInferencer()
    if name == "llm":
        if client is None:
            raise ValueError("inferencer 'llm' requires an OpenRouter client")
        return LLMInferencer(client, work_key=work_key, evidence_mode=evidence_mode)
    raise ValueError(f"unknown inferencer: {name}")


INFERENCER_NAMES = ("meek", "llm")
