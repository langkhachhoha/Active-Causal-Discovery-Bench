"""Experiment-selection policies.

Every selector answers the same question: *given the current belief PDAG and the
experiments run so far, which variable should we intervene on next?*

The intervention **value** is held fixed across selectors (mean + 3 sd) so that
differences between arms are attributable to target choice alone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from causal_discovery.core import DAG
from causal_discovery.equivalence import CPDAG
from causal_discovery.active.llm_client import OpenRouterClient, coerce_int
from causal_discovery.active.mec import enumerate_mec, mec_entropy
from causal_discovery.active.pdag import open_targets, orient_from_truth, undirected_degree
from causal_discovery.active.state import BeliefState


@dataclass(frozen=True, slots=True)
class Selection:
    target: int
    meta: dict[str, Any]


class Selector(Protocol):
    name: str

    def choose(self, state: BeliefState) -> Selection: ...


# --------------------------------------------------------------------------- #
# analysis helpers (benchmark-owned; never exposed to a policy except `oracle`)
# --------------------------------------------------------------------------- #
def true_gain_per_target(pdag: CPDAG, true_dag: DAG) -> dict[int, int]:
    """For each candidate target: how many undirected edges intervening on it resolves."""
    gains: dict[int, int] = {}
    for target in open_targets(pdag):
        _, resolved = orient_from_truth(pdag, target, true_dag)
        gains[target] = int(resolved)
    return gains


def mec_target_entropy(
    pdag: CPDAG,
    *,
    max_members: int = 256,
    rng: np.random.Generator | None = None,
) -> tuple[dict[int, float], int, float, bool]:
    """Exact expected-information-gain per target over the equivalence class.

    An intervention on `a` reveals the true orientation of every undirected edge
    incident to `a`; members of the class that agree on those orientations remain
    indistinguishable. EIG(a) is therefore the entropy of that partition.

    Returns `(eig_per_target, mec_size, mec_entropy_nats, exhaustive)`.
    """
    members, exhaustive = enumerate_mec(pdag, max_members=max_members, rng=rng)
    n = len(members)
    if n <= 1:
        return ({t: 0.0 for t in open_targets(pdag)}, n, 0.0, exhaustive)

    prior_entropy = float(np.log(n))
    eig: dict[int, float] = {}
    for target in open_targets(pdag):
        incident = sorted(edge for edge in pdag.undirected_edges if target in edge)
        buckets: dict[tuple[int, ...], int] = {}
        for member in members:
            key = tuple(1 if (a, b) in member.edges else 0 for a, b in incident)
            buckets[key] = buckets.get(key, 0) + 1
        counts = np.array(list(buckets.values()), dtype=float)
        posterior_entropy = float(np.sum((counts / n) * np.log(counts)))
        eig[target] = prior_entropy - posterior_entropy
    return eig, n, prior_entropy, exhaustive


# --------------------------------------------------------------------------- #
# selectors
# --------------------------------------------------------------------------- #
class RandomSelector:
    name = "random"

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def choose(self, state: BeliefState) -> Selection:
        candidates = open_targets(state.pdag) or tuple(range(state.num_nodes))
        target = int(candidates[int(self._rng.integers(len(candidates)))])
        return Selection(target=target, meta={"rule": "uniform_over_open_targets"})


class MaxDegreeSelector:
    """The benchmark's existing heuristic: most undirected edges, ties by lowest index."""

    name = "maxdeg"

    def choose(self, state: BeliefState) -> Selection:
        degree = undirected_degree(state.pdag)
        if not degree:
            return Selection(target=0, meta={"rule": "no_open_targets"})
        target = min(degree.keys(), key=lambda node: (-degree[node], node))
        return Selection(target=int(target), meta={"rule": "max_undirected_degree", "degree": degree[target]})


class EIGSelector:
    """Bayesian optimal experimental design over the current equivalence class."""

    name = "eig"

    def __init__(self, rng: np.random.Generator, max_members: int = 256) -> None:
        self._rng = rng
        self._max_members = int(max_members)

    def choose(self, state: BeliefState) -> Selection:
        eig, size, entropy, exhaustive = mec_target_entropy(
            state.pdag, max_members=self._max_members, rng=self._rng
        )
        if not eig:
            return Selection(target=0, meta={"rule": "no_open_targets"})
        target = max(sorted(eig.keys()), key=lambda node: eig[node])
        return Selection(
            target=int(target),
            meta={
                "rule": "max_expected_information_gain",
                "eig_nats": round(float(eig[target]), 5),
                "mec_size": size,
                "mec_entropy_nats": round(entropy, 5),
                "mec_exhaustive": exhaustive,
            },
        )


class OracleSelector:
    """Upper bound: greedily maximise the number of edges actually resolved."""

    name = "oracle"

    def __init__(self, true_dag: DAG) -> None:
        self._true_dag = true_dag

    def choose(self, state: BeliefState) -> Selection:
        gains = true_gain_per_target(state.pdag, self._true_dag)
        if not gains:
            return Selection(target=0, meta={"rule": "no_open_targets"})
        target = max(sorted(gains.keys()), key=lambda node: gains[node])
        return Selection(target=int(target), meta={"rule": "max_true_gain", "gain": gains[target]})


SELECT_TOOL = {
    "type": "function",
    "function": {
        "name": "choose_experiment",
        "description": "Choose the single variable to intervene on next.",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "integer",
                    "description": "0-based index of the variable to intervene on.",
                },
                "rationale": {
                    "type": "string",
                    "description": "One or two sentences: what this experiment will disambiguate.",
                },
            },
            "required": ["target", "rationale"],
            "additionalProperties": False,
        },
    },
}

SELECT_SYSTEM_PROMPT = (
    "You are the experiment-design module of a causal discovery system.\n"
    "The world is an unknown linear-Gaussian DAG with no hidden confounders and full observability.\n"
    "Observational data can only identify the graph up to its Markov equivalence class: edges left "
    "undirected in the current graph are exactly the ones observational data cannot orient.\n"
    "An intervention do(X_a = v) severs every incoming edge of X_a and forces it to v. "
    "Variables downstream of X_a shift their mean; variables not downstream do not. "
    "Therefore intervening on X_a reveals the true orientation of every undirected edge incident to X_a, "
    "and orientations propagated from those by acyclicity and collider constraints.\n"
    "Your job is to pick the single target that resolves the most remaining direction uncertainty. "
    "Intervening on a variable with no incident undirected edges is a wasted experiment.\n"
    "Call choose_experiment exactly once."
)


class LLMSelector:
    name = "llm"

    def __init__(self, client: OpenRouterClient, work_key: str = "") -> None:
        self._client = client
        self._work_key = work_key
        self.wasted_targets = 0
        self.repeat_targets = 0
        self._history: list[int] = []

    def choose(self, state: BeliefState) -> Selection:
        candidates = open_targets(state.pdag)
        payload = {
            "variables": [f"X{i}" for i in range(state.num_nodes)],
            "current_graph": state.graph_payload(),
            "undirected_degree": undirected_degree(state.pdag),
            "targets_with_open_edges": list(candidates),
            "observational_stats": {
                "n_rows": int(state.obs_data.shape[0]),
                "means": [round(v, 3) for v in state.obs_means],
                "stds": [round(v, 3) for v in state.obs_stds],
            },
            "experiments_so_far": state.evidence_payload(),
            "remaining_budget": state.remaining_budget,
            "already_intervened_on": sorted(set(self._history)),
        }
        user_prompt = (
            "Session state JSON:\n"
            + json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
            + "\n\nChoose the next intervention target."
        )

        def validate(data: dict[str, Any]) -> None:
            target = coerce_int(data.get("target"), "target")
            if target < 0 or target >= state.num_nodes:
                raise ValueError(f"target {target} out of range 0..{state.num_nodes - 1}")
            data["target"] = target

        response = self._client.call_tool(
            system_prompt=SELECT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            tool=SELECT_TOOL,
            validate=validate,
            tag="select",
            context={"work_key": self._work_key, "step": state.step},
        )
        target = int(response.payload["target"])
        wasted = target not in candidates
        repeat = target in self._history
        self.wasted_targets += int(wasted)
        self.repeat_targets += int(repeat)
        self._history.append(target)
        return Selection(
            target=target,
            meta={
                "rule": "llm",
                "rationale": str(response.payload.get("rationale", ""))[:400],
                "wasted_target": wasted,
                "repeat_target": repeat,
                "repairs": response.repairs,
            },
        )


def build_selector(
    name: str,
    *,
    rng: np.random.Generator,
    true_dag: DAG,
    client: OpenRouterClient | None,
    work_key: str,
    eig_max_members: int = 256,
) -> Selector:
    if name == "random":
        return RandomSelector(rng)
    if name == "maxdeg":
        return MaxDegreeSelector()
    if name == "eig":
        return EIGSelector(rng, max_members=eig_max_members)
    if name == "oracle":
        return OracleSelector(true_dag)
    if name == "llm":
        if client is None:
            raise ValueError("selector 'llm' requires an OpenRouter client")
        return LLMSelector(client, work_key=work_key)
    raise ValueError(f"unknown selector: {name}")


SELECTOR_NAMES = ("random", "maxdeg", "eig", "llm", "oracle")
