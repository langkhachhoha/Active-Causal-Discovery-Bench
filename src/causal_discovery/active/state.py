"""Shared belief state and evidence record for the active-experiment studies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from causal_discovery.equivalence import CPDAG


@dataclass(frozen=True, slots=True)
class Evidence:
    """One completed experiment, reduced to the statistics every method may use."""

    step: int
    target: int
    value: float
    n_rows: int
    means: tuple[float, ...]
    stds: tuple[float, ...]
    data: np.ndarray

    def summary(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "target": self.target,
            "value": round(self.value, 3),
            "n_rows": self.n_rows,
            "means": [round(v, 3) for v in self.means],
            "stds": [round(v, 3) for v in self.stds],
        }


@dataclass(slots=True)
class BeliefState:
    """What a selector is allowed to see when choosing the next experiment."""

    num_nodes: int
    pdag: CPDAG
    obs_data: np.ndarray
    obs_means: tuple[float, ...]
    obs_stds: tuple[float, ...]
    evidence: list[Evidence] = field(default_factory=list)
    remaining_budget: int = 0
    step: int = 0

    @classmethod
    def create(cls, pdag: CPDAG, obs_data: np.ndarray, budget: int) -> "BeliefState":
        return cls(
            num_nodes=pdag.num_nodes,
            pdag=pdag,
            obs_data=obs_data,
            obs_means=tuple(float(v) for v in obs_data.mean(axis=0)),
            obs_stds=tuple(float(v) for v in obs_data.std(axis=0, ddof=1)),
            remaining_budget=int(budget),
            step=0,
        )

    def graph_payload(self) -> dict[str, Any]:
        return {
            "directed_edges": [list(edge) for edge in sorted(self.pdag.directed_edges)],
            "undirected_edges": [list(edge) for edge in sorted(self.pdag.undirected_edges)],
        }

    def evidence_payload(self) -> list[dict[str, Any]]:
        return [item.summary() for item in self.evidence]
