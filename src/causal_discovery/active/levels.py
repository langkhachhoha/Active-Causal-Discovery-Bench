"""Difficulty ladder for the active-experiment studies.

Deliberately smaller and denser-sampled than the parent benchmark's ladder: these
studies need exact Markov-equivalence-class enumeration, which is cheap up to d = 10.
`n_obs` / `n_int` are exposed on the CLI because the data-poor vs data-rich contrast
is one of the paper's ablation axes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from causal_discovery import build_benchmark_instance, make_v1_config


@dataclass(frozen=True, slots=True)
class LevelSpec:
    level_id: int
    d: int
    k: int
    noise_var: float = 1.0
    budget_slack: int = 1


# Calibrated so that the PC front-end is a competent but imperfect baseline:
# at n_obs = 300 its skeleton-F1 ceiling is ~0.95, it leaves 3-5 undirected edges,
# the true equivalence class has 7-9 members, and |I*| is 1.5-2.2. Denser graphs
# (k ~ 1.4 d) collapse the study: PC's skeleton is then wrong so often that every
# arm is capped at the same low ceiling and no experiment can help.
LEVELS: dict[int, LevelSpec] = {
    0: LevelSpec(0, d=4, k=4),
    1: LevelSpec(1, d=6, k=6),
    2: LevelSpec(2, d=8, k=8),
    3: LevelSpec(3, d=10, k=10),
    4: LevelSpec(4, d=12, k=12),
}


def parse_levels(text: str) -> list[int]:
    values = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not values:
        raise ValueError("no levels selected")
    for value in values:
        if value not in LEVELS:
            raise ValueError(f"unknown level {value}; available: {sorted(LEVELS)}")
    return values


def config_for(level: LevelSpec, n_obs: int, n_int: int):
    return make_v1_config(
        d=level.d,
        k=level.k,
        n_obs=n_obs,
        n_int=n_int,
        noise_var=level.noise_var,
        budget_slack=level.budget_slack,
    )


def build_instance(level: LevelSpec, seed: int, n_obs: int, n_int: int):
    return build_benchmark_instance(config_for(level, n_obs, n_int), np.random.default_rng(seed))


def _seed_accepts(level: LevelSpec, seed: int, n_obs: int, n_int: int) -> bool:
    try:
        build_instance(level, seed, n_obs, n_int)
    except RuntimeError:
        return False
    return True


def build_seed_map(
    levels: list[int],
    seeds_per_level: int,
    preflight_seed: int,
    n_obs: int,
    n_int: int,
    max_candidates: int = 20_000,
) -> dict[int, list[int]]:
    """Preflight-accepted seeds, identical for every arm so results are paired by instance."""
    rng = np.random.default_rng(preflight_seed)
    out: dict[int, list[int]] = {}
    for level_id in levels:
        level = LEVELS[level_id]
        accepted: list[int] = []
        seen: set[int] = set()
        attempts = 0
        while len(accepted) < seeds_per_level:
            if attempts >= max_candidates:
                raise RuntimeError(
                    f"only found {len(accepted)}/{seeds_per_level} accepted seeds for level {level_id}"
                )
            attempts += 1
            candidate = int(rng.integers(1, 2_147_483_647))
            if candidate in seen:
                continue
            seen.add(candidate)
            if _seed_accepts(level, candidate, n_obs, n_int):
                accepted.append(candidate)
        out[level_id] = accepted
    return out


def runtime_seed_for(level_id: int, seed: int) -> int:
    return int(seed * 10_000 + level_id * 101 + 7)
