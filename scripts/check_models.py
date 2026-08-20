#!/usr/bin/env python
"""Ping every model with one real proposal call before committing to a long run.

A capability sweep dies badly if one model id is wrong or unavailable: the failure
surfaces hours in, after the other models have already been paid for. This makes the
same call the study makes, on a throwaway instance, and prints one line per model.

    python scripts/check_models.py --models qwen3-coder-30b,gpt-4o-mini,haiku-4.5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from causal_discovery.active.episode import run_pc  # noqa: E402
from causal_discovery.active.levels import LEVELS, build_instance  # noqa: E402
from causal_discovery.active.llm_client import (  # noqa: E402
    OpenRouterClient,
    resolve_api_key,
    resolve_model,
)
from causal_discovery.active.probe import propose_skeleton_edits_llm  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="qwen3-coder-30b,gpt-4o-mini,gpt-5.4-mini,haiku-4.5,gemini-3-flash")
    p.add_argument("--env-file", default=".env")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=4000)
    p.add_argument("--reasoning-effort", default="")
    args = p.parse_args()

    key = resolve_api_key(args.env_file)
    instance = build_instance(LEVELS[1], 424242, 300, 150, None)
    obs = instance.observational_data
    pc_pdag = run_pc(obs, 0.05)

    ok = True
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        resolved = resolve_model(name)
        client = OpenRouterClient(
            name, key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort,
        )
        started = time.perf_counter()
        try:
            remove, add, _ = propose_skeleton_edits_llm(
                client, obs, pc_pdag, work_key="preflight", max_edits=4
            )
            usage = client.usage.as_row()
            print(
                f"[ok]   {name:18s} -> {resolved:42s} "
                f"remove={len(remove)} add={len(add)} "
                f"tokens={usage['total_tokens']:6d} cost=${usage['cost_usd']:.5f} "
                f"{time.perf_counter() - started:5.1f}s"
            )
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"[FAIL] {name:18s} -> {resolved:42s} {type(exc).__name__}: {str(exc)[:180]}")
    if not ok:
        print("\nAt least one model is unusable. Fix the id (or drop it) before starting a sweep.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
