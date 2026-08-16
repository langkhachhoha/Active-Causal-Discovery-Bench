"""Study 1 — Decomposing active causal discovery into selection and inference.

An active causal-discovery agent does two different jobs: it *chooses experiments*
and it *infers structure from their results*. A single end-to-end score cannot say
which one is broken. This runner factorises the agent into

    selector   in {random, maxdeg, eig, llm, oracle}
    inferencer in {meek, llm}

and runs the full cross-product on paired instances, plus the end-to-end LLM agent
as a reference. Every arm shares the same PC front-end, the same intervention values
and the same instances, so the difference between arms is attributable.

Example
-------
    python run_study1_decompose.py --levels 0,1 --seeds-per-level 2 \
        --models qwen3-coder-30b --out-dir traces/study1/smoke
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_discovery.active.episode import (  # noqa: E402
    run_decomposition_episode,
    run_e2e_llm_episode,
)
from causal_discovery.active.inference import INFERENCER_NAMES, build_inferencer  # noqa: E402
from causal_discovery.active.io import (  # noqa: E402
    CsvSink,
    TraceWriter,
    aggregate,
    load_checkpoint,
    read_rows,
    run_id_now,
    utc_now,
    write_json_atomic,
)
from causal_discovery.active.levels import (  # noqa: E402
    LEVELS,
    build_instance,
    build_seed_map,
    parse_levels,
    runtime_seed_for,
)
from causal_discovery.active.llm_client import (  # noqa: E402
    OpenRouterClient,
    resolve_api_key,
    resolve_model,
    short_model_name,
)
from causal_discovery.active.selectors import SELECTOR_NAMES, build_selector  # noqa: E402

E2E_ARM = "llm_e2e"

EPISODE_COLUMNS = [
    "run_id", "timestamp_utc", "study", "arm", "selector", "inferencer", "model", "model_tag",
    "level", "seed", "runtime_seed", "d", "k", "n_obs", "n_int", "budget", "opt_set_size",
    "true_edges", "cpdag_undirected", "status", "error", "wall_sec",
    "skeleton_precision", "skeleton_recall", "skeleton_f1", "compelled_f1",
    "directed_precision", "directed_recall", "directed_f1", "dag_shd", "efficiency",
    "interventions_used", "optimal_interventions", "submit_directed", "submit_undirected",
    "steps_taken", "pc_skeleton_f1_ceiling", "pc_truth_in_class", "pc_undirected_edges",
    "pc_directed_edges", "mec_size_initial", "mec_entropy_initial", "belief_undirected_final",
    "selection_regret_total", "selection_regret_mean", "eig_regret_total",
    "selection_quality_mean", "wasted_steps", "orientations_correct", "orientations_wrong",
    "orientation_accuracy", "select_wasted_targets", "select_repeat_targets",
    "infer_rule", "infer_repairs", "infer_dropped_malformed", "infer_dropped_self_loop",
    "infer_dropped_out_of_range", "infer_dropped_cycle", "infer_dropped_overlap",
    "infer_dropped_duplicate",
    "llm_calls", "llm_failed_calls", "llm_repair_calls", "prompt_tokens", "completion_tokens",
    "cached_tokens", "total_tokens", "cost_usd", "llm_latency_sec",
]

STEP_COLUMNS = [
    "run_id", "arm", "model_tag", "level", "seed", "step", "target", "value",
    "undirected_before", "undirected_after", "mec_size_before", "mec_entropy_before",
    "mec_exhaustive", "true_gain_chosen", "true_gain_best", "selection_regret",
    "selection_quality", "eig_chosen_nats", "eig_best_nats", "eig_regret_nats",
    "edges_resolved", "orientations_correct", "orientations_wrong", "selector_meta",
]

SUMMARY_METRICS = [
    "directed_f1", "compelled_f1", "skeleton_f1", "dag_shd", "efficiency",
    "selection_regret_total", "selection_regret_mean", "selection_quality_mean",
    "eig_regret_total", "orientation_accuracy", "wasted_steps", "steps_taken",
    "interventions_used", "pc_skeleton_f1_ceiling", "pc_truth_in_class",
    "mec_size_initial", "mec_entropy_initial", "belief_undirected_final",
    "prompt_tokens", "completion_tokens", "total_tokens", "cost_usd", "llm_calls",
    "llm_repair_calls", "wall_sec",
]


@dataclass(frozen=True, slots=True)
class Work:
    arm: str
    selector: str
    inferencer: str
    model: str
    level_id: int
    seed: int

    @property
    def key(self) -> str:
        return f"L{self.level_id}|s{self.seed}|{self.arm}|{self.model or 'none'}"

    @property
    def needs_llm(self) -> bool:
        return self.selector == "llm" or self.inferencer == "llm" or self.arm == E2E_ARM


def build_arms(selectors: list[str], inferencers: list[str], include_e2e: bool) -> list[tuple[str, str, str]]:
    arms = [(f"{s}+{i}", s, i) for s in selectors for i in inferencers]
    if include_e2e:
        arms.append((E2E_ARM, "llm_e2e", "llm_e2e"))
    return arms


def make_work(
    arms: list[tuple[str, str, str]],
    levels: list[int],
    seed_map: dict[int, list[int]],
    models: list[str],
) -> list[Work]:
    items: list[Work] = []
    for level_id in levels:
        for seed in seed_map[level_id]:
            for arm, selector, inferencer in arms:
                needs_llm = selector in {"llm", "llm_e2e"} or inferencer in {"llm", "llm_e2e"}
                if needs_llm:
                    for model in models:
                        items.append(Work(arm, selector, inferencer, model, level_id, seed))
                else:
                    items.append(Work(arm, selector, inferencer, "", level_id, seed))
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--levels", default="0,1,2,3")
    parser.add_argument("--seeds-per-level", type=int, default=8)
    parser.add_argument("--models", default="qwen3-coder-30b,gpt-4o-mini")
    parser.add_argument("--selectors", default=",".join(SELECTOR_NAMES))
    parser.add_argument("--inferencers", default=",".join(INFERENCER_NAMES))
    parser.add_argument("--no-e2e", action="store_true", help="skip the end-to-end LLM reference arm")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-obs", type=int, default=200)
    parser.add_argument("--n-int", type=int, default=100)
    parser.add_argument("--preflight-seed", type=int, default=20260816)
    parser.add_argument("--eig-max-members", type=int, default=256)
    parser.add_argument("--budget-slack", type=int, default=-1,
                        help="override the intervention budget slack (budget = |I*| + slack); "
                             "-1 keeps each level's default of 1, 0 gives a tight budget")
    parser.add_argument("--evidence-mode", choices=("summary", "raw"), default="summary",
                        help="what the LLM inferencer sees: sufficient statistics or raw rows")
    parser.add_argument("--e2e-max-steps", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="debug: cap the number of work items")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    levels = parse_levels(args.levels)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    selectors = [s.strip() for s in args.selectors.split(",") if s.strip()]
    inferencers = [s.strip() for s in args.inferencers.split(",") if s.strip()]
    for name in selectors:
        if name not in SELECTOR_NAMES:
            raise SystemExit(f"unknown selector {name!r}; available: {SELECTOR_NAMES}")
    for name in inferencers:
        if name not in INFERENCER_NAMES:
            raise SystemExit(f"unknown inferencer {name!r}; available: {INFERENCER_NAMES}")

    out_dir = Path(args.out_dir) if args.out_dir else Path("traces/study1") / run_id_now()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "run_manifest.json"
    checkpoint_path = out_dir / "checkpoint.json"

    arms = build_arms(selectors, inferencers, include_e2e=not args.no_e2e)
    needs_key = any(
        s in {"llm", "llm_e2e"} or i in {"llm", "llm_e2e"} for _, s, i in arms
    )
    api_key = resolve_api_key(args.env_file) if needs_key else ""

    if args.resume and manifest_path.exists():
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_id = str(manifest["run_id"])
        seed_map = {int(k): [int(x) for x in v] for k, v in manifest["seed_map"].items()}
        levels = [int(x) for x in manifest["levels"]]
    else:
        run_id = run_id_now()
        print(f"[preflight] building seed map for levels {levels} ...", flush=True)
        seed_map = build_seed_map(levels, args.seeds_per_level, args.preflight_seed, args.n_obs, args.n_int)
        write_json_atomic(
            manifest_path,
            {
                "run_id": run_id,
                "study": "study1_decompose",
                "created_at_utc": utc_now(),
                "levels": levels,
                "level_specs": {str(k): asdict(LEVELS[k]) for k in levels},
                "seeds_per_level": args.seeds_per_level,
                "seed_map": {str(k): v for k, v in seed_map.items()},
                "models": [resolve_model(m) for m in models],
                "arms": [arm for arm, _, _ in arms],
                "args": vars(args),
            },
        )

    # None keeps each level's own slack; 0 makes the budget exactly |I*|, which is the
    # regime where a wasted experiment can no longer be recovered from.
    budget_slack = None if args.budget_slack < 0 else args.budget_slack

    completed = load_checkpoint(checkpoint_path) if args.resume else {}
    work = make_work(arms, levels, seed_map, models)
    if args.limit:
        work = work[: args.limit]

    trace = TraceWriter(out_dir / "events.jsonl")
    episodes = CsvSink(out_dir / "episodes.csv", EPISODE_COLUMNS)
    steps_sink = CsvSink(out_dir / "steps.csv", STEP_COLUMNS)
    write_lock = threading.Lock()
    checkpoint_state: dict[str, str] = dict(completed)

    instance_cache: dict[tuple[int, int], Any] = {}
    cache_lock = threading.Lock()

    def get_instance(level_id: int, seed: int):
        key = (level_id, seed)
        with cache_lock:
            if key in instance_cache:
                return instance_cache[key]
        built = build_instance(LEVELS[level_id], seed, args.n_obs, args.n_int, budget_slack)
        with cache_lock:
            instance_cache.setdefault(key, built)
            return instance_cache[key]

    total = len(work)
    counter = {"done": 0}

    def run_one(item: Work) -> None:
        prior = checkpoint_state.get(item.key, "")
        if prior == "success" or (prior == "failed" and not args.retry_failed):
            with write_lock:
                counter["done"] += 1
            return

        level = LEVELS[item.level_id]
        runtime_seed = runtime_seed_for(item.level_id, item.seed)
        model_tag = short_model_name(item.model) if item.model else "none"
        row: dict[str, Any] = {
            "run_id": run_id,
            "timestamp_utc": utc_now(),
            "study": "study1",
            "arm": item.arm,
            "selector": item.selector,
            "inferencer": item.inferencer,
            "model": resolve_model(item.model) if item.model else "",
            "model_tag": model_tag,
            "level": item.level_id,
            "seed": item.seed,
            "runtime_seed": runtime_seed,
            "d": level.d,
            "k": level.k,
            "n_obs": args.n_obs,
            "n_int": args.n_int,
        }
        started = time.perf_counter()
        client: OpenRouterClient | None = None
        try:
            instance = get_instance(item.level_id, item.seed)
            row["budget"] = instance.intervention_budget
            row["opt_set_size"] = len(instance.optimal_intervention_set)
            row["true_edges"] = len(instance.true_dag.edges)
            row["cpdag_undirected"] = instance.observational_ceiling.num_undirected_edges

            if item.needs_llm:
                client = OpenRouterClient(
                    item.model,
                    api_key,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    max_repairs=args.max_repairs,
                    on_event=lambda event, payload: trace.log(event, {"key": item.key, **payload}),
                )

            if item.arm == E2E_ARM:
                result = run_e2e_llm_episode(
                    instance=instance,
                    client=client,
                    runtime_seed=runtime_seed,
                    max_steps=args.e2e_max_steps,
                    work_key=item.key,
                    evidence_mode="raw",
                )
            else:
                selector = build_selector(
                    item.selector,
                    rng=np.random.default_rng(runtime_seed + 991),
                    true_dag=instance.true_dag,
                    client=client,
                    work_key=item.key,
                    eig_max_members=args.eig_max_members,
                )
                inferencer = build_inferencer(
                    item.inferencer,
                    client=client,
                    work_key=item.key,
                    evidence_mode=args.evidence_mode,
                )
                result = run_decomposition_episode(
                    instance=instance,
                    selector=selector,
                    inferencer=inferencer,
                    alpha=args.alpha,
                    runtime_seed=runtime_seed,
                    eig_max_members=args.eig_max_members,
                )

            row.update(result.metrics)
            row["status"] = "success"
            row["error"] = ""
            step_rows = result.steps
        except Exception as exc:  # noqa: BLE001
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"[:500]
            step_rows = []
            trace.log("work_failed", {"key": item.key, "traceback": traceback.format_exc()[-2000:]})

        if client is not None:
            row.update(client.usage.as_row())
        row["wall_sec"] = round(time.perf_counter() - started, 3)

        with write_lock:
            counter["done"] += 1
            episodes.write(row)
            for step in step_rows:
                steps_sink.write(
                    {
                        "run_id": run_id,
                        "arm": item.arm,
                        "model_tag": model_tag,
                        "level": item.level_id,
                        "seed": item.seed,
                        **step,
                    }
                )
            checkpoint_state[item.key] = row["status"]
            write_json_atomic(
                checkpoint_path,
                {"run_id": run_id, "updated_at_utc": utc_now(), "completed": checkpoint_state},
            )
            print(
                f"[{counter['done']}/{total}] {item.key} -> {row['status']} "
                f"dir_f1={row.get('directed_f1', '')} regret={row.get('selection_regret_total', '')}",
                flush=True,
            )

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(run_one, work))
    else:
        for item in work:
            run_one(item)

    trace.close()
    rows = read_rows(out_dir / "episodes.csv")
    aggregate(rows, ["arm", "selector", "inferencer", "model_tag"], SUMMARY_METRICS, out_dir / "summary_by_arm.csv")
    aggregate(rows, ["arm", "model_tag", "level"], SUMMARY_METRICS, out_dir / "summary_by_arm_level.csv")
    print(f"\n[done] episodes = {out_dir / 'episodes.csv'}")
    print(f"[done] steps    = {out_dir / 'steps.csv'}")
    print(f"[done] summary  = {out_dir / 'summary_by_arm.csv'}")
    print(f"[done] events   = {out_dir / 'events.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
