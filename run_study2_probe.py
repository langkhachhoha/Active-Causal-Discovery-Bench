"""Study 2 — PROBE: LLM-proposed hypothesis spaces + exact Bayesian experimental design.

Arms
----
    oracle              benchmark ceiling
    pc_greedy           parent benchmark's classical active baseline (no Meek closure)
    pc_greedy_meek      the same, with Meek closure after every intervention
    llm_e2e             end-to-end LLM agent (parent benchmark's `llm_raw`)

    probe               OURS: LLM skeleton repair + PC-MEC, BIC posterior, EIG selection, Bayes update

  hypothesis-space ablations (what the candidate set is made of)
    probe_repair_only   LLM-edited skeletons only, no PC equivalence class
    probe_llm_graphs    LLM invents whole DAGs (the naive way to use an LLM here)
    probe_skel_only     all acyclic orientations of PC's skeleton -- NO LLM; isolates the LLM's edits
    probe_mec_only      MEC of PC's CPDAG -- NO LLM
    probe_random_hyp    random DAGs -- space-quality floor

  decision-layer ablations (everything else held fixed)
    probe_random_sel    random experiment selection instead of EIG
    probe_maxdeg_sel    most-ambiguous-degree selection instead of EIG
    probe_no_bic        uniform prior over hypotheses (no likelihood re-scoring)
    probe_no_update     no Bayes update after each experiment
    probe_marginal      submit posterior edge marginals instead of the MAP graph

Example
-------
    python run_study2_probe.py --levels 0,1 --seeds-per-level 2 \
        --models qwen3-coder-30b --out-dir traces/study2/smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_discovery.active.episode import run_e2e_llm_episode  # noqa: E402
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
from causal_discovery.active.probe import (  # noqa: E402
    ProposalCache,
    run_oracle_episode,
    run_pc_greedy_episode,
    run_probe_episode,
)

# arm -> kwargs overrides for run_probe_episode
#
# The hypothesis-space ablation is the point of the study, so read the first column:
#   pc_mec        MEC of PC's CPDAG                          (no LLM)
#   pc_skeleton   all acyclic orientations of PC's skeleton  (no LLM — isolates the LLM's edits)
#   llm_repair    PC's skeleton edited by the LLM, orientations enumerated
#   llm_graphs    whole DAGs invented by the LLM
#   hybrid        llm_repair + pc_mec                        (this is PROBE)
#   random        random DAGs                                (space-quality floor)
PROBE_ARMS: dict[str, dict[str, Any]] = {
    "probe":            dict(hypothesis_source="hybrid",        select_rule="eig",    use_bic=True,  use_update=True,  submit_mode="map"),
    "probe_repair_only": dict(hypothesis_source="llm_repair",   select_rule="eig",    use_bic=True,  use_update=True,  submit_mode="map"),
    "probe_llm_graphs": dict(hypothesis_source="llm_graphs",    select_rule="eig",    use_bic=True,  use_update=True,  submit_mode="map"),
    "probe_skel_only":  dict(hypothesis_source="pc_skeleton",   select_rule="eig",    use_bic=True,  use_update=True,  submit_mode="map"),
    "probe_mec_only":   dict(hypothesis_source="pc_mec",        select_rule="eig",    use_bic=True,  use_update=True,  submit_mode="map"),
    "probe_random_hyp": dict(hypothesis_source="random",        select_rule="eig",    use_bic=True,  use_update=True,  submit_mode="map"),
    "probe_random_sel": dict(hypothesis_source="hybrid",        select_rule="random", use_bic=True,  use_update=True,  submit_mode="map"),
    "probe_maxdeg_sel": dict(hypothesis_source="hybrid",        select_rule="maxdeg", use_bic=True,  use_update=True,  submit_mode="map"),
    "probe_no_bic":     dict(hypothesis_source="hybrid",        select_rule="eig",    use_bic=False, use_update=True,  submit_mode="map"),
    "probe_no_update":  dict(hypothesis_source="hybrid",        select_rule="eig",    use_bic=True,  use_update=False, submit_mode="map"),
    "probe_marginal":   dict(hypothesis_source="hybrid",        select_rule="eig",    use_bic=True,  use_update=True,  submit_mode="marginal"),
}

BASELINE_ARMS = ("oracle", "pc_greedy", "pc_greedy_meek", "llm_e2e")
DEFAULT_ARMS = (
    "oracle", "pc_greedy", "pc_greedy_meek", "llm_e2e",
    "probe", "probe_repair_only", "probe_llm_graphs", "probe_skel_only", "probe_mec_only",
    "probe_random_hyp", "probe_random_sel", "probe_maxdeg_sel", "probe_no_bic",
    "probe_no_update", "probe_marginal",
)

_LLM_SOURCES = {"llm_repair", "llm_graphs", "hybrid", "hybrid_graphs"}
LLM_ARMS = {"llm_e2e"} | {name for name, cfg in PROBE_ARMS.items() if cfg["hypothesis_source"] in _LLM_SOURCES}

EPISODE_COLUMNS = [
    "run_id", "timestamp_utc", "study", "arm", "model", "model_tag", "level", "seed",
    "runtime_seed", "d", "k", "n_obs", "n_int", "budget", "opt_set_size", "true_edges",
    "cpdag_undirected", "status", "error", "wall_sec",
    "skeleton_precision", "skeleton_recall", "skeleton_f1", "compelled_f1",
    "directed_precision", "directed_recall", "directed_f1", "dag_shd", "efficiency",
    "interventions_used", "optimal_interventions", "submit_directed", "submit_undirected",
    "steps_taken", "n_hypotheses", "n_hypotheses_from_llm", "truth_in_hypotheses",
    "best_f1_in_hypotheses", "truth_rank_initial", "truth_rank_final",
    "entropy_initial_nats", "entropy_final_nats", "map_weight_final",
    "proposed_raw", "proposed_valid_unique", "propose_repairs", "propose_rounds",
    "repair_remove", "repair_add", "propose_cached",
    "pc_skeleton_f1_ceiling", "pc_undirected_edges", "pc_directed_edges",
    "llm_calls", "llm_failed_calls", "llm_repair_calls", "prompt_tokens",
    "completion_tokens", "cached_tokens", "total_tokens", "cost_usd", "llm_latency_sec",
]

STEP_COLUMNS = [
    "run_id", "arm", "model_tag", "level", "seed", "step", "target", "value",
    "n_hypotheses", "entropy_before_nats", "entropy_after_nats", "entropy_drop_nats",
    "map_weight_after", "map_directed_f1_after", "truth_rank_after", "edges_resolved",
    "undirected_after", "selector_meta",
]

SUMMARY_METRICS = [
    "directed_f1", "compelled_f1", "skeleton_f1", "dag_shd", "efficiency",
    "interventions_used", "steps_taken", "n_hypotheses", "truth_in_hypotheses",
    "best_f1_in_hypotheses", "truth_rank_final", "entropy_initial_nats",
    "entropy_final_nats", "map_weight_final", "proposed_raw", "proposed_valid_unique",
    "pc_skeleton_f1_ceiling", "prompt_tokens", "completion_tokens", "total_tokens",
    "cost_usd", "llm_calls", "llm_repair_calls", "wall_sec", "repair_remove", "repair_add",
]


@dataclass(frozen=True, slots=True)
class Work:
    arm: str
    model: str
    level_id: int
    seed: int

    @property
    def key(self) -> str:
        return f"L{self.level_id}|s{self.seed}|{self.arm}|{self.model or 'none'}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--levels", default="0,1,2,3")
    parser.add_argument("--seeds-per-level", type=int, default=8)
    parser.add_argument("--models", default="qwen3-coder-30b,gpt-4o-mini")
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-obs", type=int, default=200)
    parser.add_argument("--n-int", type=int, default=100)
    parser.add_argument("--preflight-seed", type=int, default=20260816)
    parser.add_argument("--num-candidates", type=int, default=12, help="graphs requested per proposal call")
    parser.add_argument("--propose-rounds", type=int, default=1)
    parser.add_argument("--max-hypotheses", type=int, default=48)
    parser.add_argument("--eig-outcomes", type=int, default=12, help="MC outcomes per EIG estimate")
    parser.add_argument("--budget-slack", type=int, default=-1,
                        help="override the intervention budget slack (budget = |I*| + slack); "
                             "-1 keeps each level's default of 1, 0 gives a tight budget")
    parser.add_argument("--no-skeleton-hint", action="store_true", help="hide the PC graph from the proposer")
    parser.add_argument("--max-skeleton-edits", type=int, default=4)
    parser.add_argument("--max-skeleton-variants", type=int, default=6)
    parser.add_argument("--max-dags-per-skeleton", type=int, default=1024)
    parser.add_argument("--e2e-max-steps", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--no-share-proposals", action="store_true",
                        help="let each ablation arm draw its own skeleton-repair proposal")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def make_work(arms: list[str], levels: list[int], seed_map: dict[int, list[int]], models: list[str]) -> list[Work]:
    items: list[Work] = []
    for level_id in levels:
        for seed in seed_map[level_id]:
            for arm in arms:
                if arm in LLM_ARMS:
                    for model in models:
                        items.append(Work(arm, model, level_id, seed))
                else:
                    items.append(Work(arm, "", level_id, seed))
    return items


def main() -> int:
    args = parse_args()
    levels = parse_levels(args.levels)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    known = set(PROBE_ARMS) | set(BASELINE_ARMS)
    for arm in arms:
        if arm not in known:
            raise SystemExit(f"unknown arm {arm!r}; available: {sorted(known)}")

    out_dir = Path(args.out_dir) if args.out_dir else Path("traces/study2") / run_id_now()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "run_manifest.json"
    checkpoint_path = out_dir / "checkpoint.json"

    api_key = resolve_api_key(args.env_file) if any(a in LLM_ARMS for a in arms) else ""

    if args.resume and manifest_path.exists():
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
                "study": "study2_probe",
                "created_at_utc": utc_now(),
                "levels": levels,
                "level_specs": {str(k): asdict(LEVELS[k]) for k in levels},
                "seeds_per_level": args.seeds_per_level,
                "seed_map": {str(k): v for k, v in seed_map.items()},
                "models": [resolve_model(m) for m in models],
                "arms": arms,
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
    cache_lock = threading.Lock()
    checkpoint_state: dict[str, str] = dict(completed)
    instance_cache: dict[tuple[int, int], Any] = {}
    proposal_caches: dict[tuple[int, int, str], ProposalCache] = {}

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
            "study": "study2",
            "arm": item.arm,
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

            if item.arm in LLM_ARMS:
                client = OpenRouterClient(
                    item.model,
                    api_key,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    max_repairs=args.max_repairs,
                    on_event=lambda event, payload: trace.log(event, {"key": item.key, **payload}),
                )

            if item.arm == "oracle":
                result = run_oracle_episode(instance=instance, runtime_seed=runtime_seed)
            elif item.arm in {"pc_greedy", "pc_greedy_meek"}:
                result = run_pc_greedy_episode(
                    instance=instance,
                    runtime_seed=runtime_seed,
                    alpha=args.alpha,
                    meek=(item.arm == "pc_greedy_meek"),
                )
            elif item.arm == "llm_e2e":
                result = run_e2e_llm_episode(
                    instance=instance,
                    client=client,
                    runtime_seed=runtime_seed,
                    max_steps=args.e2e_max_steps,
                    work_key=item.key,
                    evidence_mode="raw",
                )
            else:
                cache = None
                if not args.no_share_proposals:
                    cache_key = (item.level_id, item.seed, item.model)
                    with cache_lock:
                        cache = proposal_caches.setdefault(cache_key, ProposalCache())
                result = run_probe_episode(
                    instance=instance,
                    proposal_cache=cache,
                    client=client,
                    runtime_seed=runtime_seed,
                    work_key=item.key,
                    alpha=args.alpha,
                    num_candidates=args.num_candidates,
                    propose_rounds=args.propose_rounds,
                    max_hypotheses=args.max_hypotheses,
                    eig_outcomes=args.eig_outcomes,
                    skeleton_hint=not args.no_skeleton_hint,
                    max_skeleton_edits=args.max_skeleton_edits,
                    max_skeleton_variants=args.max_skeleton_variants,
                    max_dags_per_skeleton=args.max_dags_per_skeleton,
                    **PROBE_ARMS[item.arm],
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
                f"dir_f1={row.get('directed_f1', '')} truth_in_H={row.get('truth_in_hypotheses', '')}",
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
    aggregate(rows, ["arm", "model_tag"], SUMMARY_METRICS, out_dir / "summary_by_arm.csv")
    aggregate(rows, ["arm", "model_tag", "level"], SUMMARY_METRICS, out_dir / "summary_by_arm_level.csv")
    print(f"\n[done] episodes = {out_dir / 'episodes.csv'}")
    print(f"[done] steps    = {out_dir / 'steps.csv'}")
    print(f"[done] summary  = {out_dir / 'summary_by_arm.csv'}")
    print(f"[done] events   = {out_dir / 'events.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
