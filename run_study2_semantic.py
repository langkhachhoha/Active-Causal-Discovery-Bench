"""NemChua — does the proposer improve when the variables have meaningful names?

The anonymized ladder strips exactly the thing LLMs are supposed to be good at: prior
knowledge about what the variables *are*. This study puts it back, on published DAG
structures whose node names carry real domain meaning, and measures the proposal channel
with and without them.

Every pair of conditions shares the graph, the parameters, the samples and the node
indices; only the strings shown to the model differ. So any change in edit precision is
attributable to the words alone.

    named   variables are presented under their published names, plus one domain sentence
    anon    the same variables are presented as X0..X(d-1)

Example
-------
    python run_study2_semantic.py --graphs asia,sachs --seeds 4 \
        --models gpt-4o-mini --out-dir study2/semantic_smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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
from causal_discovery.active.llm_client import (  # noqa: E402
    OpenRouterClient,
    resolve_api_key,
    resolve_model,
    short_model_name,
)
from causal_discovery.active.named_graphs import (  # noqa: E402
    DEFAULT_GRAPHS,
    NAMED_GRAPHS,
    build_named_instance,
    parse_graph_names,
)
from causal_discovery.active.probe import (  # noqa: E402
    ProposalCache,
    run_oracle_episode,
    run_pc_greedy_episode,
    run_probe_episode,
)
from run_study2_probe import EPISODE_COLUMNS, NEMCHUA_ARMS, STEP_COLUMNS, SUMMARY_METRICS  # noqa: E402

ARMS = (
    "oracle", "pc_greedy_meek",
    "probe",                # LLM skeleton repair -- the arm the naming condition acts on
    "probe_skel_only",      # no LLM at all; identical in both conditions, so it anchors them
    "probe_random_edits",   # random edits at the LLM's rate
    "probe_oracle_edits",   # perfect edits -- the ceiling
    "probe_stat_edits",     # our own instruction, executed mechanically, no world knowledge
    "probe_true_skeleton",  # the true adjacency set at any edit distance -- separates
                            # orientation failure from a proposal budget that is too small
)
CONDITIONS = ("named", "anon")
LLM_ARMS = {"probe", "probe_sepset"}

EXTRA_COLUMNS = ["graph", "condition"]


@dataclass(frozen=True)
class Work:
    arm: str
    model: str
    graph: str
    seed: int
    condition: str

    @property
    def key(self) -> str:
        return f"{self.graph}|s{self.seed}|{self.condition}|{self.arm}|{self.model or 'none'}"


def load_recorded_proposals(run_dir: Path) -> dict[tuple[str, int, str, str], tuple[list, list]]:
    """Proposals from a previous semantic run, keyed by (graph, seed, condition, model)."""
    events = run_dir / "events.jsonl"
    if not events.exists():
        raise SystemExit(f"--replay-proposals: no events.jsonl in {run_dir}")
    out: dict[tuple[str, int, str, str], tuple[list, list]] = {}
    for line in events.open(encoding="utf-8"):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event_type") != "llm_call:repair":
            continue
        payload = event["payload"]
        key = payload.get("work_key") or payload.get("key")
        if not key:
            continue
        graph, seed_tag, condition, _arm, model = key.split("|")
        body = payload.get("payload") or {}
        out[(graph, int(seed_tag[1:]), condition, model)] = (
            body.get("remove") or [], body.get("add") or []
        )
    if not out:
        raise SystemExit(f"--replay-proposals: {events} contains no repair proposals")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graphs", default=",".join(DEFAULT_GRAPHS))
    p.add_argument("--seeds", type=int, default=12)
    p.add_argument("--seed-base", type=int, default=20260820)
    p.add_argument("--models", default="qwen3-coder-30b,gpt-4o-mini")
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--conditions", default=",".join(CONDITIONS))
    p.add_argument("--out-dir", default="")
    p.add_argument("--env-file", default=".env")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--n-obs", type=int, default=300)
    p.add_argument("--n-int", type=int, default=150)
    p.add_argument("--max-hypotheses", type=int, default=48)
    p.add_argument("--eig-outcomes", type=int, default=12)
    p.add_argument("--max-skeleton-edits", type=int, default=4)
    p.add_argument("--max-skeleton-variants", type=int, default=10)
    p.add_argument("--max-dags-per-skeleton", type=int, default=1024)
    p.add_argument("--reserve-frac", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=4000)
    p.add_argument("--max-repairs", type=int, default=2)
    p.add_argument("--reasoning-effort", default="",
                   help="OpenRouter reasoning effort (low/medium/high); empty leaves it to the provider")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--replay-proposals", default="",
                   help="reuse the proposals recorded in another semantic run's events.jsonl "
                        "instead of calling the model (see run_study2_probe.py)")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    # Everything that shapes the instances comes from the manifest when resuming, not from
    # this invocation's defaults. Adding an arm to an existing run used to silently rebuild
    # the instances at the CLI default sample size, so the new rows described a different
    # world than the ones already in the file.
    if args.resume:
        prior = Path(args.out_dir) / "run_manifest.json" if args.out_dir else None
        if prior is not None and prior.exists():
            recorded_args = json.loads(prior.read_text(encoding="utf-8")).get("args", {})
            restored = {}
            for key in ("graphs", "seeds", "seed_base", "conditions", "alpha", "n_obs", "n_int",
                        "max_hypotheses", "eig_outcomes", "max_skeleton_edits",
                        "max_skeleton_variants", "max_dags_per_skeleton", "reserve_frac"):
                value = recorded_args.get(key)
                if value is not None and getattr(args, key) != value:
                    restored[key] = (getattr(args, key), value)
                    setattr(args, key, value)
            if restored:
                print("[resume] restored from manifest: " + ", ".join(
                    f"{k}={was!r}->{now!r}" for k, (was, now) in sorted(restored.items())), flush=True)
    graphs = parse_graph_names(args.graphs)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conditions:
        if c not in CONDITIONS:
            raise SystemExit(f"unknown condition {c!r}; expected one of {CONDITIONS}")

    out_dir = Path(args.out_dir) if args.out_dir else Path("study2/semantic") / run_id_now()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "run_manifest.json"
    checkpoint_path = out_dir / "checkpoint.json"
    recorded = load_recorded_proposals(Path(args.replay_proposals)) if args.replay_proposals else {}
    if recorded:
        print(f"[replay] {len(recorded)} recorded proposals from {args.replay_proposals}", flush=True)
    api_key = resolve_api_key(args.env_file) if (any(a in LLM_ARMS for a in arms) and not recorded) else ""

    if args.resume and manifest_path.exists():
        run_id = str(json.loads(manifest_path.read_text(encoding="utf-8"))["run_id"])
    else:
        run_id = run_id_now()
        write_json_atomic(
            manifest_path,
            {
                "run_id": run_id,
                "study": "study2_semantic",
                "created_at_utc": utc_now(),
                "graphs": {g: {"d": NAMED_GRAPHS[g].d, "k": NAMED_GRAPHS[g].k,
                               "nodes": list(NAMED_GRAPHS[g].nodes)} for g in graphs},
                "models": [resolve_model(m) for m in models],
                "arms": arms,
                "conditions": conditions,
                "args": vars(args),
            },
        )

    seeds = [args.seed_base + i for i in range(args.seeds)]
    work: list[Work] = []
    for g in graphs:
        for s in seeds:
            for cond in conditions:
                for arm in arms:
                    # a non-LLM arm cannot see names, so running it twice would only
                    # duplicate rows; it is evaluated once and shared by both conditions
                    if arm not in LLM_ARMS and cond != conditions[0]:
                        continue
                    if arm in LLM_ARMS:
                        for model in models:
                            work.append(Work(arm, model, g, s, cond))
                    else:
                        work.append(Work(arm, "", g, s, "shared"))
    if args.limit:
        work = work[: args.limit]

    trace = TraceWriter(out_dir / "events.jsonl")
    episodes = CsvSink(out_dir / "episodes.csv", EXTRA_COLUMNS + EPISODE_COLUMNS)
    steps_sink = CsvSink(out_dir / "steps.csv", ["graph", "condition"] + STEP_COLUMNS)
    write_lock = threading.Lock()
    cache_lock = threading.Lock()
    completed = load_checkpoint(checkpoint_path) if args.resume else {}
    checkpoint_state: dict[str, str] = dict(completed)
    instance_cache: dict[tuple[str, int], Any] = {}

    def get_instance(graph: str, seed: int):
        k = (graph, seed)
        with cache_lock:
            if k in instance_cache:
                return instance_cache[k]
        built = build_named_instance(NAMED_GRAPHS[graph], seed, args.n_obs, args.n_int)
        with cache_lock:
            instance_cache.setdefault(k, built)
            return instance_cache[k]

    total = len(work)
    counter = {"done": 0}

    def run_one(item: Work) -> None:
        prior = checkpoint_state.get(item.key, "")
        if prior == "success" or (prior == "failed" and not args.retry_failed):
            with write_lock:
                counter["done"] += 1
            return
        spec = NAMED_GRAPHS[item.graph]
        runtime_seed = item.seed * 131 + 17
        model_tag = short_model_name(item.model) if item.model else "none"
        row: dict[str, Any] = {
            "graph": item.graph, "condition": item.condition,
            "run_id": run_id, "timestamp_utc": utc_now(), "study": "study2_semantic",
            "arm": item.arm, "model": resolve_model(item.model) if item.model else "",
            "model_tag": model_tag, "level": -1, "seed": item.seed,
            "runtime_seed": runtime_seed, "d": spec.d, "k": spec.k,
            "n_obs": args.n_obs, "n_int": args.n_int,
        }
        started = time.perf_counter()
        client: OpenRouterClient | None = None
        try:
            instance, names = get_instance(item.graph, item.seed)
            row["budget"] = instance.intervention_budget
            row["opt_set_size"] = len(instance.optimal_intervention_set)
            row["true_edges"] = len(instance.true_dag.edges)
            row["cpdag_undirected"] = instance.observational_ceiling.num_undirected_edges

            if item.arm in LLM_ARMS:
                client = OpenRouterClient(
                    item.model, api_key,
                    temperature=args.temperature, max_tokens=args.max_tokens,
                    max_repairs=args.max_repairs,
                    reasoning_effort=args.reasoning_effort,
                    on_event=lambda event, payload: trace.log(event, {"key": item.key, **payload}),
                )
            if item.arm == "oracle":
                result = run_oracle_episode(instance=instance, runtime_seed=runtime_seed)
            elif item.arm == "pc_greedy_meek":
                result = run_pc_greedy_episode(
                    instance=instance, runtime_seed=runtime_seed, alpha=args.alpha, meek=True
                )
            else:
                arm_cfg = dict(NEMCHUA_ARMS[item.arm])
                reserve_frac = float(arm_cfg.pop("reserve_frac", args.reserve_frac))
                cache = None
                if recorded and item.arm in LLM_ARMS:
                    key = (item.graph, item.seed, item.condition, item.model)
                    if NEMCHUA_ARMS[item.arm].get("repair_evidence", "partial") != "partial":
                        key = (*key[:3], f"{item.model}#sepset")
                    if key not in recorded:
                        raise KeyError(f"no recorded proposal for {key}")
                    cache = ProposalCache()
                    cache.prime(*recorded[key])
                result = run_probe_episode(
                    instance=instance, proposal_cache=cache, client=client,
                    runtime_seed=runtime_seed, work_key=item.key, alpha=args.alpha,
                    max_hypotheses=args.max_hypotheses, eig_outcomes=args.eig_outcomes,
                    max_skeleton_edits=args.max_skeleton_edits,
                    max_skeleton_variants=args.max_skeleton_variants,
                    max_dags_per_skeleton=args.max_dags_per_skeleton,
                    reserve_frac=reserve_frac,
                    var_names=names if item.condition == "named" else None,
                    domain=spec.domain if item.condition == "named" else "",
                    **arm_cfg,
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
                steps_sink.write({
                    "graph": item.graph, "condition": item.condition, "run_id": run_id,
                    "arm": item.arm, "model_tag": model_tag, "level": -1, "seed": item.seed, **step,
                })
            checkpoint_state[item.key] = row["status"]
            write_json_atomic(checkpoint_path,
                              {"run_id": run_id, "updated_at_utc": utc_now(), "completed": checkpoint_state})
            print(f"[{counter['done']}/{total}] {item.key} -> {row['status']} "
                  f"dir_f1={row.get('directed_f1','')} rm_ok={row.get('edits_correct_remove','')} "
                  f"add_ok={row.get('edits_correct_add','')}", flush=True)

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(run_one, work))
    else:
        for item in work:
            run_one(item)

    trace.close()
    rows = read_rows(out_dir / "episodes.csv")
    aggregate(rows, ["arm", "condition", "model_tag"], SUMMARY_METRICS, out_dir / "summary_by_arm.csv")
    aggregate(rows, ["arm", "condition", "model_tag", "graph"], SUMMARY_METRICS, out_dir / "summary_by_graph.csv")
    print(f"\n[done] episodes = {out_dir / 'episodes.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
