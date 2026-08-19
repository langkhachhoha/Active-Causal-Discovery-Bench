"""RauMa — isolating *local evidence interpretation* from *global graph reconstruction*.

The main study contrasts two readouts that do different amounts of work. The mechanical
readout decides one edge at a time, immediately after each experiment, using only the two
means of one variable. The LLM readout is handed the whole history at the end and must
rebuild the entire graph in one call. A gap between them can therefore come from bad
evidence interpretation *or* from the burden of global reconstruction.

This runner removes the second explanation. It replays exactly the mechanical agent's
decision points — same instances, same selector, same interventions, same undecided
neighbourhoods, same Meek closure — and substitutes an LLM for the mean-shift test at the
single point where evidence becomes an arrow. The LLM sees one tuple

    (a, b, xbar_b^obs, sd_b^obs, n_obs, xbar_b^int, sd_b^int, n_int)

and answers `a -> b`, `b -> a`, or `abstain`. Three prompt conditions vary how much of the
causal-discovery rule it is handed:

    stats   raw numbers only, no causal rule
    rule    + the explicit intervention-to-orientation rule
    rule_z  + the mechanically computed z statistic and the 1.96 convention

`--mode mechanical` runs the same loop with the fixed-threshold rule and costs no API
calls; it is both the reference arm and the source of the verifier reliability curve
(per-decision |Z| against orientation accuracy).

Examples
--------
    # reference arm + reliability data (free, no LLM)
    python run_study1_localreadout.py --mode mechanical \
        --seed-map-from study1/main/run_manifest.json --out-dir study1/localreadout/mechanical

    # one LLM prompt condition
    python run_study1_localreadout.py --mode llm --prompt rule --models qwen3-coder-30b \
        --seed-map-from study1/main/run_manifest.json --out-dir study1/localreadout/qwen_rule
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

import numpy as np

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_discovery import BenchmarkEnv, score_submission  # noqa: E402
from causal_discovery.scoring.submission import GraphSubmission  # noqa: E402
from causal_discovery.active.episode import run_pc, skeleton_ceiling_f1, truth_in_class  # noqa: E402
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
from causal_discovery.active.pdag import (  # noqa: E402
    MEAN_SHIFT_Z_975,
    _orient_incident,
    intervention_value,
    mean_shift_z,
)
from causal_discovery.active.selectors import build_selector, true_gain_per_target  # noqa: E402
from causal_discovery.active.state import BeliefState, Evidence  # noqa: E402


# --------------------------------------------------------------------------- #
# the one-edge decision interface
# --------------------------------------------------------------------------- #
ORIENT_TOOL = {
    "type": "function",
    "function": {
        "name": "orient_edge",
        "description": "State the direction of the single edge described in the prompt.",
        "parameters": {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": ["a_to_b", "b_to_a", "abstain"],
                    "description": "a_to_b means X_a -> X_b; b_to_a means X_b -> X_a; "
                                   "abstain leaves the edge undirected.",
                },
                "rationale": {"type": "string", "description": "One sentence."},
            },
            "required": ["decision", "rationale"],
            "additionalProperties": False,
        },
    },
}

_BASE = (
    "You are the evidence-interpretation module of a causal discovery system.\n"
    "The world is an unknown linear-Gaussian DAG with no hidden confounders and full "
    "observability.\n"
    "X_a and X_b are known to be adjacent; only the direction is unknown. An experiment "
    "forced X_a to a fixed value (a hard intervention that severs X_a's incoming edges) and "
    "drew a fresh sample. You are given X_b's mean and standard deviation in the "
    "observational panel and in the interventional panel.\n"
)

_RULE = (
    "Rule: because X_a and X_b are adjacent, X_b's mean moves under do(X_a) if and only if "
    "X_a -> X_b. If X_b's mean did not move, the edge runs the other way, X_b -> X_a. "
    "Judge 'moved' against sampling noise: the standard error of a difference of two means "
    "is sqrt(s_obs^2/n_obs + s_int^2/n_int).\n"
)

_ZLINE = (
    "The two-sample z statistic |xbar_int - xbar_obs| / SE has already been computed for you "
    "and is given as `z`. By convention a shift is called real when z > 1.96.\n"
)

_CLOSE = "Call orient_edge exactly once."

PROMPTS = {
    "stats": _BASE + _CLOSE,
    "rule": _BASE + _RULE + _CLOSE,
    "rule_z": _BASE + _RULE + _ZLINE + _CLOSE,
}


def decision_payload(rec: dict[str, Any], prompt_kind: str) -> str:
    payload = {
        "a": rec["a"],
        "b": rec["b"],
        "edge": f"X{rec['a']} -- X{rec['b']} (direction unknown)",
        "intervention": {"variable": f"X{rec['a']}", "value": round(rec["value"], 3)},
        "X_b_observational": {
            "mean": round(rec["mu_obs"], 4),
            "sd": round(rec["sd_obs"], 4),
            "n": rec["n_obs"],
        },
        "X_b_interventional": {
            "mean": round(rec["mu_int"], 4),
            "sd": round(rec["sd_int"], 4),
            "n": rec["n_int"],
        },
    }
    if prompt_kind == "rule_z":
        payload["se"] = round(rec["se"], 4)
        payload["z"] = round(rec["z"], 3)
    return (
        "Evidence JSON:\n"
        + json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
        + f"\n\nWhich way does the edge between X{rec['a']} and X{rec['b']} run?"
    )


# --------------------------------------------------------------------------- #
# episode
# --------------------------------------------------------------------------- #
EPISODE_COLUMNS = [
    "run_id", "timestamp_utc", "study", "arm", "selector", "readout", "prompt",
    "model", "model_tag", "level", "seed", "runtime_seed", "d", "k", "n_obs", "n_int",
    "budget", "opt_set_size", "true_edges", "status", "error", "wall_sec",
    "skeleton_f1", "compelled_f1", "directed_precision", "directed_recall", "directed_f1",
    "dag_shd", "interventions_used", "submit_directed", "submit_undirected", "steps_taken",
    "pc_skeleton_f1_ceiling", "pc_truth_in_class", "belief_undirected_final",
    "decisions", "decisions_correct", "decisions_wrong", "decisions_abstain",
    "orientations_correct", "orientations_wrong",
    "llm_calls", "llm_failed_calls", "llm_repair_calls",
    "prompt_tokens", "completion_tokens", "total_tokens", "cost_usd", "llm_latency_sec",
]

DECISION_COLUMNS = [
    "run_id", "arm", "model_tag", "prompt", "level", "seed", "step", "target",
    "a", "b", "z", "se", "mu_obs", "mu_int", "truth", "decision", "correct", "rationale",
]

SUMMARY_METRICS = [
    "directed_f1", "compelled_f1", "skeleton_f1", "dag_shd", "interventions_used",
    "belief_undirected_final", "decisions", "decisions_correct", "decisions_wrong",
    "decisions_abstain", "prompt_tokens", "completion_tokens", "total_tokens",
    "cost_usd", "llm_calls", "wall_sec",
]


@dataclass(frozen=True, slots=True)
class Work:
    level_id: int
    seed: int
    model: str

    @property
    def key(self) -> str:
        return f"L{self.level_id}|s{self.seed}|local|{self.model or 'none'}"


def run_local_episode(
    *,
    instance,
    selector,
    alpha: float,
    runtime_seed: int,
    mode: str,
    prompt_kind: str,
    client: OpenRouterClient | None,
    work_key: str,
    z_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Replay the mechanical agent's loop, swapping in the readout at the decision point."""
    env = BenchmarkEnv(instance, np.random.default_rng(runtime_seed))
    obs = env.observe()
    true_dag = instance.true_dag
    initial_pdag = run_pc(obs, alpha)
    state = BeliefState.create(initial_pdag, obs, env.remaining_budget)

    records: list[dict[str, Any]] = []
    correct_orientations = 0
    wrong_orientations = 0
    steps = 0

    while state.remaining_budget > 0 and state.pdag.undirected_edges:
        state.step += 1
        steps += 1
        before = state.pdag
        target = int(selector.choose(state).target)
        if target not in true_gain_per_target(before, true_dag):
            target = min(true_gain_per_target(before, true_dag), default=0)
        value = intervention_value(obs, target)
        int_data = env.intervene(var=target, value=value)

        n_obs = int(obs.shape[0])
        n_int = int(int_data.shape[0])

        def decide(a: int, b: int) -> tuple[int, int] | None:
            se = float(
                np.sqrt(obs[:, b].var(ddof=1) / n_obs + int_data[:, b].var(ddof=1) / n_int)
            )
            rec = {
                "step": state.step, "target": target, "a": a, "b": b,
                "value": float(value),
                "mu_obs": float(obs[:, b].mean()), "sd_obs": float(obs[:, b].std(ddof=1)),
                "mu_int": float(int_data[:, b].mean()), "sd_int": float(int_data[:, b].std(ddof=1)),
                "n_obs": n_obs, "n_int": n_int, "se": se,
                "z": mean_shift_z(obs, int_data, b),
                "truth": "a_to_b" if true_dag.has_edge(a, b)
                         else ("b_to_a" if true_dag.has_edge(b, a) else "none"),
                "rationale": "",
            }
            if mode == "mechanical":
                choice = "a_to_b" if rec["z"] > z_threshold else "b_to_a"
            else:
                response = client.call_tool(
                    system_prompt=PROMPTS[prompt_kind],
                    user_prompt=decision_payload(rec, prompt_kind),
                    tool=ORIENT_TOOL,
                    validate=_validate_decision,
                    tag="orient",
                    context={"work_key": work_key, "step": state.step, "a": a, "b": b},
                )
                choice = str(response.payload["decision"])
                rec["rationale"] = str(response.payload.get("rationale", ""))[:300]
            rec["decision"] = choice
            rec["correct"] = int(choice == rec["truth"])
            records.append(rec)
            if choice == "a_to_b":
                return (a, b)
            if choice == "b_to_a":
                return (b, a)
            return None

        after, _ = _orient_incident(before, target, decide)
        new_edges = set(after.directed_edges) - set(before.directed_edges)
        correct_orientations += sum(1 for s, d in new_edges if true_dag.has_edge(s, d))
        wrong_orientations += sum(1 for s, d in new_edges if not true_dag.has_edge(s, d))

        state.evidence.append(
            Evidence(
                step=state.step, target=target, value=float(value), n_rows=n_int,
                means=tuple(float(v) for v in int_data.mean(axis=0)),
                stds=tuple(float(v) for v in int_data.std(axis=0, ddof=1)),
                data=int_data,
            )
        )
        state.pdag = after
        state.remaining_budget = env.remaining_budget

    submission = GraphSubmission(
        num_nodes=state.num_nodes,
        directed_edges=state.pdag.directed_edges,
        undirected_edges=state.pdag.undirected_edges,
    )
    output = env.submit_graph(submission)
    scores = score_submission(instance, output.submission)

    metrics = {
        "skeleton_f1": scores.skeleton_f1,
        "compelled_f1": scores.compelled_f1,
        "directed_precision": scores.directed_precision,
        "directed_recall": scores.directed_recall,
        "directed_f1": scores.directed_f1,
        "dag_shd": scores.dag_shd,
        "interventions_used": scores.interventions_used,
        "submit_directed": len(output.submission.directed_edges),
        "submit_undirected": len(output.submission.undirected_edges),
        "steps_taken": steps,
        "pc_skeleton_f1_ceiling": round(skeleton_ceiling_f1(initial_pdag, true_dag), 6),
        "pc_truth_in_class": int(truth_in_class(initial_pdag, true_dag)),
        "belief_undirected_final": state.pdag.num_undirected_edges,
        "decisions": len(records),
        "decisions_correct": sum(r["correct"] for r in records),
        "decisions_wrong": sum(1 for r in records if r["decision"] != "abstain" and not r["correct"]),
        "decisions_abstain": sum(1 for r in records if r["decision"] == "abstain"),
        "orientations_correct": correct_orientations,
        "orientations_wrong": wrong_orientations,
    }
    return metrics, records


def _validate_decision(data: dict[str, Any]) -> None:
    if data.get("decision") not in {"a_to_b", "b_to_a", "abstain"}:
        raise ValueError("decision must be one of a_to_b, b_to_a, abstain")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--levels", default="0,1,2,3")
    p.add_argument("--seeds-per-level", type=int, default=10)
    p.add_argument("--seed-map-from", default="",
                   help="reuse the seed map of an earlier run_manifest.json (keeps this run "
                        "paired with the main study)")
    p.add_argument("--preflight-seed", type=int, default=20260816)
    p.add_argument("--mode", choices=("mechanical", "llm"), default="mechanical")
    p.add_argument("--prompt", choices=tuple(PROMPTS), default="rule")
    p.add_argument("--selector", default="oracle",
                   help="which policy picks the targets; the default matches the edge audit")
    p.add_argument("--models", default="qwen3-coder-30b,gpt-4o-mini")
    p.add_argument("--out-dir", default="")
    p.add_argument("--env-file", default=".env")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--n-obs", type=int, default=300)
    p.add_argument("--n-int", type=int, default=150)
    p.add_argument("--meanshift-z", type=float, default=MEAN_SHIFT_Z_975)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=600)
    p.add_argument("--max-repairs", type=int, default=2)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    levels = parse_levels(args.levels)
    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.mode == "llm" else [""]

    out_dir = Path(args.out_dir) if args.out_dir else Path("traces/study1_local") / run_id_now()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "run_manifest.json"
    checkpoint_path = out_dir / "checkpoint.json"

    api_key = resolve_api_key(args.env_file) if args.mode == "llm" else ""

    if args.resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_id = str(manifest["run_id"])
        seed_map = {int(k): [int(x) for x in v] for k, v in manifest["seed_map"].items()}
        levels = [int(x) for x in manifest["levels"]]
    else:
        run_id = run_id_now()
        if args.seed_map_from:
            source = json.loads(Path(args.seed_map_from).read_text(encoding="utf-8"))
            borrowed = {int(k): [int(x) for x in v] for k, v in source["seed_map"].items()}
            missing = [lv for lv in levels if lv not in borrowed]
            if missing:
                raise SystemExit(f"{args.seed_map_from} has no seeds for levels {missing}")
            seed_map = {lv: borrowed[lv][: args.seeds_per_level] for lv in levels}
        else:
            seed_map = build_seed_map(levels, args.seeds_per_level, args.preflight_seed,
                                      args.n_obs, args.n_int)
        write_json_atomic(manifest_path, {
            "run_id": run_id, "study": "study1_localreadout", "created_at_utc": utc_now(),
            "levels": levels, "seed_map": {str(k): v for k, v in seed_map.items()},
            "models": [resolve_model(m) for m in models if m], "args": vars(args),
        })

    arm = f"local_{args.mode}" + (f"_{args.prompt}" if args.mode == "llm" else "")
    completed = load_checkpoint(checkpoint_path) if args.resume else {}
    work = [Work(lv, sd, m) for lv in levels for sd in seed_map[lv] for m in models]

    trace = TraceWriter(out_dir / "events.jsonl")
    episodes = CsvSink(out_dir / "episodes.csv", EPISODE_COLUMNS)
    decisions = CsvSink(out_dir / "decisions.csv", DECISION_COLUMNS)
    lock = threading.Lock()
    checkpoint_state = dict(completed)
    total = len(work)
    counter = {"done": 0}

    def run_one(item: Work) -> None:
        if checkpoint_state.get(item.key) == "success":
            with lock:
                counter["done"] += 1
            return
        level = LEVELS[item.level_id]
        runtime_seed = runtime_seed_for(item.level_id, item.seed)
        model_tag = short_model_name(item.model) if item.model else "none"
        row: dict[str, Any] = {
            "run_id": run_id, "timestamp_utc": utc_now(), "study": "study1_local",
            "arm": arm, "selector": args.selector, "readout": args.mode,
            "prompt": args.prompt if args.mode == "llm" else "",
            "model": resolve_model(item.model) if item.model else "", "model_tag": model_tag,
            "level": item.level_id, "seed": item.seed, "runtime_seed": runtime_seed,
            "d": level.d, "k": level.k, "n_obs": args.n_obs, "n_int": args.n_int,
        }
        started = time.perf_counter()
        client: OpenRouterClient | None = None
        records: list[dict[str, Any]] = []
        try:
            instance = build_instance(level, item.seed, args.n_obs, args.n_int, None)
            row["budget"] = instance.intervention_budget
            row["opt_set_size"] = len(instance.optimal_intervention_set)
            row["true_edges"] = len(instance.true_dag.edges)
            if args.mode == "llm":
                client = OpenRouterClient(
                    item.model, api_key, temperature=args.temperature,
                    max_tokens=args.max_tokens, max_repairs=args.max_repairs,
                    on_event=lambda e, p: trace.log(e, {"key": item.key, **p}),
                )
            selector = build_selector(
                args.selector,
                rng=np.random.default_rng(runtime_seed + 991),
                true_dag=instance.true_dag,
                client=client,
                work_key=item.key,
            )
            metrics, records = run_local_episode(
                instance=instance, selector=selector, alpha=args.alpha,
                runtime_seed=runtime_seed, mode=args.mode, prompt_kind=args.prompt,
                client=client, work_key=item.key, z_threshold=args.meanshift_z,
            )
            row.update(metrics)
            row["status"] = "success"
            row["error"] = ""
        except Exception as exc:  # noqa: BLE001
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"[:500]
            trace.log("work_failed", {"key": item.key, "traceback": traceback.format_exc()[-2000:]})
        if client is not None:
            row.update(client.usage.as_row())
        row["wall_sec"] = round(time.perf_counter() - started, 3)

        with lock:
            counter["done"] += 1
            episodes.write(row)
            for rec in records:
                decisions.write({
                    "run_id": run_id, "arm": arm, "model_tag": model_tag,
                    "prompt": row["prompt"], "level": item.level_id, "seed": item.seed,
                    "step": rec["step"], "target": rec["target"], "a": rec["a"], "b": rec["b"],
                    "z": round(rec["z"], 5), "se": round(rec["se"], 5),
                    "mu_obs": round(rec["mu_obs"], 5), "mu_int": round(rec["mu_int"], 5),
                    "truth": rec["truth"], "decision": rec["decision"],
                    "correct": rec["correct"], "rationale": rec["rationale"],
                })
            checkpoint_state[item.key] = row["status"]
            write_json_atomic(checkpoint_path, {"run_id": run_id, "completed": checkpoint_state})
            print(f"[{counter['done']}/{total}] {item.key} -> {row['status']} "
                  f"dir_f1={row.get('directed_f1', '')} "
                  f"dec={row.get('decisions_correct', '')}/{row.get('decisions', '')}", flush=True)

    if args.workers > 1 and args.mode == "llm":
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(run_one, work))
    else:
        for item in work:
            run_one(item)

    trace.close()
    rows = read_rows(out_dir / "episodes.csv")
    aggregate(rows, ["arm", "model_tag", "prompt"], SUMMARY_METRICS, out_dir / "summary_by_arm.csv")
    print(f"\n[done] episodes  = {out_dir / 'episodes.csv'}")
    print(f"[done] decisions = {out_dir / 'decisions.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
