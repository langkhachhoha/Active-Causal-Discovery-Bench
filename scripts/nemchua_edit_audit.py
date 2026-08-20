#!/usr/bin/env python
"""Score each LLM skeleton proposal against the truth it was never shown.

    python scripts/nemchua_edit_audit.py study2/main study2/ladder_n60 ...

Runs launched before the audit columns existed only record *how many* edits a model
proposed, not how many were right. This replays the proposals out of `events.jsonl`,
rebuilds the instance and PC's skeleton deterministically from the manifest, and writes
`<run>/analysis/edit_audit.csv`, one row per (instance, model):

    n_remove / n_add            edits the model proposed, after the same filtering the
                                episode applies (removals must be adjacencies PC has,
                                additions must be pairs it does not)
    correct_remove / correct_add    of those, the ones matching PC's actual errors
    pc_fp / pc_fn               PC's own false positives and false negatives
    chance_remove / chance_add  what an editor with no information would hit, i.e. the
                                base rate of an error among the pairs it draws from

The chance columns are the point: a precision number alone cannot say whether a proposer
knows anything, because the base rate moves with sample size.
"""

from __future__ import annotations

import csv
import json
import sys
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from causal_discovery import BenchmarkEnv  # noqa: E402
from causal_discovery.active.episode import run_pc  # noqa: E402
from causal_discovery.active.levels import LEVELS, build_instance, runtime_seed_for  # noqa: E402
from causal_discovery.active.llm_client import resolve_model, short_model_name  # noqa: E402
from causal_discovery.equivalence.cpdag import canonical_undirected_edge  # noqa: E402

COLUMNS = [
    "level", "seed", "model_tag", "d", "n_obs",
    "n_remove", "correct_remove", "n_add", "correct_add",
    "pc_fp", "pc_fn", "n_pc_adj", "n_non_adj", "chance_remove", "chance_add", "pc_skeleton_f1",
    # additions that are wrong *in the specific way our prompt invites*: the pair are both
    # parents of a common child, so their partial correlation given all others is nonzero
    # even though they are not adjacent. See `moral_extras`.
    "add_spouse", "add_other", "n_spouse_available", "chance_spouse",
]


def moral_extras(dag) -> tuple[set, set]:
    """The DAG's skeleton, and the co-parent pairs the moral graph adds to it.

    The precision matrix of a linear-Gaussian DAG is supported on the *moral* graph, not
    the skeleton. Our proposer prompt states the opposite, so every pair of parents of a
    common child is a pair the prompt actively tells the model to add. Separating those
    additions from the rest says whether a model's errors come from ignorance or from
    following a false instruction correctly.
    """
    parents: dict[int, set[int]] = {c: set() for c in range(dag.num_nodes)}
    for a, b in dag.edges:
        parents[b].add(a)
    skeleton = {canonical_undirected_edge(a, b) for a, b in dag.edges}
    spouses = set()
    for child in range(dag.num_nodes):
        co = sorted(parents[child])
        for x in range(len(co)):
            for y in range(x + 1, len(co)):
                edge = canonical_undirected_edge(co[x], co[y])
                if edge not in skeleton:
                    spouses.add(edge)
    return skeleton, spouses


def normalise(pairs, num_nodes: int, limit: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for item in pairs if isinstance(pairs, list) else []:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        try:
            a, b = int(item[0]), int(item[1])
        except (TypeError, ValueError):
            continue
        if a == b or not (0 <= a < num_nodes and 0 <= b < num_nodes):
            continue
        edge = canonical_undirected_edge(a, b)
        if edge not in out:
            out.append(edge)
        if len(out) >= limit:
            break
    return out


def audit(run_dir: Path) -> list[dict]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    args = manifest["args"]
    slack = None if args["budget_slack"] < 0 else args["budget_slack"]
    limit = int(args["max_skeleton_edits"])
    alpha = float(args["alpha"])

    proposals: dict[tuple[int, int, str], tuple[list, list]] = {}
    events = run_dir / "events.jsonl"
    if not events.exists():
        return []
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
        level_tag, seed_tag, _arm, model = key.split("|")
        body = payload.get("payload") or {}
        # the work key carries the CLI alias; episodes.csv carries the resolved tag
        proposals[(int(level_tag[1:]), int(seed_tag[1:]), short_model_name(resolve_model(model)))] = (
            body.get("remove") or [], body.get("add") or []
        )

    cache: dict[tuple[int, int], tuple[set, set, int]] = {}
    rows: list[dict] = []
    for (level, seed, model), (raw_remove, raw_add) in sorted(proposals.items()):
        if (level, seed) not in cache:
            instance = build_instance(LEVELS[level], seed, args["n_obs"], args["n_int"], slack)
            env = BenchmarkEnv(instance, np.random.default_rng(runtime_seed_for(level, seed)))
            obs = env.observe()
            pc_pdag = run_pc(obs, alpha)
            truth, spouses = moral_extras(instance.true_dag)
            pc_adj = {canonical_undirected_edge(a, b) for a, b in pc_pdag.directed_edges} | set(
                pc_pdag.undirected_edges
            )
            cache[(level, seed)] = (truth, pc_adj, instance.true_dag.num_nodes, spouses)
        truth, pc_adj, d, spouses = cache[(level, seed)]

        remove = [e for e in normalise(raw_remove, d, limit) if e in pc_adj]
        add = [e for e in normalise(raw_add, d, limit) if e not in pc_adj]
        pc_fp = pc_adj - truth
        pc_fn = truth - pc_adj
        n_non_adj = comb(d, 2) - len(pc_adj)
        non_adjacent = {
            (i, j) for i in range(d) for j in range(i + 1, d) if (i, j) not in pc_adj
        }
        spouse_available = len(non_adjacent & spouses)
        rows.append({
            "level": level, "seed": seed, "model_tag": model, "d": d, "n_obs": args["n_obs"],
            "n_remove": len(remove), "correct_remove": sum(1 for e in remove if e in pc_fp),
            "n_add": len(add), "correct_add": sum(1 for e in add if e in truth),
            "pc_fp": len(pc_fp), "pc_fn": len(pc_fn),
            "n_pc_adj": len(pc_adj), "n_non_adj": n_non_adj,
            "chance_remove": round(len(pc_fp) / max(len(pc_adj), 1), 6),
            "chance_add": round(len(pc_fn) / max(n_non_adj, 1), 6),
            "add_spouse": sum(1 for e in add if e not in truth and e in spouses),
            "add_other": sum(1 for e in add if e not in truth and e not in spouses),
            "n_spouse_available": spouse_available,
            "chance_spouse": round(spouse_available / max(n_non_adj, 1), 6),
            "pc_skeleton_f1": round(2 * len(pc_adj & truth) / max(len(pc_adj) + len(truth), 1), 6),
        })
    return rows


def main() -> int:
    targets = [Path(x) for x in sys.argv[1:]] or sorted(
        p.parent for p in Path("study2").glob("*/run_manifest.json")
    )
    for run_dir in targets:
        rows = audit(run_dir)
        if not rows:
            print(f"[skip] {run_dir}: no repair proposals in events.jsonl")
            continue
        out = run_dir / "analysis"
        out.mkdir(parents=True, exist_ok=True)
        path = out / "edit_audit.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        for model in sorted({r["model_tag"] for r in rows}):
            sub = [r for r in rows if r["model_tag"] == model]
            proposed = sum(r["n_remove"] + r["n_add"] for r in sub)
            correct = sum(r["correct_remove"] + r["correct_add"] for r in sub)
            share_add = sum(r["n_add"] for r in sub) / max(proposed, 1)
            chance = (
                share_add * float(np.mean([r["chance_add"] for r in sub]))
                + (1 - share_add) * float(np.mean([r["chance_remove"] for r in sub]))
            )
            precision = correct / max(proposed, 1)
            wrong_adds = sum(r["add_spouse"] + r["add_other"] for r in sub)
            spouse_share = sum(r["add_spouse"] for r in sub) / max(wrong_adds, 1)
            spouse_chance = float(np.mean([r["chance_spouse"] for r in sub]))
            print(
                f"[{run_dir.name}] {model:18s} n={len(sub):3d} proposed={proposed:4d} "
                f"correct={correct:3d} precision={precision:.3f} chance={chance:.3f} "
                f"lift={precision / max(chance, 1e-9):.2f}x | "
                f"wrong-add spouse share={spouse_share:.3f} (chance {spouse_chance:.3f}, "
                f"{spouse_share / max(spouse_chance, 1e-9):.1f}x)"
            )
        print(f"          -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
