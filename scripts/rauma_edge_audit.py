#!/usr/bin/env python
"""Edge-level audit of what each readout submits (RauMa, Study 1).

Reconstructs, for every instance, the true DAG and the shared PC front-end (both
deterministic given the seed), then classifies every directed edge the LLM readout
submitted -- read back from events.jsonl -- as correct, reversed, or spurious, and
splits it by whether the observational front-end had already oriented it.

    python scripts/rauma_edge_audit.py --study-dir study1 --out figures/edge_audit.json
"""
from __future__ import annotations

import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
from causal_discovery.active.episode import run_pc  # noqa: E402
from causal_discovery.active.levels import LEVELS, build_instance  # noqa: E402

TAGS = {"qwen3-coder-30b": "qwen3-coder-30b-a3b-instruct",
        "gpt-4o-mini": "gpt-4o-mini-2024-07-18"}

_cache: dict = {}


def instance_facts(level: int, seed: int, n_obs: int, n_int: int, alpha: float = 0.05):
    key = (level, seed, n_obs, n_int)
    if key in _cache:
        return _cache[key]
    inst = build_instance(LEVELS[level], seed, n_obs, n_int, 1)
    true_edges = {tuple(e) for e in inst.true_dag.edges}
    cpdag = run_pc(inst.observational_data.values if hasattr(inst.observational_data, "values")
                   else inst.observational_data, alpha)
    pc_directed = {frozenset(e) for e in cpdag.directed_edges}
    _cache[key] = (true_edges, pc_directed)
    return _cache[key]


def audit(study_dir: Path, run: str, n_obs: int, n_int: int) -> dict:
    """Classify every directed edge submitted by an LLM readout in `run`."""
    events = defaultdict(dict)          # dedupe: keep the last call per work key
    for line in (study_dir / run / "events.jsonl").open():
        e = json.loads(line)
        if e["event_type"] != "llm_call:infer":
            continue
        p = e["payload"]
        if p.get("status") != "ok" or not p.get("payload"):
            continue
        events[p["key"]] = p["payload"]

    out = defaultdict(Counter)
    for key, graph in events.items():
        lv, sd, arm, tag = key.split("|")
        lv, sd = int(lv[1:]), int(sd[1:])
        tag = TAGS.get(tag, tag)
        true_edges, pc_directed = instance_facts(lv, sd, n_obs, n_int)
        c = out[(arm, tag)]
        c["episodes"] += 1
        for edge in graph.get("directed_edges") or []:
            u, v = int(edge[0]), int(edge[1])
            group = "pc" if frozenset((u, v)) in pc_directed else "int"
            c["total"] += 1
            if (u, v) in true_edges:
                c["correct"] += 1; c[group + "_n"] += 1
            elif (v, u) in true_edges:
                c["reversed"] += 1; c[group + "_n"] += 1; c[group + "_rev"] += 1
            else:
                c["spurious"] += 1
        c["undirected"] += len(graph.get("undirected_edges") or [])
    return {f"{a}|{t}": dict(v) for (a, t), v in out.items()}


def mechanical(study_dir: Path, run: str) -> dict:
    """Same classification for the mean-shift + Meek readout, from the episode CSV.

    Those arms submit no undirected edges, so directed and skeleton precision pin the
    correct / reversed / spurious split exactly; `orientations_wrong` counts errors on
    the edges the interventions resolved, and the remainder falls on the edges the PC
    front-end had already oriented.
    """
    df = pd.read_csv(study_dir / run / "episodes.csv")
    df = df[(df.status == "success") & (df.inferencer == "meek")]
    out = {}
    for (arm, tag), x in df.groupby(["arm", "model_tag"]):
        if x.submit_undirected.sum():
            continue
        total = float(x.submit_directed.sum())
        correct = float((x.directed_precision * x.submit_directed).sum())
        adjacent = float((x.skeleton_precision * x.submit_directed).sum())
        int_n = float(x.orientations_correct.sum() + x.orientations_wrong.sum())
        int_rev = float(x.orientations_wrong.sum())
        pc_n = float(x.pc_directed_edges.sum())
        out[f"{arm}|{tag}"] = {
            "episodes": len(x), "total": total, "correct": correct,
            "reversed": adjacent - correct, "spurious": total - adjacent,
            "int_n": int_n, "int_rev": int_rev,
            "pc_n": pc_n, "pc_rev": (adjacent - correct) - int_rev,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-dir", default="study1")
    ap.add_argument("--out", default="figures/edge_audit.json")
    args = ap.parse_args()
    sd = Path(args.study_dir)

    result = {
        "main_llm": audit(sd, "main", 300, 150),
        "main_mech": mechanical(sd, "main"),
        "raw_llm": audit(sd, "ablation_rawevidence", 300, 150),
    }
    Path(args.out).write_text(json.dumps(result, indent=1))
    for block, rows in result.items():
        print(f"--- {block}")
        for k, c in sorted(rows.items()):
            t = max(c.get("total", 0), 1)
            print(f"  {k:46s} edges={c.get('total',0):6.0f} "
                  f"corr={c.get('correct',0)/t:6.1%} rev={c.get('reversed',0)/t:6.1%} "
                  f"spur={c.get('spurious',0)/t:6.1%} "
                  f"| int {c.get('int_rev',0):.0f}/{c.get('int_n',0):.0f} "
                  f"pc {c.get('pc_rev',0):.0f}/{c.get('pc_n',0):.0f}")
    print(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
