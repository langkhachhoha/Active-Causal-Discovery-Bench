#!/usr/bin/env python
"""Is an LLM's proposal worth more than a random one *of the same size*?

    python scripts/nemchua_permutation.py study2_new/models_n60 --draws 200

The random-editor arm in the main study draws a fixed four removals and four additions,
which is the cap, not what the models actually do. Models differ a lot in how much they
propose (three edits for one, six for another), so "random matches the LLM" could just be
"more edits beat better edits". This closes that gap.

For every (instance, model) it reads the model's *realized* edit counts out of
`events.jsonl`, then runs `--draws` independent random editors constrained to exactly
those counts, each through the identical downstream pipeline. The LLM's score is then
placed inside the null distribution its own proposal volume generates:

    percentile   share of random draws the LLM beats, per instance
    p_perm       two-sided permutation p-value on the paired mean difference
    delta        mean LLM F1 minus mean random F1 at matched volume

Writes `<run>/analysis/permutation.csv` (one row per instance x model) and
`<run>/analysis/permutation_draws.csv` (every draw, for the null histogram).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# Each episode is a few small matrix operations, so a threaded BLAS buys nothing and 24
# processes each spawning a full thread pool will thrash a machine into apparent deadlock.
# This has to happen before numpy is imported to take effect.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import multiprocessing as mp  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from causal_discovery.active.levels import LEVELS, build_instance, runtime_seed_for  # noqa: E402
from causal_discovery.active.llm_client import resolve_model, short_model_name  # noqa: E402
from causal_discovery.active.probe import ProposalCache, run_probe_episode  # noqa: E402

ROW_COLUMNS = [
    "level", "seed", "model_tag", "d", "n_obs", "arm",
    "n_remove", "n_add", "llm_f1", "random_mean_f1", "random_sd_f1",
    "random_p05", "random_p95", "percentile", "n_draws",
    "llm_truth_in_h", "random_truth_in_h",
]
DRAW_COLUMNS = ["level", "seed", "model_tag", "draw", "n_remove", "n_add", "f1", "truth_in_h"]


def load_proposals(run_dir: Path, arm: str = "") -> dict[tuple[int, int, str], tuple[list, list]]:
    """Recorded proposals, optionally restricted to one arm.

    Two arms can query the same model about the same instance with different prompts, so
    the arm is part of the identity of a proposal, not an incidental label.
    """
    out: dict[tuple[int, int, str], tuple[list, list]] = {}
    events = run_dir / "events.jsonl"
    if not events.exists():
        return out
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
        level_tag, seed_tag, event_arm, model = key.split("|")
        if arm and event_arm != arm:
            continue
        body = payload.get("payload") or {}
        out[(int(level_tag[1:]), int(seed_tag[1:]), short_model_name(resolve_model(model)))] = (
            body.get("remove") or [], body.get("add") or []
        )
    return out


def one_cell(task: dict) -> dict:
    """The LLM episode plus `draws` count-matched random episodes, same instance."""
    level, seed, model = task["level"], task["seed"], task["model"]
    args = task["args"]
    slack = None if args["budget_slack"] < 0 else args["budget_slack"]
    instance = build_instance(LEVELS[level], seed, args["n_obs"], args["n_int"], slack)
    runtime_seed = runtime_seed_for(level, seed)

    shared = dict(
        instance=instance, client=None, runtime_seed=runtime_seed, work_key="perm",
        alpha=args["alpha"], num_candidates=args["num_candidates"],
        propose_rounds=args["propose_rounds"], max_hypotheses=args["max_hypotheses"],
        eig_outcomes=args["eig_outcomes"], skeleton_hint=not args["no_skeleton_hint"],
        max_skeleton_edits=args["max_skeleton_edits"],
        max_skeleton_variants=args["max_skeleton_variants"],
        max_dags_per_skeleton=args["max_dags_per_skeleton"],
        reserve_frac=task["reserve_frac"], select_rule="eig",
        use_bic=True, use_update=True, submit_mode="map",
    )

    cache = ProposalCache()
    cache.prime(task["remove"], task["add"])
    llm = run_probe_episode(hypothesis_source="llm_repair", proposal_cache=cache, **shared)
    n_remove = int(llm.metrics.get("repair_remove", 0))
    n_add = int(llm.metrics.get("repair_add", 0))

    draws = []
    for k in range(task["draws"]):
        # a fresh stream per draw, but the same instance and the same runtime seed, so the
        # only thing that varies across draws is which edits were proposed
        r = run_probe_episode(
            hypothesis_source="noise_repair",
            noise_edits_remove=n_remove, noise_edits_add=n_add,
            **{**shared, "runtime_seed": runtime_seed + 1_000_003 * (k + 1)},
        )
        draws.append((float(r.metrics["directed_f1"]), int(r.metrics.get("truth_in_hypotheses", 0))))

    f1s = np.array([d[0] for d in draws], dtype=float)
    tih = np.array([d[1] for d in draws], dtype=float)
    llm_f1 = float(llm.metrics["directed_f1"])
    # ties count half, so the percentile is unbiased under exchangeability
    pct = float((np.sum(f1s < llm_f1) + 0.5 * np.sum(f1s == llm_f1)) / max(len(f1s), 1))
    return {
        "row": {
            "level": level, "seed": seed, "model_tag": model, "d": LEVELS[level].d,
            "n_obs": args["n_obs"], "arm": task["arm"],
            "n_remove": n_remove, "n_add": n_add,
            "llm_f1": round(llm_f1, 6),
            "random_mean_f1": round(float(f1s.mean()), 6),
            "random_sd_f1": round(float(f1s.std(ddof=1)) if len(f1s) > 1 else 0.0, 6),
            "random_p05": round(float(np.percentile(f1s, 5)), 6),
            "random_p95": round(float(np.percentile(f1s, 95)), 6),
            "percentile": round(pct, 6), "n_draws": len(f1s),
            "llm_truth_in_h": int(llm.metrics.get("truth_in_hypotheses", 0)),
            "random_truth_in_h": round(float(tih.mean()), 6),
        },
        "draws": [
            {"level": level, "seed": seed, "model_tag": model, "draw": k,
             "n_remove": n_remove, "n_add": n_add, "f1": round(f, 6), "truth_in_h": int(t)}
            for k, (f, t) in enumerate(draws)
        ],
    }


def summarise(rows: list[dict]) -> None:
    from scipy.stats import wilcoxon

    print(f"\n  {'model':24s} {'n':>4s} {'LLM':>7s} {'rand':>7s} {'delta':>8s} "
          f"{'pctile':>7s} {'p_pctl':>8s} {'p_wilc':>8s} {'W/L/T':>12s}")
    print("  " + "-" * 96)
    for model in sorted({r["model_tag"] for r in rows}):
        sub = [r for r in rows if r["model_tag"] == model]
        llm = np.array([r["llm_f1"] for r in sub])
        rnd = np.array([r["random_mean_f1"] for r in sub])
        pct = np.array([r["percentile"] for r in sub])
        diff = llm - rnd
        # under the null each instance's percentile is uniform, so its mean is 1/2
        centred = pct - 0.5
        se = centred.std(ddof=1) / np.sqrt(len(centred)) if len(centred) > 1 else 0.0
        z = centred.mean() / se if se > 0 else 0.0
        from math import erf, sqrt
        p_pct = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
        nz = diff[np.abs(diff) > 1e-12]
        p_w = float(wilcoxon(nz).pvalue) if len(nz) >= 6 else float("nan")
        wins = int(np.sum(diff > 1e-12)); losses = int(np.sum(diff < -1e-12))
        ties = len(diff) - wins - losses
        print(f"  {model:24s} {len(sub):4d} {llm.mean():7.3f} {rnd.mean():7.3f} "
              f"{diff.mean():+8.4f} {pct.mean():7.3f} {p_pct:8.4f} {p_w:8.4f} "
              f"{wins:3d}/{losses:3d}/{ties:3d}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8,
                    help="1 runs everything in this process, which is the way to see a "
                         "traceback if the pool misbehaves")
    ap.add_argument("--arm", default="probe",
                    help="which arm's proposals to replay; also the label recorded on each row")
    ap.add_argument("--reserve-frac", type=float, default=-1.0,
                    help="default: take it from the run manifest")
    ap.add_argument("--models", default="", help="comma-separated short tags; default all")
    ap.add_argument("--max-skeleton-variants", type=int, default=-1,
                    help="override the manifest, e.g. to reproduce a re-run that changed it")
    args = ap.parse_args()

    for run in args.run_dirs:
        run_dir = Path(run)
        # the paper's numbers come from the re-run with the corrected variant ordering, so
        # prefer its manifest when it exists; otherwise this run's own
        mpath = run_dir.parent / f"{run_dir.name}_fix" / "run_manifest.json"
        if not mpath.exists():
            mpath = run_dir / "run_manifest.json"
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        margs = dict(manifest["args"])
        own = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))["args"]
        # instance shape always comes from the run the proposals were recorded against
        for k in ("n_obs", "n_int", "alpha", "budget_slack"):
            margs[k] = own[k]
        reserve = float(margs.get("reserve_frac", 0.5)) if args.reserve_frac < 0 else args.reserve_frac
        margs = dict(margs)
        if args.max_skeleton_variants > 0:
            margs["max_skeleton_variants"] = args.max_skeleton_variants
        proposals = load_proposals(run_dir, arm=args.arm)
        if not proposals:  # older runs recorded only one proposing arm
            proposals = load_proposals(run_dir)
        keep = {m.strip() for m in args.models.split(",") if m.strip()}
        tasks = [
            {"level": lvl, "seed": seed, "model": model, "remove": rem, "add": add,
             "args": margs, "draws": args.draws, "arm": args.arm, "reserve_frac": reserve}
            for (lvl, seed, model), (rem, add) in sorted(proposals.items())
            if not keep or model in keep
        ]
        if not tasks:
            print(f"[skip] {run_dir}: no recorded proposals")
            continue
        print(f"[{run_dir.name}] {len(tasks)} cells x {args.draws} draws "
              f"= {len(tasks) * (args.draws + 1):,} episodes, reserve_frac={reserve}", flush=True)

        rows, draw_rows = [], []
        started = time.time()

        def absorb(out: dict) -> None:
            rows.append(out["row"])
            draw_rows.extend(out["draws"])

        def report(i: int) -> None:
            done = time.time() - started
            rate = i / done if done > 0 else 0.0
            eta = (len(tasks) - i) / rate if rate > 0 else 0.0
            print(f"  {i}/{len(tasks)} cells  {done/60:5.1f} min elapsed, "
                  f"~{eta/60:5.1f} min left", flush=True)

        if args.workers <= 1:
            for i, t in enumerate(tasks, 1):
                try:
                    absorb(one_cell(t))
                except Exception as exc:  # noqa: BLE001
                    print(f"  [fail] L{t['level']} s{t['seed']} {t['model']}: "
                          f"{type(exc).__name__}: {exc}", flush=True)
                if i % 5 == 0 or i == len(tasks):
                    report(i)
        else:
            # `spawn`, not the Linux default `fork`: this module imports numpy and
            # causallearn at load time, and forking a process that already holds their
            # thread state deadlocks the children on some builds.
            ctx = mp.get_context("spawn")
            print(f"  starting {args.workers} workers (spawn)...", flush=True)
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
                futures = {pool.submit(one_cell, t): t for t in tasks}
                for i, fut in enumerate(as_completed(futures), 1):
                    try:
                        absorb(fut.result())
                    except Exception as exc:  # noqa: BLE001
                        t = futures[fut]
                        print(f"  [fail] L{t['level']} s{t['seed']} {t['model']}: "
                              f"{type(exc).__name__}: {exc}", flush=True)
                    if i <= 3 or i % 20 == 0 or i == len(tasks):
                        report(i)

        out_dir = run_dir / "analysis"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows.sort(key=lambda r: (r["model_tag"], r["level"], r["seed"]))
        with (out_dir / "permutation.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=ROW_COLUMNS); w.writeheader(); w.writerows(rows)
        with (out_dir / "permutation_draws.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=DRAW_COLUMNS); w.writeheader(); w.writerows(draw_rows)

        summarise(rows)
        print(f"\n  -> {out_dir/'permutation.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
