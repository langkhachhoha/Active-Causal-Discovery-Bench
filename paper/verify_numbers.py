"""Recompute every number quoted in `rauma_neurips2026.tex` from the committed logs.

    python paper/verify_numbers.py [--study-dir study1]

Reads the five Study-1 run directories (`main`, `ablation_tightbudget`, `ablation_n60`,
`ablation_n1000`, `ablation_rawevidence`) and prints each paper claim next to the value
recomputed from the logs. Nothing here touches the network or an LLM. The edge audit of
Figure 3 and Table 6 needs the true DAGs and the PC front-end, which are deterministic
given the seed; it is produced by `scripts/rauma_edge_audit.py` and read back here from
`figures/edge_audit.json` if that file is present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

ROOT = Path(__file__).resolve().parent.parent
STUDY_DIR = ROOT / "study1"
QWEN = "qwen3-coder-30b-a3b-instruct"
GPT = "gpt-4o-mini-2024-07-18"
TEXT = {"run_id", "timestamp_utc", "study", "arm", "selector", "inferencer",
        "model", "model_tag", "status", "error", "infer_rule"}


def load(run: str, levels: tuple[int, ...] | None = None) -> pd.DataFrame:
    """Episode rows for one run directory, numeric-coerced and filtered to successes."""
    df = pd.read_csv(STUDY_DIR / run / "episodes.csv")
    for column in df.columns:
        if column not in TEXT:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df[df.status == "success"].copy()
    if levels is not None:
        df = df[df.level.isin(levels)].copy()
    return df


MECH_ARMS = ["random+meek", "maxdeg+meek", "eig+meek", "oracle+meek", "llm+meek"]
LLM_ARMS = ["random+llm", "maxdeg+llm", "eig+llm", "oracle+llm", "llm+llm"]


def marginal_series(df, arms, tag):
    """Per-instance mean over the five selectors, holding the readout fixed."""
    sub = df[df.arm.isin(arms) & ((df.model_tag == "none") | (df.model_tag == tag))]
    counts = sub.groupby(["level", "seed"]).directed_f1.count()
    assert (counts == len(arms)).all(), counts.value_counts().to_dict()
    return sub.groupby(["level", "seed"]).directed_f1.mean().sort_index()


def series(df: pd.DataFrame, arm: str, tag: str, metric: str) -> dict:
    sub = df[(df.arm == arm) & (df.model_tag == tag)]
    return sub.set_index(["level", "seed"])[metric].to_dict()


def paired(df, arm_a, tag_a, arm_b, tag_b, metric="directed_f1") -> dict:
    a, b = series(df, arm_a, tag_a, metric), series(df, arm_b, tag_b, metric)
    keys = sorted(set(a) & set(b))
    va = np.array([a[k] for k in keys])
    vb = np.array([b[k] for k in keys])
    delta = va - vb
    win = int((delta > 1e-12).sum())
    tie = int((np.abs(delta) <= 1e-12).sum())
    loss = int((delta < -1e-12).sum())
    p = 1.0 if np.allclose(delta, 0) else float(wilcoxon(va, vb, zero_method="wilcox").pvalue)
    return {"n": len(keys), "a": va.mean(), "b": vb.mean(), "delta": delta.mean(),
            "p": p, "wtl": f"{win}-{tie}-{loss}"}


def line(claim: str, r: dict) -> None:
    print(f"  {claim:52s} delta={r['delta']:+.4f}  p={r['p']:.2g}  W-T-L={r['wtl']}  n={r['n']}")


def mean_ci(df, arm, tag, metric="directed_f1") -> tuple[float, float]:
    v = df[(df.arm == arm) & (df.model_tag == tag)][metric]
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(v.count()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-dir", default=str(STUDY_DIR))
    args = ap.parse_args()
    globals()["STUDY_DIR"] = Path(args.study_dir)

    main_run = load("main")
    tight = load("ablation_tightbudget")
    n60 = load("ablation_n60")
    n1000 = load("ablation_n1000")
    raw = load("ablation_rawevidence")
    main12 = load("main", levels=(1, 2))
    runs = {"main": main_run, "tight": tight, "n60": n60, "n1000": n1000}

    total = sum(len(d) for d in (main_run, tight, n60, n1000, raw))
    print(f"episodes: main={len(main_run)} tight={len(tight)} n60={len(n60)} "
          f"n1000={len(n1000)} raw={len(raw)} total={total} (paper says 720/240/360/360/120 "
          f"= 1,800), failed LLM calls="
          f"{int(pd.concat([main_run, tight, n60, n1000, raw]).llm_failed_calls.fillna(0).sum())}")

    print("\n[Table 2 / abstract] arm means, directed F1")
    for arm, tag in [("oracle+meek", "none"), ("random+meek", "none"), ("maxdeg+meek", "none"),
                     ("eig+meek", "none"), ("llm+meek", QWEN), ("llm+meek", GPT),
                     ("oracle+llm", QWEN), ("oracle+llm", GPT), ("llm+llm", QWEN),
                     ("llm+llm", GPT), ("llm_e2e", QWEN), ("llm_e2e", GPT)]:
        m, c = mean_ci(main_run, arm, tag)
        print(f"  {arm:12s} {tag:29s} {m:.3f} +- {c:.3f}")

    print("\n[Sec 5.1] readout marginal effect (mean over the five selectors, per instance)")
    mech, llm_readout = MECH_ARMS, LLM_ARMS
    for tag in (QWEN, GPT):
        a = marginal_series(main_run, mech, tag)
        b = marginal_series(main_run, llm_readout, tag)
        delta = (a - b).to_numpy()
        p = float(wilcoxon(delta, zero_method="wilcox").pvalue)
        win = int((delta > 1e-12).sum())
        tie = int((abs(delta) <= 1e-12).sum())
        print(f"  {tag:29s} mech={a.mean():.4f} llm={b.mean():.4f} "
              f"delta={delta.mean():+.4f} p={p:.2g} W-T-L={win}-{tie}-{len(delta)-win-tie}")

    print("\n[Sec 5.1] one-factor contrasts")
    for tag in (QWEN, GPT):
        print(f" {tag}")
        line("selection: llm+meek - oracle+meek", paired(main_run, "llm+meek", tag, "oracle+meek", "none"))
        line("inference: oracle+llm - oracle+meek", paired(main_run, "oracle+llm", tag, "oracle+meek", "none"))
        line("restore updater: llm+meek - llm+llm", paired(main_run, "llm+meek", tag, "llm+llm", tag))
        line("scaffold: llm+meek - llm_e2e", paired(main_run, "llm+meek", tag, "llm_e2e", tag))
        line("selector: llm+meek - eig+meek", paired(main_run, "llm+meek", tag, "eig+meek", "none"))

    print("\n[Sec 5.2] LLM inferencer keeps skeleton/compelled edges? (oracle selection)")
    for tag in (QWEN, GPT):
        print(f" {tag}")
        line("compelled: oracle+llm - oracle+meek",
             paired(main_run, "oracle+llm", tag, "oracle+meek", "none", "compelled_f1"))
        line("skeleton:  oracle+llm - oracle+meek",
             paired(main_run, "oracle+llm", tag, "oracle+meek", "none", "skeleton_f1"))
    cols = ["directed_precision", "directed_recall", "submit_directed", "submit_undirected"]
    print(main_run[main_run.arm.isin(["oracle+meek", "oracle+llm"])]
          .groupby(["arm", "model_tag"])[cols].mean().round(3).to_string())

    print("\n[Table 4] selection quality, inference held at meek")
    sel = main_run[main_run.inferencer == "meek"]
    stats = sel.groupby(["selector", "model_tag"]).agg(
        directed_f1=("directed_f1", "mean"),
        regret=("selection_regret_total", "mean"),
        quality=("selection_quality_mean", "mean"),
        eig_regret=("eig_regret_total", "mean"),
        zero_regret=("selection_regret_total", lambda s: (s == 0).mean()),
        interventions=("interventions_used", "mean"),
    ).round(3)
    print(stats.to_string())

    print("\n[Sec 5.4 / Table 5] tight budget")
    line("tight: llm(qwen) - random", paired(tight, "llm+meek", QWEN, "random+meek", "none"))
    line("tight: eig - random", paired(tight, "eig+meek", "none", "random+meek", "none"))
    line("tight: maxdeg - random", paired(tight, "maxdeg+meek", "none", "random+meek", "none"))
    line("tight: llm(qwen) - eig", paired(tight, "llm+meek", QWEN, "eig+meek", "none"))
    line("main : llm(qwen) - random", paired(main_run, "llm+meek", QWEN, "random+meek", "none"))
    line("main : eig - random", paired(main_run, "eig+meek", "none", "random+meek", "none"))

    print("\n[Sec 5.5 / Table 6] cost per episode")
    money = ["prompt_tokens", "completion_tokens", "total_tokens", "cost_usd",
             "llm_calls", "llm_repair_calls", "llm_failed_calls"]
    print(main_run[main_run.model_tag != "none"].groupby(["arm", "model_tag"])[money]
          .mean().round(5).to_string())
    for tag in (QWEN, GPT):
        ratio = (main_run[(main_run.arm == "llm_e2e") & (main_run.model_tag == tag)].total_tokens.mean()
                 / main_run[(main_run.arm == "llm+meek") & (main_run.model_tag == tag)].total_tokens.mean())
        print(f"  token ratio no-scaffold / RauMa ({tag}): {ratio:.1f}x")
    print(f"  main run: {int(main_run.total_tokens.sum()):,} tokens, "
          f"${main_run.cost_usd.sum():.2f}; tight: ${tight.cost_usd.sum():.3f}")
    e2e_gpt = main_run[(main_run.arm == "llm_e2e") & (main_run.model_tag == GPT)]
    print(f"  gpt-4o-mini unscaffolded: {int((e2e_gpt.submit_directed == 0).sum())}/{len(e2e_gpt)} "
          f"episodes submit zero directed edges, {e2e_gpt.submit_undirected.mean():.2f} undirected")

    print("\n[Sec 3.3 / Appendix B] mean-shift orientation errors")
    meek = main_run[main_run.inferencer == "meek"]
    by_level = meek.groupby("level")[["orientations_correct", "orientations_wrong"]].sum()
    pooled = by_level.orientations_wrong.sum() / by_level.sum().sum()
    print(f"  pooled wrong rate {pooled:.3%} over {int(by_level.sum().sum())} orientations")
    print((by_level.orientations_wrong / by_level.sum(axis=1)).round(4).to_string())

    print("\n[Table 1] instance ladder + front-end quality")
    front = main_run[main_run.arm == "oracle+meek"]
    print(front.groupby("level")[["d", "k", "opt_set_size", "budget", "pc_undirected_edges",
                                  "mec_size_initial", "pc_skeleton_f1_ceiling",
                                  "pc_truth_in_class"]].mean().round(3).to_string())
    print(f"  max MEC size seen: {int(front.mec_size_initial.max())} (implementation cap 256)")

    print("\n[Appendix A] directed F1 by level")
    keep = ["oracle+meek", "random+meek", "eig+meek", "llm+meek", "oracle+llm", "llm+llm", "llm_e2e"]
    print(main_run[main_run.arm.isin(keep)]
          .pivot_table(index=["arm", "model_tag"], columns="level",
                       values="directed_f1", aggfunc="mean").round(3).to_string())

    print("\n[Sec 5.3] selection wall-clock per episode (seconds)")
    print(main_run[main_run.inferencer == "meek"]
          .groupby(["arm", "model_tag", "level"]).wall_sec.mean().round(3).to_string())

    print("\n[Sec 5.4 / Table 8] per-round choice quality (main, readout mechanical)")
    steps = pd.read_csv(STUDY_DIR / "main" / "steps.csv")
    steps = steps[steps.arm.str.endswith("+meek")]
    for arm, tag in [("random+meek", "none"), ("maxdeg+meek", "none"), ("eig+meek", "none"),
                     ("llm+meek", QWEN), ("llm+meek", GPT), ("oracle+meek", "none")]:
        x = steps[(steps.arm == arm) & (steps.model_tag == tag)]
        print(f"  {arm:12s} {tag:29s} rounds={len(x):3d} "
              f"EIG-optimal={(x.eig_regret_nats.abs() < 1e-9).mean():6.1%} "
              f"truth-gain-optimal={(x.selection_regret.abs() < 1e-9).mean():6.1%}")

    print("\n[Table 9] selector contrasts in four evidence regimes")
    for name, df in runs.items():
        for label, arm, tag in [("RauMa(qwen)-random", "llm+meek", QWEN),
                                ("eig-random", "eig+meek", "none"),
                                ("maxdeg-random", "maxdeg+meek", "none")]:
            line(f"{name:7s} {label}", paired(df, arm, tag, "random+meek", "none"))
        line(f"{name:7s} RauMa(qwen)-eig", paired(df, "llm+meek", QWEN, "eig+meek", "none"))

    print("\n[Sec 5.3 / Table 7] sample-size sweep, d in {6,8}, marginal over five selectors")
    for label, df in [("n=60", n60), ("n=300 (main, separate draw)", main12), ("n=1000", n1000)]:
        meek = df[df.inferencer == "meek"]
        wrong = meek.orientations_wrong.sum()
        total_or = wrong + meek.orientations_correct.sum()
        for tag in (QWEN, GPT):
            a = marginal_series(df, MECH_ARMS, tag)
            b = marginal_series(df, LLM_ARMS, tag)
            d = (a - b).to_numpy()
            p = float(wilcoxon(d, zero_method="wilcox").pvalue)
            print(f"  {label:28s} {tag[:12]:13s} mech={a.mean():.3f} llm={b.mean():.3f} "
                  f"gap={d.mean():+.3f} p={p:.2g}")
        print(f"  {'':28s} mean-shift wrong {int(wrong)}/{int(total_or)}="
              f"{wrong / total_or:.1%}, PC skeleton F1={meek.skeleton_f1.mean():.3f}, "
              f"RauMa(qwen)={df[(df.arm == 'llm+meek') & (df.model_tag == QWEN)].directed_f1.mean():.3f}")

    print("\n[Sec 5.3 / Table 8] evidence format: summary vs raw rows (unpaired, separate draws)")
    for arm in ("oracle+llm", "eig+llm", "llm+llm"):
        for tag in (QWEN, GPT):
            a = main12[(main12.arm == arm) & (main12.model_tag == tag)].directed_f1
            b = raw[(raw.arm == arm) & (raw.model_tag == tag)].directed_f1
            ta = main12[(main12.arm == arm) & (main12.model_tag == tag)].total_tokens.mean()
            tb = raw[(raw.arm == arm) & (raw.model_tag == tag)].total_tokens.mean()
            print(f"  {arm:11s} {tag[:12]:13s} summary={a.mean():.3f} ({ta:6.0f} tok)  "
                  f"raw={b.mean():.3f} ({tb:6.0f} tok)  delta={b.mean() - a.mean():+.3f}  "
                  f"MWU p={mannwhitneyu(b, a).pvalue:.2g}")
    ref = main12[(main12.arm == "oracle+meek")].directed_f1.mean()
    print(f"  mechanical readout on the same ladder levels: {ref:.3f}")

    audit_path = ROOT / "figures" / "edge_audit.json"
    if audit_path.exists():
        print("\n[Fig 3 / Table 6] edge audit, selection pinned to the truth-aware reference")
        audit = json.load(open(audit_path))
        rows = [("mean-shift+Meek", audit["main_mech"]["oracle+meek|none"]),
                ("LLM qwen3-30b", audit["main_llm"][f"oracle+llm|{QWEN}"]),
                ("LLM gpt-4o-mini", audit["main_llm"][f"oracle+llm|{GPT}"])]
        for label, c in rows:
            t = c["total"]
            print(f"  {label:16s} arrows={t:.0f} correct={c['correct'] / t:.1%} "
                  f"reversed={c['reversed'] / t:.1%} spurious={c['spurious'] / t:.1%} | "
                  f"observational {c['pc_rev']:.0f}/{c['pc_n']:.0f}={c['pc_rev'] / c['pc_n']:.1%} | "
                  f"experimental {c['int_rev']:.0f}/{c['int_n']:.0f}={c['int_rev'] / c['int_n']:.1%}")
        for key, c in sorted(audit["raw_llm"].items()):
            t = c["total"]
            print(f"  raw evidence {key:44s} spurious={c['spurious'] / t:.1%}")
    else:
        print("\n[Fig 3] edge audit missing; run scripts/rauma_edge_audit.py first")

    print("\n[Reproducibility] totals across all five runs")
    allruns = pd.concat([main_run, tight, n60, n1000, raw])
    print(f"  {len(allruns)} episodes, {int(allruns.total_tokens.fillna(0).sum()):,} tokens, "
          f"${allruns.cost_usd.fillna(0).sum():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
