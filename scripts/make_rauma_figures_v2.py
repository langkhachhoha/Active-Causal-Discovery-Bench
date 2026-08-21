#!/usr/bin/env python
"""New RauMa figures drawn from the full Study-1 run tree (`study1/`).

    python scripts/make_rauma_figures_v2.py --study-dir study1 --out-dir figures

Written figures
    rauma_f3_readout    what the two readouts do to individual arrows
    rauma_f4_robust     the readout gap against sample size and evidence format
    rauma_f5_selection  per-round choice quality vs. downstream effect

Style matches scripts/make_rauma_figures.py. Nothing here calls a model; the edge
audit is read from figures/edge_audit.json (see scripts/rauma_edge_audit.py).
"""
from __future__ import annotations

import argparse, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#c8352b"
GREY, DARK, LIGHT = "#6f6e6a", "#1a1a1a", "#b9b8b2"
GRID, AXIS = "#e3e2dc", "#bfbeb4"
PALE_BLUE, PALE_ORANGE = "#8ebbee", "#f5b294"

QWEN = "qwen3-coder-30b-a3b-instruct"
GPT = "gpt-4o-mini-2024-07-18"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.7, "axes.labelcolor": DARK,
    "text.color": DARK, "xtick.color": GREY, "ytick.color": GREY,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.4, "lines.markersize": 4,
    "errorbar.capsize": 2,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

TEXT = {"run_id", "timestamp_utc", "study", "arm", "selector", "inferencer",
        "model", "model_tag", "status", "error", "infer_rule"}
MECH = ["random+meek", "maxdeg+meek", "eig+meek", "oracle+meek", "llm+meek"]
LLMR = ["random+llm", "maxdeg+llm", "eig+llm", "oracle+llm", "llm+llm"]


def save(fig, out_dir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  [written] {out_dir}/{name}.pdf + .png")


def load(study_dir, run, levels=None):
    df = pd.read_csv(os.path.join(study_dir, run, "episodes.csv"))
    for c in df.columns:
        if c not in TEXT:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df.status == "success"].copy()
    if levels is not None:
        df = df[df.level.isin(levels)].copy()
    df["inst"] = df.level.astype(str) + ":" + df.seed.astype(str)
    return df


def ci95(v):
    v = np.asarray(v, dtype=float)
    return 0.0 if len(v) < 2 else float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))


def cell(df, arm, tag, metric="directed_f1"):
    return df[(df.arm == arm) & (df.model_tag == tag)].set_index("inst")[metric].sort_index()


def contrast(a, b):
    k = a.index.intersection(b.index)
    d = (a.loc[k] - b.loc[k]).to_numpy(dtype=float)
    p = 1.0 if np.allclose(d, 0) else float(wilcoxon(d, zero_method="wilcox").pvalue)
    return {"delta": float(d.mean()), "ci": ci95(d), "p": p, "n": len(d),
            "wtl": f"{int((d > 1e-12).sum())}-{int((abs(d) <= 1e-12).sum())}-{int((d < -1e-12).sum())}"}


def marginal(df, tag, arms):
    sub = df[df.arm.isin(arms) & ((df.model_tag == "none") | (df.model_tag == tag))]
    return sub.groupby("inst").directed_f1.mean().sort_index()


# ------------------------------------------------- F3  anatomy of the readout
def fig_readout(audit, out_dir):
    llm, mech = audit["main_llm"], audit["main_mech"]
    rows = [("mean-shift + Meek", mech["oracle+meek|none"]),
            ("LLM readout · Qwen3-Coder-30B", llm[f"oracle+llm|{QWEN}"]),
            ("LLM readout · GPT-4o-mini", llm[f"oracle+llm|{GPT}"])]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 1.72),
                                   gridspec_kw={"width_ratios": [1.25, 1.0]})

    # (a) where every submitted arrow ends up
    ypos = np.arange(len(rows))[::-1]
    segs = [("points the right way", "correct", BLUE),
            ("points backwards", "reversed", RED),
            ("not a real adjacency", "spurious", LIGHT)]
    for y, (label, c) in zip(ypos, rows):
        left, total = 0.0, c["total"]
        for slabel, key, colour in segs:
            w = c[key] / total
            ax1.barh(y, w, 0.52, left=left, color=colour, edgecolor="white", linewidth=0.5,
                     label=slabel if y == ypos[0] else None)
            if w > 0.05:
                ax1.text(left + w / 2, y, f"{w:.0%}", ha="center", va="center",
                         fontsize=6.3, color="white" if colour != LIGHT else DARK)
            left += w
        ax1.text(1.012, y, f"{int(total)} arrows", va="center", ha="left", fontsize=6.0,
                 color=GREY)
    ax1.set_yticks(ypos)
    ax1.set_yticklabels([r[0] for r in rows], fontsize=6.5)
    ax1.set_xlim(0, 1.0); ax1.set_ylim(-0.65, len(rows) - 0.35)
    ax1.set_xticks(np.arange(0, 1.01, 0.25))
    ax1.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax1.set_xlabel("share of the arrows the readout submitted")
    ax1.xaxis.grid(True); ax1.set_axisbelow(True)
    # Keep panel titles short: the full experimental condition is already stated in
    # the figure caption, and a long title collides with panel (b) at journal width.
    ax1.set_title("(a) submitted arrows by outcome", loc="left", fontsize=7.0)
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.16), ncol=3, handlelength=1.0,
               handletextpad=0.4, columnspacing=1.0, fontsize=6.3)

    # (b) split by who had to decide the direction
    groups = [("already fixed by\nobservation", "pc"),
              ("left to the\nexperiment", "int")]
    width = 0.24
    for ai, (label, c) in enumerate(rows):
        colour = [BLUE, ORANGE, PALE_ORANGE][ai]
        for gi, (_, key) in enumerate(groups):
            r = c[key + "_rev"] / c[key + "_n"]
            ax2.bar(gi + (ai - 1) * width, r, width * 0.9, color=colour,
                    edgecolor="white", linewidth=0.4)
            # Values close to the 50% reference line are placed inside the bar so
            # that neither the label nor its glyphs are crossed by the dashed line.
            near_reference = abs(r - 0.5) < 0.04
            label_y = r - 0.014 if near_reference else r + 0.012
            ax2.text(gi + (ai - 1) * width, label_y, f"{r:.0%}", ha="center",
                     va="top" if near_reference else "bottom", fontsize=6.0, color=DARK)
    ax2.axhline(0.5, color=RED, lw=0.8, ls=(0, (4, 2)))
    ax2.text(-0.42, 0.512, "coin flip", ha="left", va="bottom", fontsize=6.2, color=RED)
    ax2.set_xticks(range(len(groups)))
    ax2.set_xticklabels([g[0] for g in groups], fontsize=6.3)
    ax2.set_ylim(0, 0.66)
    ax2.set_yticks([0, 0.2, 0.4, 0.6])
    ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax2.set_ylabel("arrows pointing backwards", labelpad=1.5)
    ax2.set_xlim(-0.5, 1.5)
    ax2.yaxis.grid(True); ax2.set_axisbelow(True)
    ax2.set_title("(b) reversal rate by decision source", loc="left", fontsize=7.0)

    fig.subplots_adjust(wspace=0.52)
    save(fig, out_dir, "rauma_f3_readout")


# ------------------------------------ F4  robustness: sample size and evidence
def fig_robust(runs, audit, out_dir):
    n60, n300, n1k, raw = runs["n60"], runs["main12"], runs["n1000"], runs["raw"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 1.72),
                                   gridspec_kw={"width_ratios": [1.0, 1.0]})

    # (a) directed F1 against observational sample size
    xs = [60, 300, 1000]
    dfs = [n60, n300, n1k]
    lines = [("mean-shift + Meek readout", MECH, BLUE, "o", "-"),
             ("LLM readout · Qwen", LLMR, ORANGE, "s", "-"),
             ("LLM readout · GPT", LLMR, PALE_ORANGE, "^", "-")]
    curves = {}
    for label, arms, colour, marker, ls in lines:
        tag = GPT if "GPT" in label else QWEN
        ys, es = [], []
        for df in dfs:
            v = marginal(df, tag, arms)
            ys.append(v.mean()); es.append(ci95(v))
        curves[label] = ys
        ax1.errorbar(xs, ys, yerr=es, color=colour, marker=marker, ls=ls, lw=1.4,
                     elinewidth=0.8, capsize=1.8, label=label, zorder=3)
    ax1.fill_between(xs, curves["mean-shift + Meek readout"],
                     curves["LLM readout · Qwen"], color=BLUE, alpha=0.07, zorder=1)
    for x, a, b in zip(xs, curves["mean-shift + Meek readout"], curves["LLM readout · Qwen"]):
        ax1.annotate("", xy=(x, a), xytext=(x, b),
                     arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6, ls=(0, (1, 1.5))))
        ax1.text(x * 1.06, (a + b) / 2, f"{a - b:.2f}", fontsize=6.0, color=GREY,
                 ha="left", va="center")
    ax1.set_xscale("log")
    ax1.set_xticks(xs); ax1.set_xticklabels([str(x) for x in xs])
    ax1.set_xlim(45, 1700)
    ax1.set_ylim(0.35, 1.0)
    ax1.set_xlabel("observational sample size $n_{\\mathrm{obs}}$ (log scale)")
    ax1.set_ylabel("directed-edge F1")
    ax1.grid(True, lw=0.6); ax1.set_axisbelow(True)
    ax1.set_title("(a) better data lifts the rule, not the reader", loc="left", fontsize=7.0)
    ax1.legend(loc="lower right", handlelength=1.4, labelspacing=0.25, fontsize=6.2)

    # (b) sufficient statistics vs. raw interventional rows
    pairs = [("Qwen", QWEN, BLUE), ("GPT", GPT, PALE_BLUE)]
    width = 0.3
    for mi, (name, tag, colour) in enumerate(pairs):
        for fi, (flabel, df, hatch) in enumerate([("summary", n300, None), ("raw rows", raw, "///")]):
            v = cell(df, "oracle+llm", tag)
            tok = df[(df.arm == "oracle+llm") & (df.model_tag == tag)].total_tokens.mean()
            ax2.bar(mi + (fi - 0.5) * width, v.mean(), width * 0.86,
                    color=ORANGE if fi else PALE_ORANGE, edgecolor="white", linewidth=0.4,
                    hatch=hatch, label=flabel if mi == 0 else None)
            ax2.errorbar(mi + (fi - 0.5) * width, v.mean(), yerr=ci95(v), fmt="none",
                         ecolor=DARK, elinewidth=0.7, capsize=1.6)
            ax2.text(mi + (fi - 0.5) * width, 0.028, f"{tok/1000:.1f}k tok",
                     ha="center", va="bottom", fontsize=5.8, color="white", rotation=90)
    ref = cell(n300, "oracle+meek", "none")
    ax2.axhline(ref.mean(), color=BLUE, lw=0.9, ls=(0, (4, 2)))
    ax2.text(1.48, ref.mean() + 0.018, f"mean-shift + Meek ({ref.mean():.2f})",
             ha="right", va="bottom", fontsize=6.2, color=BLUE)
    ax2.set_xticks(range(len(pairs)))
    ax2.set_xticklabels([p[0] for p in pairs], fontsize=6.8)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("directed-edge F1")
    ax2.set_xlabel("LLM readout, evidence it was given")
    ax2.yaxis.grid(True); ax2.set_axisbelow(True)
    ax2.set_title("(b) raw data does not rescue the reader", loc="left", fontsize=7.0)
    ax2.legend(loc="upper left", bbox_to_anchor=(0.02, 0.99), handlelength=1.0,
               labelspacing=0.2, fontsize=6.3)

    fig.subplots_adjust(wspace=0.34)
    save(fig, out_dir, "rauma_f4_robust")


# --------------------------------------------------------------- F5 selection
SEL_ROWS = [("random", "random+meek", "none"),
            ("LLM · GPT", "llm+meek", GPT),
            ("LLM · Qwen", "llm+meek", QWEN),
            ("max-degree", "maxdeg+meek", "none"),
            ("MEC-EIG (exact)", "eig+meek", "none")]


def fig_selection(main_steps, runs, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.72),
                             gridspec_kw={"width_ratios": [1.0, 0.85, 1.05]})
    ypos = np.arange(len(SEL_ROWS))[::-1]

    # (a) per-round share of EIG-optimal targets
    for y, (_, arm, tag) in zip(ypos, SEL_ROWS):
        s = main_steps[(main_steps.arm == arm) & (main_steps.model_tag == tag)]
        frac = float((s.eig_regret_nats.abs() < 1e-9).mean())
        colour = BLUE if "llm" in arm else GREY
        axes[0].barh(y, frac, 0.5, color=colour, edgecolor="white", linewidth=0.4)
        axes[0].text(frac + 0.015, y, f"{frac:.0%}", va="center", ha="left", fontsize=6.2,
                     color=DARK)
    axes[0].set_xlim(0, 1.18)
    axes[0].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[0].xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axes[0].set_xlabel("rounds that picked an\nEIG-optimal target")
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([r[0] for r in SEL_ROWS], fontsize=6.6)

    # (b) the downstream score, unpaired CIs
    for y, (_, arm, tag) in zip(ypos, SEL_ROWS):
        v = cell(runs["main"], arm, tag)
        axes[1].errorbar(v.mean(), y, xerr=ci95(v), fmt="o", color=BLUE if "llm" in arm else GREY,
                         ecolor=BLUE if "llm" in arm else GREY, elinewidth=0.9, capsize=1.8,
                         markersize=3.4)
    axes[1].set_xlim(0.79, 0.92)
    axes[1].set_xticks([0.80, 0.85, 0.90])
    axes[1].set_xlabel("directed-edge F1\n(main budget)")

    # (c) paired gain over random across four evidence regimes
    regimes = [("main", "main", "o", BLUE),
               ("tight budget", "tight", "s", ORANGE),
               ("$n_{\\mathrm{obs}}{=}60$", "n60", "^", RED),
               ("$n_{\\mathrm{obs}}{=}1000$", "n1000", "v", GREY)]
    rpos = np.arange(len(regimes))[::-1]
    for y, (label, key, marker, colour) in zip(rpos, regimes):
        df = runs[key]
        for dx, (arm, tag, face) in enumerate([("llm+meek", QWEN, colour), ("eig+meek", "none", "white")]):
            c = contrast(cell(df, arm, tag), cell(df, "random+meek", "none"))
            axes[2].errorbar(c["delta"], y + (0.17 if dx == 0 else -0.17), xerr=c["ci"],
                             fmt=marker, color=colour, markerfacecolor=face,
                             ecolor=colour, elinewidth=0.9, capsize=1.8, markersize=3.4,
                             markeredgewidth=0.8)
    axes[2].axvline(0, color=DARK, lw=0.7)
    axes[2].set_yticks(rpos)
    axes[2].yaxis.tick_right()
    axes[2].set_yticklabels([r[0] for r in regimes], fontsize=6.4)
    axes[2].tick_params(axis="y", length=0, pad=2)
    axes[2].set_ylim(-0.65, len(regimes) - 0.35)
    axes[2].set_xlim(-0.03, 0.13)
    axes[2].set_xticks([0.0, 0.05, 0.10])
    axes[2].set_xlabel("paired $\\Delta$ F1 over random\n(filled: RauMa,  open: MEC-EIG)")

    for ax in axes:
        ax.xaxis.grid(True); ax.set_axisbelow(True)
    for ax in axes[:2]:
        ax.set_ylim(-0.65, len(SEL_ROWS) - 0.35)
    axes[1].set_yticks(ypos); axes[1].set_yticklabels([])
    for ax, t in zip(axes, ("(a) choice quality, per round",
                            "(b) what it buys",
                            "(c) and when it starts to matter")):
        ax.set_title(t, loc="left", fontsize=6.9)
    fig.subplots_adjust(wspace=0.20)
    save(fig, out_dir, "rauma_f5_selection")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-dir", default="study1")
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--audit", default="figures/edge_audit.json")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    runs = {
        "main": load(args.study_dir, "main"),
        "main12": load(args.study_dir, "main", levels=[1, 2]),
        "tight": load(args.study_dir, "ablation_tightbudget"),
        "n60": load(args.study_dir, "ablation_n60"),
        "n1000": load(args.study_dir, "ablation_n1000"),
        "raw": load(args.study_dir, "ablation_rawevidence"),
    }
    steps = pd.read_csv(os.path.join(args.study_dir, "main", "steps.csv"))
    audit = json.load(open(args.audit))
    print({k: len(v) for k, v in runs.items()})

    fig_readout(audit, args.out_dir)
    fig_robust(runs, audit, args.out_dir)
    fig_selection(steps, runs, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
