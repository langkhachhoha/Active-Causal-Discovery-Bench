#!/usr/bin/env python
"""Figures for the RauMa paper (Study 1).

    python scripts/make_rauma_figures.py --result-dir result/study1 --out-dir figures

Every panel is drawn from the committed episode CSVs. Error bars are 95% normal-
approximation confidence intervals; for arm means they are taken over the 40 paired
instances, for contrasts over the 40 paired per-instance differences. Nothing here
re-runs an episode or calls a model.

Written figures
    rauma_f2_factorial     main   (a) arm means, (b) paired-contrast forest
    rauma_f3_dissociation  main   selection regret vs directed F1, both budgets
    rauma_f4_scaling       main   directed F1 against graph size, per model
    rauma_a1_cost          appx   tokens per episode against directed F1
    rauma_a2_diagnostic    appx   precision/recall/skeleton/compelled at fixed selection
    rauma_a3_meanshift     appx   mean-shift misorientation rate by graph size
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# --------------------------------------------------------------------- tokens
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#c8352b"
GREY, DARK, LIGHT = "#6f6e6a", "#1a1a1a", "#b9b8b2"
GRID, AXIS = "#e3e2dc", "#bfbeb4"

QWEN = "qwen3-coder-30b-a3b-instruct"
GPT = "gpt-4o-mini-2024-07-18"
QL, GL = "Qwen3-Coder-30B", "GPT-4o-mini"

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


def save(fig, out_dir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  [written] {out_dir}/{name}.pdf + .png")


# ----------------------------------------------------------------------- data
TEXT = {"run_id", "timestamp_utc", "study", "arm", "selector", "inferencer",
        "model", "model_tag", "status", "error", "infer_rule"}


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for column in df.columns:
        if column not in TEXT:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df[df.status == "success"].copy()
    df["inst"] = df.level.astype(str) + ":" + df.seed.astype(str)
    return df


def ci95(values) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    return float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))


def cell(df, arm, tag, metric="directed_f1") -> pd.Series:
    """Per-instance series for one arm, indexed by instance."""
    sub = df[(df.arm == arm) & (df.model_tag == tag)]
    return sub.set_index("inst")[metric].sort_index()


def mean_ci(df, arm, tag, metric="directed_f1") -> tuple[float, float]:
    v = cell(df, arm, tag, metric)
    return float(v.mean()), ci95(v)


def contrast(a: pd.Series, b: pd.Series) -> dict:
    """Paired difference a - b over shared instances."""
    keys = a.index.intersection(b.index)
    d = (a.loc[keys] - b.loc[keys]).to_numpy(dtype=float)
    p = 1.0 if np.allclose(d, 0) else float(wilcoxon(d, zero_method="wilcox").pvalue)
    return {"delta": float(d.mean()), "ci": ci95(d), "p": p, "n": len(d),
            "wtl": f"{int((d > 1e-12).sum())}-{int((abs(d) <= 1e-12).sum())}-{int((d < -1e-12).sum())}"}


MECH_SELECTORS = ["random+meek", "maxdeg+meek", "eig+meek", "oracle+meek", "llm+meek"]
LLM_SELECTORS = ["random+llm", "maxdeg+llm", "eig+llm", "oracle+llm", "llm+llm"]


def readout_marginal(df, tag, arms) -> pd.Series:
    """Per-instance mean over the five selectors, holding the readout fixed."""
    sub = df[df.arm.isin(arms) & ((df.model_tag == "none") | (df.model_tag == tag))]
    counts = sub.groupby("inst").directed_f1.count()
    assert (counts == len(arms)).all(), f"unbalanced marginal: {counts.value_counts().to_dict()}"
    return sub.groupby("inst").directed_f1.mean().sort_index()


# --------------------------------------------------------- F2  factorial view
SELECTOR_GROUPS = [
    ("random", "random", "none"),
    ("max-\ndegree", "maxdeg", "none"),
    ("MEC-\nEIG", "eig", "none"),
    ("LLM\n(Qwen)", "llm", QWEN),
    ("LLM\n(GPT)", "llm", GPT),
    ("truth-gain\nreference", "oracle", "none"),
]
READOUTS = [("mean-shift + Meek", "meek", ORANGE), ("LLM · Qwen", "llm", BLUE),
            ("LLM · GPT", "llm", "#8ebbee")]


def fig_factorial(df, out_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 2.95),
                                   gridspec_kw={"height_ratios": [1.0, 1.05]})

    # ---- (a) arm means -----------------------------------------------------
    width, xs, seen = 0.26, [], set()
    for gi, (label, selector, sel_tag) in enumerate(SELECTOR_GROUPS):
        xs.append(gi)
        for ri, (rlabel, readout, colour) in enumerate(READOUTS):
            arm = f"{selector}+{readout}"
            if readout == "meek":
                tag = sel_tag
            else:                                   # the readout model fixes the tag
                tag = QWEN if ri == 1 else GPT
                if selector == "llm" and sel_tag != tag:
                    continue                        # cell not run
            if cell(df, arm, tag).empty:
                continue
            m, c = mean_ci(df, arm, tag)
            ax1.bar(gi + (ri - 1) * width, m, width * 0.9, color=colour,
                    edgecolor="white", linewidth=0.4,
                    label=None if rlabel in seen else rlabel)
            seen.add(rlabel)
            ax1.errorbar(gi + (ri - 1) * width, m, yerr=c, fmt="none",
                         ecolor=DARK, elinewidth=0.7, capsize=1.6)

    base = len(SELECTOR_GROUPS) + 0.60           # unscaffolded reference, set apart
    for k, (tag, colour, lab) in enumerate([(QWEN, RED, "no scaffold · Qwen"),
                                            (GPT, "#e79a93", "no scaffold · GPT")]):
        m, c = mean_ci(df, "llm_e2e", tag)
        ax1.bar(base + (k - 0.5) * 0.3, m, width * 0.9, color=colour,
                edgecolor="white", linewidth=0.4, label=lab)
        ax1.errorbar(base + (k - 0.5) * 0.3, m, yerr=c, fmt="none",
                     ecolor=DARK, elinewidth=0.7, capsize=1.6)
        ax1.text(base + (k - 0.5) * 0.3, m + c + 0.02, f"{m:.3f}", ha="center",
                 va="bottom", fontsize=5.8, color=DARK)
    ax1.axvline(len(SELECTOR_GROUPS) - 0.25, color=LIGHT, lw=0.7)

    ceiling = df[df.arm == "oracle+meek"].pc_skeleton_f1_ceiling.mean()
    ax1.axhline(ceiling, color=GREY, lw=0.8, ls=(0, (4, 2)))
    ax1.text(base + 0.45, ceiling + 0.015, f"PC skeleton ceiling {ceiling:.3f}",
             ha="right", va="bottom", fontsize=6.0, color=GREY)

    ax1.set_xticks(xs + [base])
    ax1.set_xticklabels([g[0] for g in SELECTOR_GROUPS] + ["no\nscaffold"], fontsize=6.5)
    ax1.set_xlim(-0.55, base + 0.5)
    ax1.set_xlabel("selector (who chooses the experiment)", fontsize=7)
    ax1.set_ylabel("directed-edge F1")
    ax1.set_ylim(0, 1.05)
    ax1.set_yticks(np.arange(0, 1.01, 0.2))
    ax1.yaxis.grid(True); ax1.set_axisbelow(True)
    ax1.set_title("(a) arm means, 40 paired instances", loc="left", fontsize=7.5)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.20), ncol=5,
               handlelength=1.0, handletextpad=0.4, columnspacing=1.1, fontsize=6.4)

    # ---- (b) paired contrasts ---------------------------------------------
    rows = []
    for tag, lab in ((QWEN, "Qwen"), (GPT, "GPT")):
        rows.append((f"marginal over the 5 selectors · {lab}",
                     contrast(readout_marginal(df, tag, MECH_SELECTORS),
                              readout_marginal(df, tag, LLM_SELECTORS)), ORANGE))
    for tag, lab in ((QWEN, "Qwen"), (GPT, "GPT")):
        rows.append((f"at truth-gain selection · {lab}",
                     contrast(cell(df, "oracle+meek", "none"), cell(df, "oracle+llm", tag)), ORANGE))
    for tag, lab in ((QWEN, "Qwen"), (GPT, "GPT")):
        rows.append((f"at LLM selection · {lab}",
                     contrast(cell(df, "llm+meek", tag), cell(df, "llm+llm", tag)), ORANGE))
    sep = len(rows)
    for name, arm, tag in [("LLM · Qwen", "llm+meek", QWEN), ("LLM · GPT", "llm+meek", GPT),
                           ("MEC-EIG", "eig+meek", "none"), ("max-degree", "maxdeg+meek", "none"),
                           ("truth-gain reference", "oracle+meek", "none")]:
        rows.append((f"{name} − random", contrast(cell(df, arm, tag),
                                                  cell(df, "random+meek", "none")), BLUE))

    ypos = np.arange(len(rows))[::-1]
    for y, (label, r, colour) in zip(ypos, rows):
        ax2.errorbar(r["delta"], y, xerr=r["ci"], fmt="o", color=colour,
                     ecolor=colour, elinewidth=1.0, capsize=1.8, markersize=3.2)
        ax2.text(r["delta"] + r["ci"] + 0.008, y, f"{r['delta']:+.3f}", fontsize=6.0,
                 va="center", ha="left", color=DARK)
    ax2.axvline(0, color=DARK, lw=0.7)
    ax2.axhline(ypos[sep] + 0.5, color=LIGHT, lw=0.7)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels([r[0] for r in rows], fontsize=6.5)
    for tick, (_, _, colour) in zip(ax2.get_yticklabels(), rows):
        tick.set_color(colour)
    ax2.set_ylim(-0.7, len(rows) - 0.3)
    ax2.set_xlim(-0.06, 0.46)
    ax2.set_xticks(np.arange(0.0, 0.41, 0.1))
    ax2.set_xlabel("paired Δ directed F1 (95% CI)")
    ax2.xaxis.grid(True); ax2.set_axisbelow(True)
    ax2.set_title("(b) paired contrasts: readout (orange) and selector (blue)",
                  loc="left", fontsize=7.5)

    fig.subplots_adjust(hspace=0.80)
    save(fig, out_dir, "rauma_f2_factorial")


# ------------------------------------------------------ F3  regret vs outcome
DISSOC = [                       # ordered by selection regret, worst first
    ("random", "random+meek", "none"),
    ("LLM · GPT", "llm+meek", GPT),
    ("LLM · Qwen", "llm+meek", QWEN),
    ("MEC-EIG", "eig+meek", "none"),
    ("max-degree", "maxdeg+meek", "none"),
    ("truth-gain reference", "oracle+meek", "none"),
]


def fig_dissociation(main, tight, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.95), sharey=True,
                             gridspec_kw={"width_ratios": [1.0, 1.0, 0.8]})
    ypos = np.arange(len(DISSOC))[::-1]
    series = [("budget $|I^*|+1$ (main)", main, BLUE, "o", 0.16),
              ("budget $|I^*|$ (tight)", tight, ORANGE, "s", -0.16)]

    for label, df, colour, marker, dy in series:
        for y, (_, arm, tag) in zip(ypos, DISSOC):
            r = cell(df, arm, tag, "selection_regret_total")
            axes[0].errorbar(r.mean(), y + dy, xerr=ci95(r), fmt=marker, color=colour,
                             ecolor=colour, elinewidth=0.9, capsize=1.8, markersize=3.6,
                             label=label if y == ypos[0] else None)
            f = cell(df, arm, tag)
            axes[1].errorbar(f.mean(), y + dy, xerr=ci95(f), fmt=marker, color=colour,
                             ecolor=colour, elinewidth=0.9, capsize=1.8, markersize=3.6)
            if arm == "random+meek":
                continue
            c = contrast(f, cell(df, "random+meek", "none"))
            axes[2].errorbar(c["delta"], y + dy, xerr=c["ci"], fmt=marker, color=colour,
                             ecolor=colour, elinewidth=0.9, capsize=1.8, markersize=3.6)

    axes[0].set_xlabel("selection regret per episode")
    axes[0].set_xlim(-0.15, 2.3)
    axes[1].set_xlabel("directed-edge F1")
    axes[1].set_xlim(0.74, 0.95)
    axes[1].set_xticks([0.75, 0.80, 0.85, 0.90, 0.95])
    axes[2].set_xlabel("paired Δ F1 vs. random")
    axes[2].set_xlim(-0.035, 0.075)
    axes[2].set_xticks([-0.02, 0.0, 0.02, 0.04, 0.06])
    axes[2].axvline(0, color=DARK, lw=0.7)
    for ax, title in zip(axes, ("(a) selection regret (lower better)",
                                "(b) outcome, unpaired 95% CI",
                                "(c) paired contrast")):
        ax.set_ylim(-0.65, len(DISSOC) - 0.35)
        ax.xaxis.grid(True); ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=6.8)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([d[0] for d in DISSOC], fontsize=6.8)
    axes[0].legend(loc="lower right", bbox_to_anchor=(1.03, -0.04), handlelength=1.1,
                   labelspacing=0.25, fontsize=6.4)
    fig.subplots_adjust(wspace=0.08)
    save(fig, out_dir, "rauma_f3_dissociation")


# ------------------------------------------------------------- F4  graph size
SCALE_LINES = [
    ("PC skeleton ceiling", None, None, GREY, (0, (1, 1.6)), "None"),
    ("RauMa (LLM select + mean-shift)", "llm+meek", "TAG", BLUE, "-", "o"),
    ("MEC-EIG + mean-shift", "eig+meek", "none", "#6f6e6a", (0, (2, 2)), "s"),
    ("LLM select + LLM readout", "llm+llm", "TAG", ORANGE, "-", "^"),
    ("no scaffold (end-to-end LLM)", "llm_e2e", "TAG", RED, "-", "v"),
]


def fig_scaling(df, out_dir):
    levels = sorted(df.level.unique())
    dvals = [int(df[df.level == lv].d.iloc[0]) for lv in levels]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.85), sharey=True)
    for ax, tag, name in ((axes[0], QWEN, QL), (axes[1], GPT, GL)):
        for label, arm, tagspec, colour, ls, marker in SCALE_LINES:
            if arm is None:
                y = [df[(df.level == lv) & (df.arm == "oracle+meek")].pc_skeleton_f1_ceiling.mean()
                     for lv in levels]
                ax.plot(dvals, y, ls=ls, color=colour, lw=1.0, zorder=1)
                continue
            use = tag if tagspec == "TAG" else tagspec
            m, e = [], []
            for lv in levels:
                v = df[(df.level == lv) & (df.arm == arm) & (df.model_tag == use)].directed_f1
                m.append(v.mean()); e.append(ci95(v))
            ax.errorbar(dvals, m, yerr=e, color=colour, ls=ls, marker=marker,
                        markersize=3.0 if arm == "eig+meek" else 3.6,
                        lw=1.0 if arm == "eig+meek" else 1.4,
                        elinewidth=0.8, capsize=1.8,
                        zorder=3 if arm == "eig+meek" else 2)
        ax.set_xticks(dvals)
        ax.set_xlabel("number of variables $d$")
        ax.grid(True, lw=0.6); ax.set_axisbelow(True)
        ax.set_title(name, loc="left", fontsize=6.8)
    axes[0].set_ylabel("directed-edge F1")
    axes[0].set_ylim(-0.05, 1.05)
    handles = [plt.Line2D([], [], color=c, ls=ls, marker=(None if m == "None" else m),
                          markersize=3.6, lw=1.2, label=lab)
               for lab, _, _, c, ls, m in SCALE_LINES]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
               ncol=3, handlelength=1.8, columnspacing=1.4, fontsize=6.4)
    fig.subplots_adjust(wspace=0.08)
    save(fig, out_dir, "rauma_f4_scaling")


# ------------------------------------------------------------------ A1  cost
COST_POINTS = [
    ("RauMa", "llm+meek", QWEN, BLUE, "o"), ("RauMa", "llm+meek", GPT, BLUE, "s"),
    ("LLM readout", "llm+llm", QWEN, ORANGE, "o"), ("LLM readout", "llm+llm", GPT, ORANGE, "s"),
    ("LLM readout", "oracle+llm", QWEN, ORANGE, "o"), ("LLM readout", "oracle+llm", GPT, ORANGE, "s"),
    ("no scaffold", "llm_e2e", QWEN, RED, "o"), ("no scaffold", "llm_e2e", GPT, RED, "s"),
]
SHORT = {"llm+meek": "RauMa (LLM selector, mechanical readout)",
         "llm+llm": "LLM selector, LLM readout",
         "oracle+llm": "truth-gain selector, LLM readout",
         "llm_e2e": "no scaffold"}
COST_COLOUR = {"llm+meek": BLUE, "llm+llm": ORANGE, "oracle+llm": "#f2a07b",
               "llm_e2e": RED}


def fig_cost(df, out_dir):
    """Accuracy against context size. Direct labels collided, so the arms go in a legend
    and the marker shape carries the model."""
    fig, ax = plt.subplots(figsize=(5.0, 2.5))
    ref = df[(df.arm == "eig+meek")].directed_f1.mean()
    ax.axhline(ref, color=GREY, lw=0.9, ls=(0, (4, 2)))
    ax.text(0.99, ref + 0.02, f"mechanical selection + readout, 0 LLM tokens ({ref:.3f})",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=6.8, color=GREY)
    for arm in ("llm+meek", "llm+llm", "oracle+llm", "llm_e2e"):
        for tag, marker in ((QWEN, "o"), (GPT, "s")):
            x = df[(df.arm == arm) & (df.model_tag == tag)].total_tokens.mean()
            y = cell(df, arm, tag)
            ax.errorbar(x, y.mean(), yerr=ci95(y), color=COST_COLOUR[arm], marker=marker,
                        markersize=4.2, elinewidth=0.8, capsize=1.8, ls="none",
                        label=SHORT[arm] if tag == QWEN else None)
    for marker, label in (("o", QL), ("s", GL)):
        ax.plot([], [], marker=marker, ls="none", color=GREY, markersize=4.2, label=label)
    ax.set_xscale("log")
    ax.set_xlim(500, 1.4e5)
    ax.set_ylim(-0.05, 1.0)
    ax.set_xlabel("total tokens per episode (log scale)")
    ax.set_ylabel("directed-edge F1")
    ax.grid(True, lw=0.6); ax.set_axisbelow(True)
    ax.legend(loc="center left", bbox_to_anchor=(0.44, 0.46), handlelength=0.9,
              labelspacing=0.28, fontsize=6.2, borderpad=0.2)
    save(fig, out_dir, "rauma_a1_cost")


# ------------------------------------------------------------ A2  diagnostic
DIAG_METRICS = [("directed\nprecision", "directed_precision"), ("directed\nrecall", "directed_recall"),
                ("skeleton F1", "skeleton_f1"), ("compelled-edge\nF1", "compelled_f1")]
DIAG_ARMS = [("mean-shift + Meek", "oracle+meek", "none", ORANGE),
             (f"LLM · {QL}", "oracle+llm", QWEN, BLUE),
             (f"LLM · {GL}", "oracle+llm", GPT, "#7fb2ea")]


def fig_diagnostic(df, out_dir):
    fig, ax = plt.subplots(figsize=(5.0, 2.2))
    width = 0.26
    for ai, (label, arm, tag, colour) in enumerate(DIAG_ARMS):
        for mi, (_, metric) in enumerate(DIAG_METRICS):
            v = cell(df, arm, tag, metric)
            ax.bar(mi + (ai - 1) * width, v.mean(), width * 0.9, color=colour,
                   edgecolor="white", linewidth=0.4, label=label if mi == 0 else None)
            ax.errorbar(mi + (ai - 1) * width, v.mean(), yerr=ci95(v), fmt="none",
                        ecolor=DARK, elinewidth=0.7, capsize=1.8)
    ax.set_xticks(range(len(DIAG_METRICS)))
    ax.set_xticklabels([m[0] for m in DIAG_METRICS], fontsize=6.4)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, handlelength=1.1,
              columnspacing=1.2, labelspacing=0.25, fontsize=6.4)
    save(fig, out_dir, "rauma_a2_diagnostic")


# ------------------------------------------------------------- A3  mean-shift
def fig_meanshift(df, out_dir):
    mech = df[df.inferencer == "meek"]
    grouped = mech.groupby("level")[["orientations_correct", "orientations_wrong"]].sum()
    dvals = [int(mech[mech.level == lv].d.iloc[0]) for lv in grouped.index]
    total = grouped.sum(axis=1)
    rate = grouped.orientations_wrong / total
    pooled = grouped.orientations_wrong.sum() / total.sum()

    fig, ax = plt.subplots(figsize=(4.2, 2.0))
    ax.bar(range(len(dvals)), rate, 0.55, color=ORANGE, edgecolor="white", linewidth=0.4)
    for i, (r, w, n) in enumerate(zip(rate, grouped.orientations_wrong, total)):
        ax.text(i, r + 0.004, f"{int(w)}/{int(n)}", ha="center", va="bottom",
                fontsize=6.8, color=DARK)
    ax.axhline(pooled, color=GREY, lw=0.9, ls=(0, (4, 2)))
    ax.text(len(dvals) - 0.45, pooled + 0.004, f"pooled {pooled:.1%}",
            ha="right", va="bottom", fontsize=6.8, color=GREY)
    ax.set_xticks(range(len(dvals)))
    ax.set_xticklabels([f"$d={d}$" for d in dvals])
    ax.set_ylabel("orientations in the wrong direction")
    ax.set_ylim(0, max(rate) * 1.35)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    save(fig, out_dir, "rauma_a3_meanshift")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", default="result/study1")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    main_run = load(os.path.join(args.result_dir, "episodes.csv"))
    tight = load(os.path.join(args.result_dir, "ablation_episodes.csv"))
    print(f"loaded {len(main_run)} main and {len(tight)} tight-budget episodes")

    fig_factorial(main_run, args.out_dir)
    fig_dissociation(main_run, tight, args.out_dir)
    fig_scaling(main_run, args.out_dir)
    fig_cost(main_run, args.out_dir)
    fig_diagnostic(main_run, args.out_dir)
    fig_meanshift(main_run, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
