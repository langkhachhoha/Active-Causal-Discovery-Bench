#!/usr/bin/env python
"""Paper figures for the two active-experiment studies.

    python scripts/make_figures.py --result-dir result --out-dir figures

Reads the episode-level CSVs (and the Study 2 n_obs sweep tables) and writes one
PNG at 300 dpi plus one PDF per figure. Every panel is drawn from paired
per-instance data, so the error bars are 95% CIs over instances, not over runs.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ---------------------------------------------------------------- design tokens
# Validated categorical palette (adjacent CVD dE 9.1, normal-vision 22.9 on white).
# Aqua and yellow sit below 3:1 on white, so every series carrying them is direct-
# labelled -- that is the relief the contrast check requires, not an optional extra.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#4a3aa7", "#e34948"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
FLAT = "#d8d7d0"          # de-emphasis fill for "context, not the point"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

Q = "qwen3-coder-30b-a3b-instruct"
G = "gpt-4o-mini-2024-07-18"
QL, GL = "qwen3-coder-30b", "gpt-4o-mini"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": False,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "lines.markersize": 5,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def style(ax, xgrid=False, ygrid=False):
    ax.set_axisbelow(True)
    if xgrid:
        ax.xaxis.grid(True)
    if ygrid:
        ax.yaxis.grid(True)
    ax.tick_params(length=3, pad=2)
    return ax


def save(fig, out_dir, name):
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  [written] {out_dir}/{name}.png + .pdf")


# ------------------------------------------------------------------ data access
def load(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def num(row, col):
    try:
        return float(row[col])
    except (TypeError, ValueError, KeyError):
        return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else float("nan")


def ci95(xs):
    xs = [x for x in xs if x is not None]
    return 1.96 * st.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0


def cell(rows, arm, model_tag, col="directed_f1"):
    """Per-instance values for one arm, keyed by (level, seed) so arms stay paired."""
    return {(r["level"], r["seed"]): num(r, col)
            for r in rows if r["arm"] == arm and r["model_tag"] == model_tag}


def gap(rows, a_arm, a_model, b_arm, b_model, col="directed_f1"):
    """Paired mean difference a - b, plus its 95% CI, over shared instances."""
    a, b = cell(rows, a_arm, a_model, col), cell(rows, b_arm, b_model, col)
    keys = sorted(a.keys() & b.keys())
    diffs = [a[k] - b[k] for k in keys]
    return (mean(diffs), ci95(diffs), len(keys)) if diffs else (float("nan"), 0.0, 0)


def agg(rows, arm, model_tag, col="directed_f1"):
    vals = list(cell(rows, arm, model_tag, col).values())
    return mean(vals), ci95(vals)


def parse_sweep(result_dir):
    """Pull (mean, ci) per arm out of each n_obs_*.md ablation table."""
    out = {}
    pattern = re.compile(r"\|\s*([a-z_0-9]+)\s*\|\s*([\w.\-]+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*±\s*([\d.]+)")
    for path in glob.glob(os.path.join(result_dir, "study2", "n_obs_*.md")):
        n_obs = int(re.search(r"n_obs_(\d+)", path).group(1))
        table = {}
        for line in open(path, encoding="utf-8"):
            m = pattern.match(line.strip())
            if m and (m.group(1), m.group(2)) not in table:
                table[(m.group(1), m.group(2))] = (float(m.group(4)), float(m.group(5)))
        out[n_obs] = table
    return dict(sorted(out.items()))


# ===================================================================== STUDY 1
def fig_decomposition(s1, out_dir):
    """Where the end-to-end gap comes from: a paired waterfall per model.

    Selection and inference are measured independently against the same ceiling,
    so their sum need not reproduce the joint arm -- the residual is shown rather
    than absorbed, which is the honest way to present a two-factor decomposition.
    """
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9), sharey=True)
    for ax, model, label in zip(axes, (Q, G), (QL, GL)):
        ceiling, _ = agg(s1, "oracle+meek", "none")
        joint, _ = agg(s1, "llm+llm", model)
        sel, sel_ci, _ = gap(s1, "llm+meek", model, "oracle+meek", "none")
        inf, inf_ci, _ = gap(s1, "oracle+llm", model, "oracle+meek", "none")
        resid = (joint - ceiling) - sel - inf
        e2e, _ = agg(s1, "llm_e2e", model)

        steps = [("ceiling", ceiling, None, FLAT),
                 ("LLM\nselects", sel, sel_ci, BLUE),
                 ("LLM\ninfers", inf, inf_ci, ORANGE),
                 ("inter-\naction", resid, None, FLAT),
                 ("full LLM\nagent", joint, None, FLAT)]
        running = 0.0
        for i, (name, val, err, color) in enumerate(steps):
            if i in (0, 4):
                ax.bar(i, val, 0.6, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
                running = val
                ax.text(i, val + 0.02, f"{val:.3f}", ha="center", va="bottom",
                        fontsize=8, color=INK, fontweight="bold")
            else:
                bottom = running + val if val < 0 else running
                height = max(abs(val), 0.004)          # keep a hairline visible at |val| ~ 0
                # A surface-coloured edge would eat a bar this thin, so drop it there.
                ax.bar(i, height, 0.6, bottom=bottom, color=color, edgecolor=SURFACE,
                       linewidth=0.8 if height > 0.05 else 0.0, zorder=3)
                if err:
                    ax.errorbar(i, bottom + abs(val) / 2, yerr=err, color=SURFACE,
                                elinewidth=1.4, capsize=2.5, capthick=1.4, zorder=4)
                running += val
                ax.text(i, bottom + height + 0.02, f"{val:+.3f}", ha="center",
                        va="bottom", fontsize=8, color=INK, fontweight="bold")
            if i < 4:
                ax.plot([i + 0.3, i + 1 - 0.3], [running, running], color=AXIS,
                        lw=0.7, zorder=2)
        ax.axhline(e2e, color=RED, lw=1.2, ls=(0, (4, 2)), zorder=1)
        ax.text(4.45, e2e + 0.02, f"no scaffold  {e2e:.3f}", ha="right", va="bottom",
                fontsize=7.2, color=RED)
        ax.set_xticks(range(5))
        ax.set_xticklabels([s[0] for s in steps], fontsize=7.4, color=INK2, linespacing=1.3)
        ax.set_ylim(0, 1.02)
        ax.set_title(label, color=INK, pad=6, loc="left")
        style(ax, ygrid=True)
    axes[0].set_ylabel("Directed F1")
    fig.suptitle("Every point of the end-to-end gap is spent on reading results, not on choosing experiments",
                 y=1.06, fontsize=10, color=INK, x=0.02, ha="left")
    save(fig, out_dir, "s1_f1_decomposition")


def fig_grid(s1, out_dir):
    """The 6x3 arm grid. Rows are who chooses; columns are who reads."""
    sel_rows = [("random", "none", "random"), ("maxdeg", "none", "max-degree"),
                ("eig", "none", "EIG (BOED)"), ("llm", Q, f"LLM · {QL}"),
                ("llm", G, f"LLM · {GL}"), ("oracle", "none", "oracle (|I*|)")]
    inf_cols = [("meek", "none", "Meek rules"), ("llm", Q, f"LLM · {QL}"),
                ("llm", G, f"LLM · {GL}")]
    mat = np.full((len(sel_rows), len(inf_cols)), np.nan)
    for i, (sname, smodel, _) in enumerate(sel_rows):
        for j, (iname, imodel, _) in enumerate(inf_cols):
            if sname == "llm" and iname == "llm" and smodel != imodel:
                continue
            if sname == "llm" and iname == "meek":
                arm, tag = "llm+meek", smodel
            elif sname == "llm":
                arm, tag = "llm+llm", smodel
            elif iname == "meek":
                arm, tag = f"{sname}+meek", "none"
            else:
                arm, tag = f"{sname}+llm", imodel
            vals = list(cell(s1, arm, tag).values())
            if vals:
                mat[i, j] = mean(vals)

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
    im = ax.imshow(mat, cmap=cmap, vmin=0.45, vmax=0.90, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isnan(mat[i, j]):
                ax.text(j, i, "—", ha="center", va="center", color=MUTED, fontsize=9)
            else:
                dark = mat[i, j] > 0.72
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                        color="#ffffff" if dark else INK, fontsize=8.5,
                        fontweight="bold" if dark else "normal")
    ax.set_xticks(range(len(inf_cols)))
    ax.set_xticklabels([c[2] for c in inf_cols], fontsize=7.6, color=INK2)
    ax.set_yticks(range(len(sel_rows)))
    ax.set_yticklabels([r[2] for r in sel_rows], fontsize=7.6, color=INK2)
    ax.set_xlabel("who reads the intervention results", labelpad=6)
    ax.set_ylabel("who chooses the experiment", labelpad=6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-.5, len(inf_cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(sel_rows), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Directed F1", fontsize=8, color=INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, labelsize=7.5, color=MUTED)
    ax.set_title("Columns move the score; rows barely do", color=INK, pad=8, loc="left")
    save(fig, out_dir, "s1_f2_grid")


def fig_dissociation(s1, out_dir):
    """Selection regret separates the selectors 8x; final accuracy does not."""
    sels = [("random", "none", "random"), ("llm", G, f"LLM · {GL}"),
            ("eig", "none", "EIG (BOED)"), ("llm", Q, f"LLM · {QL}"),
            ("maxdeg", "none", "max-degree"), ("oracle", "none", "oracle")]
    regret, regret_ci, f1, f1_ci, labels = [], [], [], [], []
    for name, model, label in sels:
        arm = "llm+meek" if name == "llm" else f"{name}+meek"
        tag = model if name == "llm" else "none"
        r = list(cell(s1, arm, tag, "selection_regret_total").values())
        f = list(cell(s1, arm, tag).values())
        regret.append(mean(r)); regret_ci.append(ci95(r))
        f1.append(mean(f)); f1_ci.append(ci95(f)); labels.append(label)
    order = np.argsort(regret)[::-1]
    y = np.arange(len(order))

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.5), sharey=True)
    colors = [ORANGE if labels[i] == "random" else BLUE for i in order]
    axes[0].barh(y, [regret[i] for i in order], 0.5, xerr=[regret_ci[i] for i in order],
                 color=colors, edgecolor=SURFACE, linewidth=1.5,
                 error_kw=dict(ecolor=MUTED, elinewidth=1, capsize=2.5, capthick=1))
    for k, i in enumerate(order):
        axes[0].text(regret[i] + regret_ci[i] + 0.06, k, f"{regret[i]:.2f}",
                     va="center", fontsize=8, color=INK2)
    axes[0].set_xlabel("Selection regret over the episode  (lower is better)")
    axes[0].set_title("Choices differ 9-fold …", color=INK, pad=6, loc="left")
    axes[0].set_xlim(0, max(r + c for r, c in zip(regret, regret_ci)) * 1.28)

    # A dot plot, not bars: the panel is deliberately zoomed to 0.79-0.90 to show that
    # nothing separates, and a bar truncated away from zero would overstate the spread.
    axes[1].errorbar([f1[i] for i in order], y, xerr=[f1_ci[i] for i in order],
                     fmt="o", markersize=6, color=BLUE, ecolor=MUTED, elinewidth=1.2,
                     capsize=3, capthick=1.2, markeredgecolor=SURFACE,
                     markeredgewidth=1.4, linestyle="none", zorder=3)
    axes[1].scatter([f1[order[0]]], [0], s=52, color=ORANGE, edgecolor=SURFACE,
                    linewidth=1.4, zorder=4)
    for k, i in enumerate(order):
        axes[1].text(f1[i] + f1_ci[i] + 0.004, k, f"{f1[i]:.3f}", va="center",
                     fontsize=8, color=INK2)
    axes[1].set_xlabel("Directed F1  (higher is better)")
    axes[1].set_title("… but the score does not follow", color=INK, pad=6, loc="left")
    axes[1].set_xlim(0.772, 0.912)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([labels[i] for i in order], fontsize=8, color=INK2)
    axes[0].invert_yaxis()
    for ax in axes:
        style(ax, xgrid=True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    save(fig, out_dir, "s1_f3_dissociation")


def fig_budget(s1, tight, out_dir):
    """Same graphs, same data, one less experiment: does the selector matter now?"""
    sels = [("random+meek", "none", "random"), ("oracle+meek", "none", "oracle (|I*|)"),
            ("maxdeg+meek", "none", "max-degree"), ("eig+meek", "none", "EIG (BOED)"),
            ("llm+meek", G, f"LLM · {GL}"), ("llm+meek", Q, f"LLM · {QL}")]
    rows = []
    for arm, tag, label in sels:
        a = mean(list(cell(s1, arm, tag).values()))
        b = mean(list(cell(tight, arm, tag).values()))
        rows.append((label, a, b))
    rows.sort(key=lambda r: r[2])
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    for i, (label, a, b) in enumerate(rows):
        ax.plot([a, b], [i, i], color=AXIS, lw=1.4, zorder=1, solid_capstyle="round")
    ax.scatter([r[1] for r in rows], y, s=52, color=FLAT, edgecolor=SURFACE,
               linewidth=1.6, zorder=3, label="budget = |I*| + 1  (main)")
    ax.scatter([r[2] for r in rows], y, s=52, color=BLUE, edgecolor=SURFACE,
               linewidth=1.6, zorder=3, label="budget = |I*|  (tight)")
    for i, (label, a, b) in enumerate(rows):
        ax.text(min(a, b) - 0.006, i, f"{b:.3f}", va="center", ha="right",
                fontsize=7.8, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8, color=INK2)
    ax.set_xlabel("Directed F1")
    ax.set_xlim(0.785, 0.875)
    ax.set_title("Under a tight budget random selection falls to last, alone",
                 color=INK, pad=6, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2,
              handletextpad=0.4, columnspacing=1.6)
    style(ax, xgrid=True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, out_dir, "s1_f4_tight_budget")


def fig_efficiency(s1, out_dir):
    """Tokens spent against accuracy bought. The no-scaffold agent spends the most
    and buys the least, which is the whole argument for the scaffold in one panel."""
    picks = [("llm+meek", Q, f"RauMa · {QL}", BLUE, (11, 4), "left"),
             ("llm+meek", G, f"RauMa · {GL}", BLUE, (11, -11), "left"),
             ("llm+llm", Q, f"llm+llm · {QL}", ORANGE, (12, -2), "left"),
             ("llm+llm", G, f"llm+llm · {GL}", ORANGE, (0, -17), "center"),
             ("oracle+llm", Q, f"oracle+llm · {QL}", ORANGE, (11, 9), "left"),
             ("llm_e2e", Q, f"no scaffold · {QL}", RED, (-11, 4), "right"),
             ("llm_e2e", G, f"no scaffold · {GL}", RED, (-11, 0), "right")]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ceiling, _ = agg(s1, "oracle+meek", "none")
    ax.axhline(ceiling, color=AXIS, lw=1.0, ls=(0, (4, 2)), zorder=1)
    ax.text(9.0e4, ceiling + 0.018, f"symbolic ceiling  {ceiling:.3f}  (0 tokens)",
            ha="right", fontsize=7.5, color=MUTED)
    for arm, tag, label, color, off, ha in picks:
        tok = mean(list(cell(s1, arm, tag, "total_tokens").values()))
        f1 = mean(list(cell(s1, arm, tag).values()))
        ax.scatter(tok, f1, s=64, color=color, edgecolor=SURFACE, linewidth=1.8, zorder=4)
        ax.annotate(label, (tok, f1), textcoords="offset points", xytext=off,
                    ha=ha, fontsize=7.2, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("Total tokens per episode  (log scale)")
    ax.set_ylabel("Directed F1")
    ax.set_xlim(7e2, 1.0e5)
    ax.set_ylim(-0.08, 0.99)
    ax.set_title("Spending 30x more tokens buys less than nothing",
                 color=INK, pad=6, loc="left")
    style(ax, xgrid=True, ygrid=True)
    save(fig, out_dir, "s1_f5_efficiency")


# ===================================================================== STUDY 2
def fig_probe_main(s2, out_dir):
    """Full ranking with the three reference lines a reader needs to place it."""
    arms = [("oracle", "none", "oracle (upper bound)", FLAT),
            ("probe", G, f"NemChua · {GL}", BLUE),
            ("probe", Q, f"NemChua · {QL}", BLUE),
            ("probe_skel_only", "none", "PC skeleton only (no LLM)", FLAT),
            ("probe_mec_only", "none", "PC MEC only", FLAT),
            ("pc_greedy_meek", "none", "PC + greedy + Meek", ORANGE),
            ("pc_greedy", "none", "PC + greedy", ORANGE),
            ("probe_llm_graphs", Q, f"LLM whole-graph proposals · {QL}", FLAT),
            ("probe_no_update", Q, f"NemChua without Bayesian update · {QL}", FLAT),
            ("probe_random_hyp", "none", "random hypotheses", FLAT),
            ("llm_e2e", Q, f"no scaffold · {QL}", RED),
            ("llm_e2e", G, f"no scaffold · {GL}", RED)]
    vals = [(label, *agg(s2, arm, tag), color) for arm, tag, label, color in arms]
    vals.sort(key=lambda r: r[1])
    y = np.arange(len(vals))

    fig, ax = plt.subplots(figsize=(6.9, 3.6))
    ax.barh(y, [v[1] for v in vals], 0.5, xerr=[v[2] for v in vals],
            color=[v[3] for v in vals], edgecolor=SURFACE, linewidth=1.5,
            error_kw=dict(ecolor=MUTED, elinewidth=1, capsize=2.5, capthick=1), zorder=3)
    for i, v in enumerate(vals):
        ax.text(v[1] + v[2] + 0.012, i, f"{v[1]:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([v[0] for v in vals], fontsize=7.8, color=INK2)
    ax.set_xlabel("Directed F1")
    ax.set_xlim(0, 1.12)
    ax.set_title("NemChua beats the classical pipeline and the end-to-end agent alike",
                 color=INK, pad=6, loc="left")
    ax.legend(handles=[Patch(facecolor=BLUE, label="NemChua (ours)"),
                       Patch(facecolor=ORANGE, label="classical baseline"),
                       Patch(facecolor=RED, label="LLM-agent baseline"),
                       Patch(facecolor=FLAT, label="ablation / reference")],
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4,
              handlelength=1.1, handletextpad=0.5, columnspacing=1.4)
    style(ax, xgrid=True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, out_dir, "s2_f6_main")


def fig_crossover(sweep, out_dir):
    """The LLM proposer earns its keep exactly where PC's skeleton is unreliable."""
    ns = sorted(sweep)
    series = [(("probe", G), f"NemChua · {GL}", BLUE, "o"),
              (("probe", Q), f"NemChua · {QL}", AQUA, "s"),
              (("probe_skel_only", "none"), "PC skeleton only (no LLM)", ORANGE, "^"),
              (("pc_greedy_meek", "none"), "PC + greedy + Meek", YELLOW, "D")]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.1))
    ax = axes[0]
    # The four series cross repeatedly, so identity rides on the legend + marker shape
    # rather than end-of-line labels, which collide exactly where the lines converge.
    for key, label, color, marker in series:
        ys = [sweep[n].get(key, (np.nan, 0))[0] for n in ns]
        es = [sweep[n].get(key, (np.nan, 0))[1] for n in ns]
        ax.errorbar(ns, ys, yerr=es, color=color, marker=marker, markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, capsize=2,
                    elinewidth=0.8, ecolor=color, alpha=0.95, zorder=3, label=label)
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlim(34, 1250)
    ax.set_xlabel("Observational sample size $n_{obs}$  (log scale)")
    ax.set_ylabel("Directed F1")
    ax.set_title("All methods improve with data …", color=INK, pad=6, loc="left")
    ax.legend(loc="lower right", fontsize=7, handletextpad=0.5, labelspacing=0.35)
    style(ax, ygrid=True)

    ax = axes[1]
    best = []
    for n in ns:
        skel = sweep[n].get(("probe_skel_only", "none"), (np.nan, 0))[0]
        probe = max(sweep[n].get(("probe", G), (np.nan, 0))[0],
                    sweep[n].get(("probe", Q), (np.nan, 0))[0])
        best.append(probe - skel)
    ax.axhline(0, color=AXIS, lw=1.0, zorder=2)
    ax.plot(ns, best, color=BLUE, marker="o", markersize=5, markeredgecolor=SURFACE,
            markeredgewidth=1.2, zorder=4)
    ax.fill_between(ns, 0, best, where=[b >= 0 for b in best], color=BLUE, alpha=0.16,
                    interpolate=True, zorder=1)
    ax.fill_between(ns, 0, best, where=[b <= 0 for b in best], color=RED, alpha=0.16,
                    interpolate=True, zorder=1)
    for k, (n, b) in enumerate(zip(ns, best)):
        ax.annotate(f"{b:+.3f}", (n, b), textcoords="offset points",
                    xytext=(12 if k in (0, 1) else 0,
                            4 if k in (0, 1) else (10 if b >= 0 else -15)),
                    ha="left" if k in (0, 1) else "center", fontsize=7.2,
                    color=BLUE if b >= 0 else RED)
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlim(34, 1400)
    ax.set_xlabel("Observational sample size $n_{obs}$  (log scale)")
    ax.set_ylabel("NemChua − PC skeleton only")
    ax.set_ylim(-0.05, 0.085)
    ax.set_title("… but the LLM only helps when data is scarce", color=INK, pad=6,
                 loc="left", fontsize=9)
    style(ax, ygrid=True)
    save(fig, out_dir, "s2_f7_crossover")


def fig_components(s2, out_dir):
    """What each part of NemChua is worth, measured by removing it."""
    parts = [(("probe_no_update", Q), "Bayesian posterior update"),
             (("probe_random_hyp", "none"), "an informed hypothesis space"),
             (("probe_llm_graphs", Q), "skeleton repair (vs whole-graph proposals)"),
             (("probe_mec_only", "none"), "hybrid space (vs PC's MEC alone)"),
             (("probe_skel_only", "none"), "the LLM proposer"),
             (("probe_no_bic", Q), "BIC weighting"),
             (("probe_random_sel", Q), "EIG experiment selection"),
             (("probe_marginal", Q), "MAP submission (vs marginal)")]
    rows = []
    for (arm, tag), label in parts:
        d, e, _ = gap(s2, "probe", Q, arm, tag)
        rows.append((label, d, e))
    rows.sort(key=lambda r: r[1])
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    # Emphasis, not a value ramp: the three components that actually carry the method
    # take the accent hue and everything else recedes.
    colors = [BLUE if r[1] > 0.05 else FLAT for r in rows]
    ax.barh(y, [r[1] for r in rows], 0.5, xerr=[r[2] for r in rows], color=colors,
            edgecolor=SURFACE, linewidth=1.5,
            error_kw=dict(ecolor=MUTED, elinewidth=1, capsize=2.5, capthick=1), zorder=3)
    for i, r in enumerate(rows):
        ax.text(r[1] + r[2] + 0.012, i, f"{r[1]:+.3f}", va="center", fontsize=8, color=INK2)
    ax.axvline(0, color=AXIS, lw=0.8, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.8, color=INK2)
    ax.set_xlabel("Directed F1 lost when this component is removed  (paired, n = 40)")
    ax.set_xlim(-0.03, 0.70)
    ax.set_title("The hypothesis space carries NemChua; the selection rule barely matters",
                 color=INK, pad=6, loc="left")
    style(ax, xgrid=True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, out_dir, "s2_f8_components")


def fig_hypotheses(s2, out_dir):
    """Why NemChua wins: the true graph is more often inside the space it searches."""
    sources = [("probe_random_hyp", "none", "random"),
               ("probe_llm_graphs", Q, f"LLM whole graphs · {QL}"),
               ("probe_mec_only", "none", "PC MEC"),
               ("probe_skel_only", "none", "PC skeleton"),
               ("probe", Q, f"NemChua · {QL}"),
               ("probe", G, f"NemChua · {GL}")]
    labels, truth, bestf1 = [], [], []
    for arm, tag, label in sources:
        labels.append(label)
        truth.append(mean(list(cell(s2, arm, tag, "truth_in_hypotheses").values())))
        bestf1.append(mean(list(cell(s2, arm, tag, "best_f1_in_hypotheses").values())))
    y = np.arange(len(labels))
    h = 0.36

    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    ax.barh(y + h / 2, bestf1, h, color=FLAT, edgecolor=SURFACE, linewidth=1.5,
            zorder=3, label="best F1 available in the space")
    ax.barh(y - h / 2, truth, h, color=BLUE, edgecolor=SURFACE, linewidth=1.5,
            zorder=3, label="true DAG is in the space")
    for i in range(len(labels)):
        ax.text(bestf1[i] + 0.012, i + h / 2, f"{bestf1[i]:.3f}", va="center",
                fontsize=7.6, color=INK2)
        ax.text(truth[i] + 0.012, i - h / 2, f"{truth[i]:.3f}", va="center",
                fontsize=7.6, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.8, color=INK2)
    ax.set_xlabel("Fraction of instances  /  F1")
    ax.set_xlim(0, 1.12)
    ax.set_title("Skeleton repair raises the ceiling NemChua can reach",
                 color=INK, pad=6, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2,
              handlelength=1.1, handletextpad=0.5, columnspacing=1.6)
    style(ax, xgrid=True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, out_dir, "s2_f9_hypothesis_space")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", default="result")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    s1 = load(os.path.join(args.result_dir, "study1", "episodes.csv"))
    tight = load(os.path.join(args.result_dir, "study1", "ablation_episodes.csv"))
    s2 = load(os.path.join(args.result_dir, "study2", "episodes.csv"))
    sweep = parse_sweep(args.result_dir)

    print("Study 1")
    fig_decomposition(s1, args.out_dir)
    fig_grid(s1, args.out_dir)
    fig_dissociation(s1, args.out_dir)
    fig_budget(s1, tight, args.out_dir)
    fig_efficiency(s1, args.out_dir)
    print("Study 2")
    fig_probe_main(s2, args.out_dir)
    fig_crossover(sweep, args.out_dir)
    fig_components(s2, args.out_dir)
    fig_hypotheses(s2, args.out_dir)


if __name__ == "__main__":
    main()
