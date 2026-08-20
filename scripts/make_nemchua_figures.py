#!/usr/bin/env python
"""NemChua (study 2) figures.

    python scripts/make_nemchua_figures.py --study-dir study2 --out-dir figures

Each figure resolves its own run directory, preferring the 20-seed re-runs and falling
back to the original 10-seed ones, so the script produces a complete figure set at any
point during the run schedule. Every figure prints which directory it used.

Written figures
    nemchua_f2_main       headline accuracy, and accuracy per experiment
    nemchua_f3_wrong      the proposals are near chance, and the set still improves
    nemchua_f4_mechanism  why a wrong proposal is free: the guard, and the posterior
    nemchua_f5_crossover  where the proposal channel earns its keep
    nemchua_f6_models     capability sweep and the semantic condition
"""
from __future__ import annotations

import argparse
import json
import os
from math import comb

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#c8352b"
GREEN, PURPLE = "#2e8b6f", "#7b5ea7"
GREY, DARK, LIGHT = "#6f6e6a", "#1a1a1a", "#b9b8b2"
GRID, AXIS = "#e3e2dc", "#bfbeb4"
PALE_BLUE, PALE_ORANGE = "#8ebbee", "#f5b294"

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
    "lines.linewidth": 1.4, "lines.markersize": 4, "errorbar.capsize": 2,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

STUDY_DIR = "study2"


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def pick(*names: str) -> str | None:
    """First run directory that exists, so new re-runs supersede old ones."""
    for name in names:
        path = os.path.join(STUDY_DIR, name, "episodes.csv")
        if os.path.exists(path):
            return os.path.join(STUDY_DIR, name)
    return None


def load(run_dir: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(run_dir, "episodes.csv"))
    return df[df["status"] == "success"].copy()


def steps_of(run_dir: str) -> pd.DataFrame:
    path = os.path.join(run_dir, "steps.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def save(fig, out_dir: str, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  [written] {out_dir}/{name}.pdf")


def series(df: pd.DataFrame, arm: str, metric: str, model: str | None = None) -> dict:
    sub = df[df["arm"] == arm]
    if model is not None and (sub["model_tag"] != "none").any():
        sub = sub[sub["model_tag"] == model]
    return {(int(r.level), int(r.seed)): float(getattr(r, metric)) for r in sub.itertuples()}


def paired(a: dict, b: dict) -> tuple[float, float, float, float, int]:
    """Mean paired difference, bootstrap CI, Wilcoxon p, n."""
    keys = sorted(set(a) & set(b))
    if not keys:
        return np.nan, np.nan, np.nan, np.nan, 0
    d = np.array([a[k] - b[k] for k in keys])
    rng = np.random.default_rng(0)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)])
    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return d.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5), p, len(d)


def mean_ci(values) -> tuple[float, float]:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if len(v) < 2:
        return (float(v.mean()) if len(v) else np.nan), 0.0
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))


def models_in(df: pd.DataFrame) -> list[str]:
    return sorted(m for m in df["model_tag"].unique() if m != "none")


SHORT = {
    "qwen3-coder-30b-a3b-instruct": "qwen3-30b",
    "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
    "claude-haiku-4.5": "haiku-4.5",
    "claude-sonnet-4.6": "sonnet-4.6",
    "gemini-3-flash-preview": "gemini-3-flash",
    "gpt-5.4-mini": "gpt-5.4-mini",
    "gpt-5.5": "gpt-5.5",
}
def short(tag: str) -> str:
    return SHORT.get(tag, tag[:14])


# --------------------------------------------------------------------------- #
# edit precision, recomputed from what each episode already records
# --------------------------------------------------------------------------- #
def edit_stats_from_audit(run_dir: str) -> pd.DataFrame:
    """Per-model precision and chance rate from scripts/nemchua_edit_audit.py."""
    path = os.path.join(run_dir, "analysis", "edit_audit.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    a = pd.read_csv(path)
    rows = []
    for model, g in a.groupby("model_tag"):
        proposed = int((g["n_remove"] + g["n_add"]).sum())
        correct = int((g["correct_remove"] + g["correct_add"]).sum())
        share_add = g["n_add"].sum() / max(proposed, 1)
        chance = share_add * g["chance_add"].mean() + (1 - share_add) * g["chance_remove"].mean()
        precision = correct / max(proposed, 1)
        rows.append(dict(
            model_tag=model, n=len(g), proposed=proposed, correct=correct,
            precision=precision, chance=chance, lift=precision / max(chance, 1e-9),
            add_recall=g["correct_add"].sum() / max(g["pc_fn"].sum(), 1e-9),
        ))
    return pd.DataFrame(rows)


def edit_stats(df: pd.DataFrame, arm: str = "probe") -> pd.DataFrame:
    """Per-model edit precision and the chance rate an uninformed editor would hit.

    `pc_skeleton_f1_ceiling` plus the two PC edge counts pin down PC's own confusion
    matrix against the truth, which is what the chance rate is defined against.
    """
    need = {"repair_remove", "repair_add", "edits_correct_remove", "edits_correct_add",
            "pc_directed_edges", "pc_undirected_edges", "pc_skeleton_f1_ceiling", "true_edges", "d"}
    sub = df[df["arm"] == arm].dropna(subset=list(need & set(df.columns)))
    if not need <= set(df.columns) or sub.empty:
        return pd.DataFrame()
    rows = []
    for model, g in sub.groupby("model_tag"):
        n_pc = g["pc_directed_edges"] + g["pc_undirected_edges"]
        n_true = g["true_edges"]
        overlap = g["pc_skeleton_f1_ceiling"] * (n_pc + n_true) / 2.0
        pc_fp = (n_pc - overlap).clip(lower=0)
        pc_fn = (n_true - overlap).clip(lower=0)
        n_nonadj = g["d"].map(lambda d: comb(int(d), 2)) - n_pc
        proposed = g["repair_remove"] + g["repair_add"]
        correct = g["edits_correct_remove"] + g["edits_correct_add"]
        chance_add = (pc_fn / n_nonadj.clip(lower=1)).mean()
        chance_rm = (pc_fp / n_pc.clip(lower=1)).mean()
        # an uninformed editor spends its edits in the same mix the model does
        share_add = g["repair_add"].sum() / max(proposed.sum(), 1)
        chance = share_add * chance_add + (1 - share_add) * chance_rm
        precision = correct.sum() / max(proposed.sum(), 1)
        rows.append(dict(
            model_tag=model, n=len(g), proposed=int(proposed.sum()), correct=int(correct.sum()),
            precision=precision, chance=chance, lift=precision / max(chance, 1e-9),
            add_recall=g["edits_correct_add"].sum() / max(pc_fn.sum(), 1e-9),
        ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figure 2 — headline accuracy and accuracy per experiment
# --------------------------------------------------------------------------- #
def fig_main(out_dir: str) -> None:
    run = pick("main_v2", "main")
    if run is None:
        print("  [skip] nemchua_f2_main: no main run"); return
    df = load(run)
    print(f"  [f2] {run}")
    mods = models_in(df)
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.1), gridspec_kw={"width_ratios": [1.35, 1]})

    # (a) arm means
    order = [("oracle", "oracle", None), ("probe", "NemChua", "llm"),
             ("probe_skel_only", "no LLM,\nsame pipeline", None),
             ("pc_greedy_meek", "PC + greedy", None),
             ("probe_llm_graphs", "LLM writes\nthe DAG", "llm"),
             ("llm_e2e", "LLM agent\nend-to-end", "llm")]
    ax = axes[0]
    width = 0.36
    for mi, model in enumerate(mods):
        heights, errs, xs = [], [], []
        for i, (arm, _, kind) in enumerate(order):
            vals = df[(df["arm"] == arm) & ((df["model_tag"] == model) if kind else True)]["directed_f1"]
            m, e = mean_ci(vals)
            heights.append(m); errs.append(e)
            xs.append(i + (mi - (len(mods) - 1) / 2) * width)
        ax.bar(xs, heights, width * 0.92, yerr=errs, color=[BLUE, ORANGE][mi % 2],
               edgecolor="white", linewidth=0.4, label=short(model),
               error_kw=dict(ecolor=GREY, elinewidth=0.7))
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([lab for _, lab, _ in order], fontsize=6.2)
    ax.set_ylabel("directed-edge F1"); ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.legend(loc="upper right", ncol=2, fontsize=6)
    ax.set_title("(a)  same instances, same budget", loc="left", fontweight="bold")

    # (b) accuracy per experiment, carried forward after an arm stops
    ax = axes[1]
    st = steps_of(run)
    if not st.empty and "map_directed_f1_after" in st.columns:
        for arm, model, color, label in [
            ("probe", mods[0] if mods else None, BLUE, "NemChua"),
            ("pc_greedy_meek", None, GREY, "PC + greedy"),
        ]:
            sub = st[st["arm"] == arm]
            if model is not None and (sub["model_tag"] != "none").any():
                sub = sub[sub["model_tag"] == model]
            sub = sub.dropna(subset=["map_directed_f1_after"])
            if sub.empty:
                continue
            wide = sub.pivot_table(index=["level", "seed"], columns="step",
                                   values="map_directed_f1_after")
            wide = wide.reindex(columns=sorted(wide.columns)).ffill(axis=1)
            xs = list(wide.columns)
            ys = [wide[c].mean() for c in wide.columns]
            es = [1.96 * wide[c].std(ddof=1) / np.sqrt(wide[c].count()) for c in wide.columns]
            ax.errorbar(xs, ys, yerr=es, color=color, marker="o", label=label,
                        markerfacecolor="white", markeredgewidth=1.1)
            if arm == "pc_greedy_meek":
                final = load(run)
                final = final[final["arm"] == "pc_greedy_meek"]["directed_f1"].mean()
                ax.axhline(final, color=GREY, lw=0.8, ls=(0, (3, 2)))
                ax.text(3.9, final + 0.004, "PC + greedy, full budget",
                        fontsize=6, color=GREY, ha="right")
        ax.set_xlabel("interventions performed"); ax.set_ylabel("directed-edge F1")
        ax.set_xticks(range(1, 5))
        ax.yaxis.grid(True); ax.set_axisbelow(True); ax.legend(loc="lower right")
    ax.set_title("(b)  accuracy per experiment", loc="left", fontweight="bold")
    fig.tight_layout(w_pad=1.6)
    save(fig, out_dir, "nemchua_f2_main")


# --------------------------------------------------------------------------- #
# Figure 3 — the proposals are near chance, and the set improves anyway
# --------------------------------------------------------------------------- #
def fig_wrong(out_dir: str) -> None:
    run = pick("main_v2", "main")
    ladder = [(n, pick(f"ladder_n{n}", f"n_obs_{n}")) for n in (40, 60, 120, 300, 1000)]
    if run is None:
        print("  [skip] nemchua_f3_wrong"); return
    df = load(run)
    print(f"  [f3] {run}")
    mods = models_in(df)
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.0))

    # (a) precision vs chance
    ax = axes[0]
    stats = edit_stats(df)
    if stats.empty:
        stats = edit_stats_from_audit(run)
    if not stats.empty:
        xs = np.arange(len(stats))
        ax.bar(xs - 0.19, stats["chance"], 0.36, color=LIGHT, edgecolor="white",
               linewidth=0.4, label="uninformed editor")
        ax.bar(xs + 0.19, stats["precision"], 0.36, color=BLUE, edgecolor="white",
               linewidth=0.4, label="LLM proposer")
        top = float(max(stats["precision"].max(), stats["chance"].max()))
        for x, row in zip(xs, stats.itertuples()):
            ax.text(x, top * 1.06, f"{row.lift:.1f}× chance", ha="center", va="bottom",
                    fontsize=6.2, color=BLUE)
        ax.set_ylim(0, top * 1.55)
        ax.set_xticks(xs); ax.set_xticklabels([short(m) for m in stats["model_tag"]], fontsize=6.2)
        ax.set_ylabel("fraction of edits correct")
        ax.legend(loc="upper center", ncol=2, handlelength=1.2, columnspacing=1.0,
                  borderpad=0.1)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title("(a)  proposals are near chance", loc="left", fontweight="bold")

    # (b) but the truth enters the candidate set more often
    ax = axes[1]
    labels, base_v, llm_v, errs = [], [], [], []
    for model in mods:
        a = series(df, "probe", "truth_in_hypotheses", model)
        b = series(df, "probe_skel_only", "truth_in_hypotheses")
        d, lo, hi, p, n = paired(a, b)
        keys = sorted(set(a) & set(b))
        labels.append(short(model))
        base_v.append(np.mean([b[k] for k in keys])); llm_v.append(np.mean([a[k] for k in keys]))
        errs.append((d - lo, hi - d))
    xs = np.arange(len(labels))
    ax.bar(xs - 0.19, base_v, 0.36, color=LIGHT, edgecolor="white", linewidth=0.4, label="no LLM")
    ax.bar(xs + 0.19, llm_v, 0.36, color=BLUE, edgecolor="white", linewidth=0.4, label="LLM edits")
    for x, lo_, hi_ in zip(xs, base_v, llm_v):
        ax.annotate("", xy=(x + 0.19, hi_), xytext=(x - 0.19, lo_),
                    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.0))
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=6.2)
    ax.set_ylabel("true DAG is in the candidate set")
    ax.set_ylim(0, 1.0); ax.yaxis.grid(True); ax.set_axisbelow(True); ax.legend(loc="upper left")
    ax.set_title("(b)  one right guess is enough", loc="left", fontweight="bold")

    # (c) proposal-content ladder
    ax = axes[2]
    lad_run = pick("ladder_n60", "n_obs_60") or run
    lad = load(lad_run)
    rung_defs = [("probe_random_edits", "random\nedits", RED),
                 ("probe_skel_only", "no\nedits", GREY),
                 ("probe", "LLM\nedits", BLUE),
                 ("probe_oracle_edits", "perfect\nedits", GREEN)]
    names, vals, errs, colors = [], [], [], []
    for arm, label, color in rung_defs:
        sub = lad[lad["arm"] == arm]
        if sub.empty:
            continue
        if arm == "probe" and mods:
            sub = sub[sub["model_tag"] == models_in(lad)[0]]
        m, e = mean_ci(sub["directed_f1"])
        names.append(label); vals.append(m); errs.append(e); colors.append(color)
    ax.bar(range(len(names)), vals, 0.62, yerr=errs, color=colors, edgecolor="white",
           linewidth=0.4, error_kw=dict(ecolor=GREY, elinewidth=0.7))
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=6.2)
    ax.set_ylabel("directed-edge F1")
    if vals:
        ax.set_ylim(min(vals) - 0.12, min(1.02, max(vals) + 0.08))
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title("(c)  content matters, not width", loc="left", fontweight="bold")
    fig.tight_layout(w_pad=1.4)
    save(fig, out_dir, "nemchua_f3_wrong")


# --------------------------------------------------------------------------- #
# Figure 4 — why a wrong proposal is free
# --------------------------------------------------------------------------- #
def fig_mechanism(out_dir: str) -> None:
    run = pick("reserve_n60", "ladder_n60", "n_obs_60", "reserve_n300")
    main = pick("main_v2", "main")
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.05))

    # (a) the guard
    ax = axes[0]
    if run is not None:
        df = load(run)
        print(f"  [f4a] {run}")
        mods = models_in(df)
        pairs = [("probe_random_edits", "probe_random_edits_noreserve", "random\nedits", RED),
                 ("probe", "probe_noreserve", "LLM\nedits", BLUE),
                 ("probe_oracle_edits", "probe_oracle_edits_noreserve", "perfect\nedits", GREEN)]
        rows = []
        for on, off, label, color in pairs:
            a = df[df["arm"] == on]
            b = df[df["arm"] == off]
            if on == "probe" and mods:
                a = a[a["model_tag"] == mods[0]]; b = b[b["model_tag"] == mods[0]]
            if a.empty or b.empty:
                print(f"  [f4a] no paired data for {on}/{off}; column omitted")
                continue
            rows.append((label, color, mean_ci(a["directed_f1"]), mean_ci(b["directed_f1"])))
        xs = np.arange(len(rows))
        labels = [r[0] for r in rows]
        ax.bar(xs - 0.19, [r[3][0] for r in rows], 0.36, yerr=[r[3][1] for r in rows],
               color="none", edgecolor=[r[1] for r in rows], linewidth=1.0, hatch="////",
               label="guard off", error_kw=dict(ecolor=GREY, elinewidth=0.7))
        ax.bar(xs + 0.19, [r[2][0] for r in rows], 0.36, yerr=[r[2][1] for r in rows],
               color=[r[1] for r in rows], edgecolor="white", linewidth=0.4,
               label="guard on", error_kw=dict(ecolor=GREY, elinewidth=0.7))
        base = df[df["arm"] == "probe_skel_only"]["directed_f1"].mean()
        ax.axhline(base, color=DARK, lw=0.9, ls=(0, (3, 2)))
        ax.text(-0.45, base + 0.008, "no edits at all", fontsize=6, color=DARK, ha="left")
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=6.2)
        ax.set_ylabel("directed-edge F1")
        vals = [r[2][0] for r in rows] + [r[3][0] for r in rows]
        ax.set_ylim(min(vals) - 0.10, min(1.06, max(vals) + 0.12))
        ax.legend(loc="upper left", ncol=2, handlelength=1.4, columnspacing=1.0)
    else:
        print("  [skip] f4a: no reserve run")
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title("(a)  the guard bounds the damage", loc="left", fontweight="bold")

    # (b) posterior collapse
    ax = axes[1]
    if main is not None:
        st = steps_of(main)
        if not st.empty:
            print(f"  [f4b] {main}")
            for arm, color, label, ls in [("probe", BLUE, "NemChua", "-"),
                                          ("probe_random_sel", ORANGE, "random experiment", "-"),
                                          ("probe_no_update", RED, "no Bayes update", (0, (3, 2)))]:
                sub = st[st["arm"] == arm]
                if sub.empty:
                    continue
                wide = sub.pivot_table(index=["level", "seed", "model_tag"], columns="step",
                                       values="entropy_after_nats")
                wide = wide.reindex(columns=sorted(wide.columns)).ffill(axis=1)
                start = sub[sub["step"] == 1]["entropy_before_nats"].mean()
                xs = [0] + list(wide.columns)
                ys = [start] + [wide[c].mean() for c in wide.columns]
                ax.plot(xs, ys, color=color, label=label, ls=ls, marker="o",
                        markerfacecolor="white", markeredgewidth=1.1)
            ax.set_xlabel("interventions performed")
            ax.set_ylabel("posterior entropy (nats)")
            ax.set_xticks(range(0, 5)); ax.legend(loc="upper right")
            ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title("(b)  experiments collapse the posterior", loc="left", fontweight="bold")
    fig.tight_layout(w_pad=1.6)
    save(fig, out_dir, "nemchua_f4_mechanism")


# --------------------------------------------------------------------------- #
# Figure 5 — where the proposal channel earns its keep
# --------------------------------------------------------------------------- #
def fig_crossover(out_dir: str) -> None:
    runs = [(n, pick(f"ladder_n{n}", f"n_obs_{n}")) for n in (40, 60, 120, 300, 1000)]
    runs = [(n, r) for n, r in runs if r]
    if not runs:
        print("  [skip] nemchua_f5_crossover"); return
    print(f"  [f5] {[r for _, r in runs]}")
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.05))

    # (a) gain over the identical no-LLM pipeline
    ax = axes[0]
    all_models = sorted({m for _, r in runs for m in models_in(load(r))})
    for mi, model in enumerate(all_models):
        xs, ys, los, his = [], [], [], []
        for n, r in runs:
            df = load(r)
            if model not in models_in(df):
                continue
            d, lo, hi, p, k = paired(series(df, "probe", "directed_f1", model),
                                     series(df, "probe_skel_only", "directed_f1"))
            xs.append(n); ys.append(d); los.append(d - lo); his.append(hi - d)
        if not xs:
            continue
        color = [BLUE, ORANGE, GREEN, PURPLE, RED][mi % 5]
        ax.errorbar(xs, ys, yerr=[los, his], color=color, marker="o",
                    markerfacecolor="white", markeredgewidth=1.1, label=short(model))
    ax.axhline(0, color=DARK, lw=0.8)
    ax.fill_between([30, 1300], 0, 0.09, color=BLUE, alpha=0.05, lw=0)
    ax.fill_between([30, 1300], -0.06, 0, color=RED, alpha=0.05, lw=0)
    ax.text(42, 0.062, "LLM edits help", fontsize=6, color=BLUE)
    ax.text(42, -0.043, "LLM edits hurt", fontsize=6, color=RED)
    ax.set_xscale("log"); ax.set_xlim(33, 1300)
    ax.set_xticks([40, 60, 120, 300, 1000]); ax.set_xticklabels(["40", "60", "120", "300", "1000"])
    ax.set_xlabel("observational sample size")
    ax.set_ylabel("F1 gain over the same\npipeline without the LLM")
    ax.legend(loc="upper right"); ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title("(a)  the proposer earns its keep only when data is scarce",
                 loc="left", fontweight="bold", fontsize=7)

    # (b) and the lift explains it
    ax = axes[1]
    for mi, model in enumerate(all_models):
        xs, ys = [], []
        for n, r in runs:
            stats = edit_stats(load(r))
            if stats.empty:
                stats = edit_stats_from_audit(r)
            if stats.empty or model not in set(stats["model_tag"]):
                continue
            xs.append(n); ys.append(float(stats[stats["model_tag"] == model]["lift"].iloc[0]))
        if not xs:
            continue
        color = [BLUE, ORANGE, GREEN, PURPLE, RED][mi % 5]
        ax.plot(xs, ys, color=color, marker="s", markerfacecolor="white",
                markeredgewidth=1.1, label=short(model))
    ax.axhline(1.0, color=DARK, lw=0.8, ls=(0, (3, 2)))
    ax.text(1150, 1.02, "chance", fontsize=6, color=DARK, ha="right")
    ax.set_xscale("log"); ax.set_xlim(33, 1300)
    ax.set_xticks([40, 60, 120, 300, 1000]); ax.set_xticklabels(["40", "60", "120", "300", "1000"])
    ax.set_xlabel("observational sample size")
    ax.set_ylabel("edit precision ÷ chance")
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title("(b)  because its edits stop beating chance", loc="left",
                 fontweight="bold", fontsize=7)
    fig.tight_layout(w_pad=1.6)
    save(fig, out_dir, "nemchua_f5_crossover")


# --------------------------------------------------------------------------- #
# Figure 6 — capability sweep and the semantic condition
# --------------------------------------------------------------------------- #
def fig_models(out_dir: str) -> None:
    sweep = pick("models_n60", "models_n300")
    sem = pick("semantic_n60", "semantic")
    if sweep is None and sem is None:
        print("  [skip] nemchua_f6_models: neither sweep nor semantic run present"); return
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.05))

    # (a) capability sweep
    ax = axes[0]
    if sweep is not None:
        df = load(sweep)
        print(f"  [f6a] {sweep}")
        stats = edit_stats(df)
        if stats.empty:
            stats = edit_stats_from_audit(sweep)
        stats = stats.sort_values("lift")
        base = df[df["arm"] == "probe_skel_only"]["directed_f1"].mean()
        gains, labels, errs, lifts = [], [], [], []
        for row in stats.itertuples():
            d, lo, hi, p, n = paired(series(df, "probe", "directed_f1", row.model_tag),
                                     series(df, "probe_skel_only", "directed_f1"))
            gains.append(d); errs.append((d - lo, hi - d)); labels.append(short(row.model_tag))
            lifts.append(row.lift)
        xs = np.arange(len(labels))
        ax.bar(xs, gains, 0.6, yerr=np.array(errs).T, color=BLUE, edgecolor="white",
               linewidth=0.4, error_kw=dict(ecolor=GREY, elinewidth=0.7))
        ax.axhline(0, color=DARK, lw=0.8)
        for x, g, lf in zip(xs, gains, lifts):
            ax.text(x, g + (0.004 if g >= 0 else -0.004), f"{lf:.1f}×",
                    ha="center", va="bottom" if g >= 0 else "top", fontsize=6, color=GREY)
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=6.2, rotation=18, ha="right")
        ax.set_ylabel("F1 gain over the same\npipeline without the LLM")
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title("(a)  every proposer, same architecture", loc="left", fontweight="bold")

    # (b) named vs anonymized variables
    ax = axes[1]
    if sem is not None:
        df = pd.read_csv(os.path.join(sem, "episodes.csv"))
        df = df[df["status"] == "success"]
        print(f"  [f6b] {sem}")
        sub = df[df["arm"] == "probe"]
        rows = []
        for (model, cond), g in sub.groupby(["model_tag", "condition"]):
            proposed = (g["repair_remove"] + g["repair_add"]).sum()
            correct = (g["edits_correct_remove"] + g["edits_correct_add"]).sum()
            rows.append(dict(model=short(model), condition=cond,
                             precision=correct / max(proposed, 1)))
        piv = pd.DataFrame(rows).pivot_table(index="model", columns="condition", values="precision")
        if {"named", "anon"} <= set(piv.columns):
            piv = piv.sort_values("named")
            xs = np.arange(len(piv))
            ax.bar(xs - 0.19, piv["anon"], 0.36, color=LIGHT, edgecolor="white",
                   linewidth=0.4, label="X0 … Xd")
            ax.bar(xs + 0.19, piv["named"], 0.36, color=PURPLE, edgecolor="white",
                   linewidth=0.4, label="real variable names")
            ax.set_xticks(xs); ax.set_xticklabels(piv.index, fontsize=6.2, rotation=18, ha="right")
            ax.set_ylabel("fraction of edits correct")
            ax.legend(loc="upper left")
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title("(b)  does domain knowledge help?", loc="left", fontweight="bold")
    fig.tight_layout(w_pad=1.6)
    save(fig, out_dir, "nemchua_f6_models")


def main() -> int:
    global STUDY_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-dir", default="study2")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()
    STUDY_DIR = args.study_dir
    os.makedirs(args.out_dir, exist_ok=True)
    fig_main(args.out_dir)
    fig_wrong(args.out_dir)
    fig_mechanism(args.out_dir)
    fig_crossover(args.out_dir)
    fig_models(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
