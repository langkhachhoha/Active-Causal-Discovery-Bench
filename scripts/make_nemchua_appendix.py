#!/usr/bin/env python
"""NemChua appendix figures — everything the runs support that the main text has no room for.

    python scripts/make_nemchua_appendix.py --study-dir study2_new --out-dir figures

Written figures
    nemchua_a1_edits      what each proposer actually proposes: removals vs additions
    nemchua_a2_chain      where a better proposal stops turning into a better answer
    nemchua_a3_semantic   which domains the names help in
    nemchua_a4_ladder     every arm across sample size, and across graph size
    nemchua_a5_cost       accuracy against tokens spent
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

BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#c8352b"
GREEN, PURPLE, GOLD = "#2e8b6f", "#7b5ea7", "#c99a2e"
GREY, DARK, LIGHT = "#6f6e6a", "#1a1a1a", "#b9b8b2"
GRID, AXIS = "#e3e2dc", "#bfbeb4"

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

STUDY_DIR = "study2_new"
SHORT = {
    "qwen3-coder-30b-a3b-instruct": "qwen3-30b", "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
    "claude-haiku-4.5": "haiku-4.5", "gemini-3-flash-preview": "gemini-3-flash",
    "gpt-5.4-mini": "gpt-5.4-mini",
}
# the chain panels are too narrow for full names and truncation collides the two gpt models
TINY = {
    "qwen3-coder-30b-a3b-instruct": "qwen3", "gpt-4o-mini-2024-07-18": "4o-mini",
    "claude-haiku-4.5": "haiku", "gemini-3-flash-preview": "gemini", "gpt-5.4-mini": "5.4-mini",
}
ORDER = ["gpt-4o-mini-2024-07-18", "qwen3-coder-30b-a3b-instruct", "gpt-5.4-mini",
         "claude-haiku-4.5", "gemini-3-flash-preview"]
def short(t): return SHORT.get(t, t[:14])


def exists(run): return os.path.exists(os.path.join(STUDY_DIR, run, "episodes.csv"))


def load(run):
    """`<run>_fix` for the arms it re-ran, plus the arms only `<run>` has."""
    frames = []
    if exists(f"{run}_fix"):
        d = pd.read_csv(os.path.join(STUDY_DIR, f"{run}_fix", "episodes.csv"))
        frames.append(d[d["status"] == "success"])
    if exists(run):
        d = pd.read_csv(os.path.join(STUDY_DIR, run, "episodes.csv"))
        d = d[d["status"] == "success"]
        if frames:
            d = d[~d["arm"].isin(set(frames[0]["arm"]))]
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def audit(run):
    p = os.path.join(STUDY_DIR, run, "analysis", "edit_audit.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


def save(fig, out_dir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  [written] {out_dir}/{name}.pdf")


def mean_ci(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if len(v) < 2:
        return (float(v.mean()) if len(v) else np.nan), 0.0
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))


def series(df, arm, metric="directed_f1", model=None):
    s = df[df["arm"] == arm]
    if model is not None and (s["model_tag"] != "none").any():
        s = s[s["model_tag"] == model]
    return {(int(r.level), int(r.seed)): float(getattr(r, metric)) for r in s.itertuples()}


def contrast(a, b):
    k = sorted(set(a) & set(b))
    if not k:
        return None
    d = np.array([a[i] - b[i] for i in k])
    rng = np.random.default_rng(0)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)])
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), \
        (wilcoxon(d).pvalue if np.any(d != 0) else 1.0)


# --------------------------------------------------------------------------- #
def fig_edits(out_dir):
    a = audit("models_n60")
    if a.empty:
        print("  [skip] a1"); return
    print("  [a1] models_n60 audit")
    models = [m for m in ORDER if m in set(a["model_tag"])]
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.1))

    rows = []
    for m in models:
        g = a[a["model_tag"] == m]
        rows.append(dict(
            m=m, n_rm=g["n_remove"].mean(), n_add=g["n_add"].mean(),
            p_rm=g["correct_remove"].sum() / max(g["n_remove"].sum(), 1),
            p_add=g["correct_add"].sum() / max(g["n_add"].sum(), 1),
            c_rm=g["chance_remove"].mean(), c_add=g["chance_add"].mean(),
            fn=g["correct_add"].sum() / max(g["pc_fn"].sum(), 1),
        ))

    # (a) how many edits, of which kind
    ax = axes[0]
    xs = np.arange(len(rows))
    ax.bar(xs, [r["n_rm"] for r in rows], 0.62, color=RED, edgecolor="white", linewidth=0.4,
           label="removals")
    ax.bar(xs, [r["n_add"] for r in rows], 0.62, bottom=[r["n_rm"] for r in rows],
           color=BLUE, edgecolor="white", linewidth=0.4, label="additions")
    ax.axhline(1.75, color=DARK, lw=0.9, ls=(0, (3, 2)))
    top = max(r["n_rm"] + r["n_add"] for r in rows)
    ax.set_ylim(0, top * 1.28)
    ax.annotate("adjacencies PC\nactually missed", xy=(0.9, 1.75), xytext=(1.0, top * 0.60),
                fontsize=5.6, color=DARK, ha="center", linespacing=1.15,
                arrowprops=dict(arrowstyle="->", color=DARK, lw=0.7))
    ax.set_xticks(xs); ax.set_xticklabels([short(r["m"]) for r in rows], fontsize=6,
                                          rotation=18, ha="right")
    ax.set_ylabel("edits proposed per instance"); ax.legend(loc="upper right", fontsize=6)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title("(a)  how much they propose", loc="left", fontweight="bold")

    # (b) precision of each kind against its own chance rate
    ax = axes[1]
    w = 0.36
    ax.bar(xs - w / 2, [r["p_rm"] for r in rows], w, color=RED, edgecolor="white",
           linewidth=0.4, label="removals")
    ax.bar(xs + w / 2, [r["p_add"] for r in rows], w, color=BLUE, edgecolor="white",
           linewidth=0.4, label="additions")
    for x, r in zip(xs, rows):
        ax.plot([x - w, x], [r["c_rm"]] * 2, color=DARK, lw=1.0)
        ax.plot([x, x + w], [r["c_add"]] * 2, color=DARK, lw=1.0)
    ax.plot([], [], color=DARK, lw=1.0, label="chance")
    ax.set_xticks(xs); ax.set_xticklabels([short(r["m"]) for r in rows], fontsize=6,
                                          rotation=18, ha="right")
    ax.set_ylabel("fraction correct")
    ax.legend(loc="upper left", ncol=3, fontsize=6, columnspacing=0.9, handlelength=1.2)
    ax.set_ylim(0, 0.48)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title("(b)  and how often they are right", loc="left", fontweight="bold")

    # (c) selectivity: proposing less is proposing better
    ax = axes[2]
    for r in rows:
        ax.scatter(r["n_rm"] + r["n_add"], r["p_rm"] * r["n_rm"] / max(r["n_rm"] + r["n_add"], 1e-9)
                   + r["p_add"] * r["n_add"] / max(r["n_rm"] + r["n_add"], 1e-9),
                   s=34, color=BLUE, zorder=3, edgecolor="white", linewidth=0.8)
        total = r["n_rm"] + r["n_add"]
        y = (r["p_rm"] * r["n_rm"] + r["p_add"] * r["n_add"]) / max(total, 1e-9)
        # the two weakest proposers sit almost on top of each other
        dy = -13 if r["m"] == "gpt-4o-mini-2024-07-18" else 8
        ax.annotate(short(r["m"]), (total, y), textcoords="offset points", xytext=(0, dy),
                    ha="center", fontsize=5.8, color=GREY)
    ax.set_xlabel("edits proposed per instance")
    ax.set_ylabel("fraction correct")
    ax.set_xlim(2, 8); ax.set_ylim(0, 0.42)
    ax.yaxis.grid(True); ax.xaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title("(c)  restraint, not volume", loc="left", fontweight="bold")
    fig.tight_layout(w_pad=1.4)
    save(fig, out_dir, "nemchua_a1_edits")


# --------------------------------------------------------------------------- #
def fig_chain(out_dir):
    d = load("models_n60")
    a = audit("models_n60")
    if d.empty or a.empty:
        print("  [skip] a2"); return
    print("  [a2] models_n60")
    models = [m for m in ORDER if m in set(d["model_tag"])]
    stages, labels = [], ["edits\ncorrect", "PC's misses\nrecovered", "truth in\ncandidate set",
                          "best F1\navailable", "F1\nachieved"]
    base = d[d["arm"] == "probe_skel_only"]
    for m in models:
        g = a[a["model_tag"] == m]
        p = d[(d["arm"] == "probe") & (d["model_tag"] == m)]
        stages.append([
            g[["correct_remove", "correct_add"]].sum().sum() / max(g[["n_remove", "n_add"]].sum().sum(), 1),
            g["correct_add"].sum() / max(g["pc_fn"].sum(), 1),
            p["truth_in_hypotheses"].mean(),
            p["best_f1_in_hypotheses"].mean(),
            p["directed_f1"].mean(),
        ])
    ref = [np.nan, 0.0, base["truth_in_hypotheses"].mean(),
           base["best_f1_in_hypotheses"].mean(), base["directed_f1"].mean()]

    # Two rows rather than five squeezed panels: each stage has its own y range, and at
    # one-fifth of the text width the tick labels were unreadable and the key collided
    # with them. The layout is 3 + 2 with the legend in the free sixth cell.
    fig, axes = plt.subplots(2, 3, figsize=(6.9, 4.0))
    flat = axes.ravel()
    for j, lab in enumerate(labels):
        ax = flat[j]
        vals = [s[j] for s in stages]
        xs = np.arange(len(models))
        ax.bar(xs, vals, 0.62, color=BLUE, edgecolor="white", linewidth=0.4)
        if np.isfinite(ref[j]):
            ax.axhline(ref[j], color=RED, lw=1.1, ls=(0, (3, 2)))
        lo, hi = min(vals), max(vals)
        pad = max((hi - lo) * 0.35, 0.02)
        ax.set_ylim(max(0, min(lo, ref[j] if np.isfinite(ref[j]) else lo) - pad), hi + pad)
        ax.set_xticks(xs)
        ax.set_xticklabels([TINY.get(m, short(m)) for m in models], fontsize=6.4,
                           rotation=32, ha="right")
        ax.set_title(f"({chr(97 + j)}) " + lab.replace("\n", " "), loc="left",
                     fontsize=7.2, fontweight="bold")
        ax.yaxis.grid(True); ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=6.6)
    for ax in (flat[0], flat[3]):
        ax.set_ylabel("value", fontsize=7)
    key = flat[5]
    key.axis("off")
    key.plot([], [], color=BLUE, lw=6, solid_capstyle="butt", label="proposer, ordered by edit precision")
    key.plot([], [], color=RED, lw=1.1, ls=(0, (3, 2)), label="no edits at all")
    key.legend(loc="center", fontsize=7, frameon=False, handlelength=1.8)
    fig.suptitle("proposal quality propagates through four stages, then stops",
                 fontsize=8.2, fontweight="bold", x=0.012, ha="left", y=1.0)
    fig.tight_layout(w_pad=1.6, h_pad=1.8)
    save(fig, out_dir, "nemchua_a2_chain")


# --------------------------------------------------------------------------- #
def fig_semantic(out_dir):
    d = load("semantic_n60")
    if d.empty:
        print("  [skip] a3"); return
    print("  [a3] semantic_n60")
    p = d[d["arm"] == "probe"].copy()
    p["proposed"] = p["repair_remove"] + p["repair_add"]
    p["correct"] = p["edits_correct_remove"] + p["edits_correct_add"]
    graphs = ["cancer", "earthquake", "survey", "asia", "sachs"]
    graphs = [g for g in graphs if g in set(p["graph"])]
    models = [m for m in ORDER if m in set(p["model_tag"])]

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.25),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    # (a) precision gain from names, per structure, pooled over models
    ax = axes[0]
    xs = np.arange(len(graphs))
    an, na = [], []
    for g in graphs:
        gg = p[p["graph"] == g]
        for cond, box in (("anon", an), ("named", na)):
            s = gg[gg["condition"] == cond]
            box.append(s["correct"].sum() / max(s["proposed"].sum(), 1))
    ax.bar(xs - 0.19, an, 0.36, color=LIGHT, edgecolor="white", linewidth=0.4, label="X0 … Xd")
    ax.bar(xs + 0.19, na, 0.36, color=PURPLE, edgecolor="white", linewidth=0.4,
           label="real names")
    for x, (u, v) in enumerate(zip(an, na)):
        if v > u:
            ax.annotate("", xy=(x + 0.19, v + 0.012), xytext=(x - 0.19, u + 0.012),
                        arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.9))
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{g}\n$d$={int(p[p.graph == g].d.iloc[0])}" for g in graphs],
                       fontsize=6, linespacing=1.2)
    ax.set_ylabel("fraction of edits correct")
    ax.legend(loc="upper left", fontsize=6)
    ax.set_ylim(0, 0.48)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title("(a)  names help in everyday domains, not specialist ones",
                 loc="left", fontweight="bold", fontsize=7)

    # (b) per model x condition, pooled over structures
    ax = axes[1]
    xs = np.arange(len(models))
    an, na = [], []
    for m in models:
        gg = p[p["model_tag"] == m]
        for cond, box in (("anon", an), ("named", na)):
            s = gg[gg["condition"] == cond]
            box.append(s["correct"].sum() / max(s["proposed"].sum(), 1))
    ax.bar(xs - 0.19, an, 0.36, color=LIGHT, edgecolor="white", linewidth=0.4, label="X0 … Xd")
    ax.bar(xs + 0.19, na, 0.36, color=PURPLE, edgecolor="white", linewidth=0.4,
           label="real names")
    ax.set_xticks(xs); ax.set_xticklabels([short(m) for m in models], fontsize=6,
                                          rotation=18, ha="right")
    ax.set_ylabel("fraction of edits correct")
    ax.set_ylim(0, 0.85)
    ax.legend(loc="upper left", fontsize=6)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title("(b)  every proposer, pooled over structures", loc="left",
                 fontweight="bold", fontsize=7)
    fig.tight_layout(w_pad=1.5)
    save(fig, out_dir, "nemchua_a3_semantic")


# --------------------------------------------------------------------------- #
def fig_ladder(out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.15))

    ax = axes[0]
    arms = [("probe_oracle_edits", None, "perfect edits", GREEN, "-"),
            ("probe_stat_edits", None, "statistical ranker", ORANGE, "-"),
            ("probe", "gpt-4o-mini-2024-07-18", "NemChua (gpt-4o-mini)", BLUE, "-"),
            ("probe_random_edits", None, "random edits (cap)", RED, "-"),
            ("probe_skel_only", None, "no edits", GREY, "-"),
            ("probe_mec_only", None, "PC equivalence class", LIGHT, (0, (3, 2))),
            ("pc_greedy_meek", None, "PC + greedy", DARK, (0, (1, 1.6)))]
    ns = [40, 60, 120, 300, 1000]
    for arm, model, label, color, ls in arms:
        xs, ys, es = [], [], []
        for n in ns:
            d = load(f"ladder_n{n}")
            if d.empty:
                continue
            s = d[d["arm"] == arm]
            if model is not None and (s["model_tag"] != "none").any():
                s = s[s["model_tag"] == model]
            if s.empty:
                continue
            m, e = mean_ci(s["directed_f1"])
            xs.append(n); ys.append(m); es.append(e)
        if xs:
            ax.errorbar(xs, ys, yerr=es, color=color, ls=ls, marker="o", label=label,
                        markerfacecolor="white", markeredgewidth=1.0, markersize=3.4)
    ax.set_xscale("log"); ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("observational sample size"); ax.set_ylabel("directed-edge F1")
    ax.legend(loc="lower right", fontsize=5.6, ncol=1, labelspacing=0.28)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title("(a)  every arm against sample size ($d{=}6,8$)", loc="left",
                 fontweight="bold", fontsize=7)

    ax = axes[1]
    d = load("main_v2")
    if not d.empty:
        for arm, model, label, color, ls in arms:
            xs, ys, es = [], [], []
            for lv in sorted(d["level"].unique()):
                s = d[(d["arm"] == arm) & (d["level"] == lv)]
                if model is not None and (s["model_tag"] != "none").any():
                    s = s[s["model_tag"] == model]
                if s.empty:
                    continue
                m, e = mean_ci(s["directed_f1"])
                xs.append(int(s["d"].iloc[0])); ys.append(m); es.append(e)
            if xs:
                ax.errorbar(xs, ys, yerr=es, color=color, ls=ls, marker="o", label=label,
                            markerfacecolor="white", markeredgewidth=1.0, markersize=3.4)
        ax.set_xticks([4, 6, 8, 10])
        ax.set_xlabel("number of variables $d$"); ax.set_ylabel("directed-edge F1")
        ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title("(b)  and against graph size ($n_{\\mathrm{obs}}{=}300$)", loc="left",
                 fontweight="bold", fontsize=7)
    fig.tight_layout(w_pad=1.5)
    save(fig, out_dir, "nemchua_a4_ladder")


# --------------------------------------------------------------------------- #
def fig_cost(out_dir):
    d = load("main_v2")
    if d.empty:
        print("  [skip] a5"); return
    print("  [a5] main_v2")
    fig, ax = plt.subplots(figsize=(3.9, 2.5))
    pts = [("llm_e2e", "gpt-4o-mini-2024-07-18", RED, "o"),
           ("llm_e2e", "qwen3-coder-30b-a3b-instruct", RED, "s"),
           ("probe_llm_graphs", "gpt-4o-mini-2024-07-18", ORANGE, "o"),
           ("probe_llm_graphs", "qwen3-coder-30b-a3b-instruct", ORANGE, "s"),
           ("probe", "gpt-4o-mini-2024-07-18", BLUE, "o"),
           ("probe", "qwen3-coder-30b-a3b-instruct", BLUE, "s")]
    orig = pd.read_csv(os.path.join(STUDY_DIR, "main_v2", "episodes.csv"))
    orig = orig[orig["status"] == "success"]
    seen = set()
    for arm, model, color, marker in pts:
        s = d[(d["arm"] == arm) & (d["model_tag"] == model)]
        o = orig[(orig["arm"] == arm) & (orig["model_tag"] == model)]
        if s.empty or o.empty:
            continue
        tok = o["total_tokens"].mean()
        f1, e = mean_ci(s["directed_f1"])
        lab = {"llm_e2e": "LLM end-to-end", "probe_llm_graphs": "LLM writes the DAG",
               "probe": "NemChua"}[arm]
        ax.errorbar(tok, f1, yerr=e, color=color, marker=marker, markersize=5,
                    markerfacecolor="white", markeredgewidth=1.2,
                    label=lab if lab not in seen else None)
        seen.add(lab)
    # Three zero-token references, two of which sit within 0.03 F1 of each other, so the
    # labels are staggered and drawn on an opaque patch rather than across their own lines.
    # Three zero-token references, two of which land within 0.03 F1 of each other. Inline
    # labels cannot be separated at that spacing, so they go in the key instead.
    refs = [("probe_stat_edits", ORANGE, "statistical ranker"),
            ("probe_random_edits", GREEN, "random edits (cap)"),
            ("pc_greedy_meek", GREY, "PC + greedy")]
    for arm, color, label in refs:
        s = d[d["arm"] == arm]
        if s.empty:
            continue
        f1, _ = mean_ci(s["directed_f1"])
        ax.axhline(f1, color=color, lw=0.9, ls=(0, (3, 2)),
                   label=f"{label} \u2014 no model")
    ax.set_xscale("log"); ax.set_xlim(6e2, 1.1e5)
    ax.set_xlabel("tokens per episode"); ax.set_ylabel("directed-edge F1")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower left", fontsize=6, labelspacing=0.4, handlelength=1.9,
              borderaxespad=0.5)
    ax.yaxis.grid(True); ax.xaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title("accuracy against tokens spent", loc="left", fontweight="bold")
    fig.tight_layout()
    save(fig, out_dir, "nemchua_a5_cost")


def main():
    global STUDY_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-dir", default="study2_new")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()
    STUDY_DIR = args.study_dir
    os.makedirs(args.out_dir, exist_ok=True)
    fig_edits(args.out_dir)
    fig_chain(args.out_dir)
    fig_semantic(args.out_dir)
    fig_ladder(args.out_dir)
    fig_cost(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
