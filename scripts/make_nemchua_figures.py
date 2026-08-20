#!/usr/bin/env python
"""NemChua (study 2) figures.

    python scripts/make_nemchua_figures.py --study-dir study2_new --out-dir figures

Runs replayed through the corrected enumeration live in `<run>_fix`; arms that the
correction cannot touch (the end-to-end agent, the whole-graph proposer) are carried over
from `<run>`. `load_merged` does that join, so every panel reads one consistent method.

Written figures
    nemchua_f2_main       headline accuracy, and accuracy per experiment
    nemchua_f3_credit     who is actually doing the work
    nemchua_f4_mechanism  the guard, and the posterior collapse
    nemchua_f5_when       the two conditions under which a proposer is worth anything
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
    "claude-haiku-4.5": "haiku-4.5", "claude-sonnet-4.6": "sonnet-4.6",
    "gemini-3-flash-preview": "gemini-3-flash", "gpt-5.4-mini": "gpt-5.4-mini",
}
def short(tag: str) -> str:
    return SHORT.get(tag, tag[:14])


def exists(run: str) -> bool:
    return os.path.exists(os.path.join(STUDY_DIR, run, "episodes.csv"))


def load_merged(run: str) -> pd.DataFrame:
    """`<run>_fix` for every arm it re-ran, plus the arms only `<run>` has."""
    frames = []
    if exists(f"{run}_fix"):
        fix = pd.read_csv(os.path.join(STUDY_DIR, f"{run}_fix", "episodes.csv"))
        frames.append(fix[fix["status"] == "success"])
    if exists(run):
        base = pd.read_csv(os.path.join(STUDY_DIR, run, "episodes.csv"))
        base = base[base["status"] == "success"]
        if frames:
            base = base[~base["arm"].isin(set(frames[0]["arm"]))]
        frames.append(base)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def steps_of(run: str) -> pd.DataFrame:
    for name in (f"{run}_fix", run):
        path = os.path.join(STUDY_DIR, name, "steps.csv")
        if os.path.exists(path):
            return pd.read_csv(path)
    return pd.DataFrame()


def save(fig, out_dir: str, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  [written] {out_dir}/{name}.pdf")


def series(df, arm, metric="directed_f1", model=None):
    sub = df[df["arm"] == arm]
    if model is not None and (sub["model_tag"] != "none").any():
        sub = sub[sub["model_tag"] == model]
    return {(int(r.level), int(r.seed)): float(getattr(r, metric)) for r in sub.itertuples()}


def contrast(a, b):
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    d = np.array([a[k] - b[k] for k in keys])
    rng = np.random.default_rng(0)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)])
    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return d.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5), p, len(d)


def mean_ci(values):
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if len(v) < 2:
        return (float(v.mean()) if len(v) else np.nan), 0.0
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))


def models_in(df):
    return sorted(m for m in df["model_tag"].unique() if m != "none")


def precision_of(df, arm="probe"):
    """Edit precision per model, straight from what each episode records."""
    need = {"repair_remove", "repair_add", "edits_correct_remove", "edits_correct_add"}
    if not need <= set(df.columns):
        return {}
    sub = df[df["arm"] == arm].dropna(subset=list(need))
    out = {}
    for model, g in sub.groupby("model_tag"):
        proposed = (g["repair_remove"] + g["repair_add"]).sum()
        correct = (g["edits_correct_remove"] + g["edits_correct_add"]).sum()
        out[model] = correct / max(proposed, 1)
    return out


# --------------------------------------------------------------------------- #
def fig_main(out_dir):
    df = load_merged("main_v2")
    if df.empty:
        print("  [skip] f2"); return
    print(f"  [f2] main_v2 ({len(df)} episodes)")
    mods = models_in(df)
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.1), gridspec_kw={"width_ratios": [1.45, 1]})

    order = [("oracle", "oracle", False), ("probe", "NemChua", True),
             ("probe_random_edits", "random\nedits", False),
             ("probe_skel_only", "no edits", False),
             ("pc_greedy_meek", "PC +\ngreedy", False),
             ("probe_llm_graphs", "LLM writes\nthe DAG", True),
             ("llm_e2e", "LLM agent\nend-to-end", True)]
    ax = axes[0]
    width = 0.36
    for mi, model in enumerate(mods):
        xs, hs, es = [], [], []
        for i, (arm, _, is_llm) in enumerate(order):
            sub = df[df["arm"] == arm]
            if is_llm:
                sub = sub[sub["model_tag"] == model]
            m, e = mean_ci(sub["directed_f1"])
            xs.append(i + (mi - (len(mods) - 1) / 2) * width); hs.append(m); es.append(e)
        ax.bar(xs, hs, width * 0.92, yerr=es, color=[BLUE, ORANGE][mi % 2], edgecolor="white",
               linewidth=0.4, label=short(model), error_kw=dict(ecolor=GREY, elinewidth=0.7))
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([lab for _, lab, _ in order], fontsize=6.2)
    ax.set_ylabel("directed-edge F1"); ax.set_ylim(0, 1.08)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.legend(loc="upper right", ncol=2, fontsize=6)
    ax.set_title("(a)  80 paired instances, same budget", loc="left", fontweight="bold")

    ax = axes[1]
    st = steps_of("main_v2")
    if not st.empty and "map_directed_f1_after" in st.columns:
        for arm, model, color, label in [("probe", mods[0] if mods else None, BLUE, "NemChua"),
                                         ("pc_greedy_meek", None, GREY, "PC + greedy")]:
            sub = st[st["arm"] == arm]
            if model is not None and (sub["model_tag"] != "none").any():
                sub = sub[sub["model_tag"] == model]
            sub = sub.dropna(subset=["map_directed_f1_after"])
            if sub.empty:
                continue
            wide = sub.pivot_table(index=["level", "seed"], columns="step",
                                   values="map_directed_f1_after")
            wide = wide.reindex(columns=sorted(wide.columns)).ffill(axis=1)
            ys = [wide[c].mean() for c in wide.columns]
            es = [1.96 * wide[c].std(ddof=1) / np.sqrt(wide[c].count()) for c in wide.columns]
            ax.errorbar(list(wide.columns), ys, yerr=es, color=color, marker="o", label=label,
                        markerfacecolor="white", markeredgewidth=1.1)
        final = df[df["arm"] == "pc_greedy_meek"]["directed_f1"].mean()
        ax.axhline(final, color=GREY, lw=0.8, ls=(0, (3, 2)))
        ax.text(3.9, final + 0.004, "PC + greedy, full budget", fontsize=6, color=GREY, ha="right")
        ax.set_xlabel("interventions performed"); ax.set_ylabel("directed-edge F1")
        ax.set_xticks(range(1, 5)); ax.legend(loc="lower right")
        ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title("(b)  accuracy per experiment", loc="left", fontweight="bold")
    fig.tight_layout(w_pad=1.6)
    save(fig, out_dir, "nemchua_f2_main")


# --------------------------------------------------------------------------- #
def fig_credit(out_dir):
    df = load_merged("main_v2")
    sweep = load_merged("models_n60")
    if df.empty:
        print("  [skip] f3"); return
    print("  [f3] main_v2 + models_n60")
    mods = models_in(df)
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.15),
                             gridspec_kw={"width_ratios": [0.95, 1.05, 1.25]})

    # (a) proposal-content ladder
    ax = axes[0]
    rungs = [("probe_skel_only", None, "no\nedits", GREY),
             ("probe_random_edits", None, "random\nedits", RED),
             ("probe", mods[0] if mods else None, "LLM\nedits", BLUE),
             ("probe_oracle_edits", None, "perfect\nedits", GREEN)]
    names, vals, errs, colors = [], [], [], []
    for arm, model, label, color in rungs:
        sub = df[df["arm"] == arm]
        if model is not None:
            sub = sub[sub["model_tag"] == model]
        if sub.empty:
            continue
        m, e = mean_ci(sub["directed_f1"])
        names.append(label); vals.append(m); errs.append(e); colors.append(color)
    ax.bar(range(len(names)), vals, 0.62, yerr=errs, color=colors, edgecolor="white",
           linewidth=0.4, error_kw=dict(ecolor=GREY, elinewidth=0.7))
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=6.2)
    ax.set_ylabel("directed-edge F1")
    ax.set_ylim(min(vals) - 0.05, 1.03)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title("(a)  random edits match the LLM", loc="left", fontweight="bold", fontsize=7)

    # (b) five proposers: precision spans 4x, accuracy does not move
    ax = axes[1]
    if not sweep.empty:
        prec = precision_of(sweep)
        base = series(sweep, "probe_skel_only")
        rnd = sweep[sweep["arm"] == "probe_random_edits"]["directed_f1"].mean()
        nb = sweep[sweep["arm"] == "probe_skel_only"]["directed_f1"].mean()
        pts = []
        for model in models_in(sweep):
            c = contrast(series(sweep, "probe", model=model), base)
            if c is None or model not in prec:
                continue
            pts.append((prec[model] * 100, c[0], (c[0] - c[1], c[2] - c[0]), short(model)))
        pts.sort()
        ax.errorbar([p[0] for p in pts], [p[1] for p in pts],
                    yerr=np.array([p[2] for p in pts]).T, fmt="o", color=BLUE,
                    markerfacecolor="white", markeredgewidth=1.1, ecolor=LIGHT, elinewidth=0.8)
        for i, (x, y, _, name) in enumerate(pts):
            ax.annotate(name, (x, y), textcoords="offset points",
                        xytext=(0, 9 if i in (0, 2, 4) else -15), ha="center",
                        fontsize=5.8, color=GREY)
        ax.axhline(rnd - nb, color=RED, lw=1.0, ls=(0, (3, 2)))
        ax.axhline(0, color=DARK, lw=0.8)
        ax.set_xlabel("edits the proposer got right (%)")
        ax.set_ylabel("F1 gain over no edits")
        ax.set_xlim(2, 41); ax.set_ylim(-0.055, 0.115)
        ax.text(3.2, rnd - nb + 0.003, "random editor", fontsize=6, color=RED,
                ha="left", va="bottom")
        ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title("(b)  4$\\times$ better proposals, same answer", loc="left",
                 fontweight="bold", fontsize=7)

    # (c) what each component is worth
    ax = axes[2]
    m0 = mods[0] if mods else None
    items = []
    for label, a, b, ma, mb, color in [
        ("Bayes update", "probe", "probe_no_update", m0, m0, DARK),
        ("EIG vs random", "probe", "probe_random_sel", m0, m0, DARK),
        ("the guard", "probe", "probe_noreserve", m0, m0, DARK),
        ("BIC weights", "probe", "probe_no_bic", m0, m0, DARK),
        ("EIG vs max-degree", "probe", "probe_maxdeg_sel", m0, m0, LIGHT),
        ("LLM vs random edits", "probe", "probe_random_edits", m0, None, BLUE),
    ]:
        c = contrast(series(df, a, model=ma), series(df, b, model=mb))
        if c:
            items.append((label, c[0], c[0] - c[1], c[2] - c[0], color, c[3]))
    if not sweep.empty:
        pm = precision_of(sweep)
        gains = []
        b2 = series(sweep, "probe_skel_only")
        for model in models_in(sweep):
            c = contrast(series(sweep, "probe", model=model), b2)
            if c:
                gains.append(c[0])
        if gains:
            items.append(("best vs worst proposer", max(gains) - min(gains), 0, 0, BLUE, 1.0))
    items.reverse()
    ys = np.arange(len(items))
    ax.barh(ys, [i[1] for i in items], 0.6, xerr=[[i[2] for i in items], [i[3] for i in items]],
            color=[i[4] for i in items], edgecolor="white", linewidth=0.4,
            error_kw=dict(ecolor=GREY, elinewidth=0.7))
    ax.set_yticks(ys); ax.set_yticklabels([i[0] for i in items], fontsize=6.2)
    ax.set_xlabel("directed-edge F1 it is worth")
    ax.axvline(0, color=DARK, lw=0.8)
    ax.xaxis.grid(True); ax.set_axisbelow(True)
    span = max(i[1] + i[3] for i in items)
    for y, i in zip(ys, items):
        if i[5] > 0.05:
            ax.text(i[1] + i[3] + 0.012 * span, y, "n.s.", va="center", fontsize=5.8, color=GREY)
    ax.set_xlim(-0.02 * span, 1.28 * span)
    ax.set_title("(c)  the architecture, not the model", loc="left", fontweight="bold", fontsize=7)
    fig.tight_layout(w_pad=1.3)
    save(fig, out_dir, "nemchua_f3_credit")


# --------------------------------------------------------------------------- #
def fig_mechanism(out_dir):
    df = load_merged("main_v2")
    if df.empty:
        print("  [skip] f4"); return
    print("  [f4] main_v2")
    mods = models_in(df)
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.05))

    ax = axes[0]
    pairs = [("probe_random_edits", "probe_random_edits_noreserve", None, "random\nedits", RED),
             ("probe", "probe_noreserve", mods[0] if mods else None, "LLM\nedits", BLUE),
             ("probe_oracle_edits", "probe_oracle_edits_noreserve", None, "perfect\nedits", GREEN)]
    rows = []
    for on, off, model, label, color in pairs:
        a, b = df[df["arm"] == on], df[df["arm"] == off]
        if model is not None:
            a, b = a[a["model_tag"] == model], b[b["model_tag"] == model]
        if a.empty or b.empty:
            continue
        rows.append((label, color, mean_ci(a["directed_f1"]), mean_ci(b["directed_f1"]),
                     a["truth_in_hypotheses"].mean(), b["truth_in_hypotheses"].mean()))
    xs = np.arange(len(rows))
    ax.bar(xs - 0.19, [r[3][0] for r in rows], 0.36, yerr=[r[3][1] for r in rows],
           color="none", edgecolor=[r[1] for r in rows], linewidth=1.0, hatch="////",
           label="guard off", error_kw=dict(ecolor=GREY, elinewidth=0.7))
    ax.bar(xs + 0.19, [r[2][0] for r in rows], 0.36, yerr=[r[2][1] for r in rows],
           color=[r[1] for r in rows], edgecolor="white", linewidth=0.4,
           label="guard on", error_kw=dict(ecolor=GREY, elinewidth=0.7))
    for x, r in zip(xs, rows):
        ax.text(x, 1.012, f"truth kept\n{r[5]:.0%} to {r[4]:.0%}", ha="center", va="bottom",
                fontsize=5.6, color=GREY, linespacing=1.15)
    base = df[df["arm"] == "probe_skel_only"]["directed_f1"].mean()
    ax.axhline(base, color=DARK, lw=0.9, ls=(0, (3, 2)))
    ax.text(-0.44, base - 0.022, "no edits at all", fontsize=6, color=DARK, ha="left")
    ax.set_xticks(xs); ax.set_xticklabels([r[0] for r in rows], fontsize=6.2)
    ax.set_ylabel("directed-edge F1"); ax.set_ylim(0.83, 1.13)
    ax.set_yticks([0.85, 0.90, 0.95, 1.00])
    ax.legend(loc="lower left", ncol=2, handlelength=1.4, columnspacing=1.0, fontsize=6)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title("(a)  the guard bounds a bad proposal", loc="left", fontweight="bold")

    ax = axes[1]
    st = steps_of("main_v2")
    if not st.empty:
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
            ax.plot([0] + list(wide.columns), [start] + [wide[c].mean() for c in wide.columns],
                    color=color, label=label, ls=ls, marker="o",
                    markerfacecolor="white", markeredgewidth=1.1)
        ax.set_xlabel("interventions performed"); ax.set_ylabel("posterior entropy (nats)")
        ax.set_xticks(range(0, 5)); ax.legend(loc="upper right")
        ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title("(b)  experiments collapse the posterior", loc="left", fontweight="bold")
    fig.tight_layout(w_pad=1.6)
    save(fig, out_dir, "nemchua_f4_mechanism")


# --------------------------------------------------------------------------- #
def fig_when(out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.15))

    # (a) anonymized synthetic: LLM minus random editor, against front-end quality
    ax = axes[0]
    pts = []
    for n in (40, 60, 120, 300, 1000):
        df = load_merged(f"ladder_n{n}")
        if df.empty:
            continue
        pcq = df[df["arm"] == "probe_skel_only"]["pc_skeleton_f1_ceiling"].mean()
        rnd = series(df, "probe_random_edits")
        for model in models_in(df):
            c = contrast(series(df, "probe", model=model), rnd)
            if c:
                pts.append((n, pcq, model, c))
    if pts:
        for mi, model in enumerate(sorted({p[2] for p in pts})):
            sel = [p for p in pts if p[2] == model]
            xs = [p[0] for p in sel]; ys = [p[3][0] for p in sel]
            es = [[p[3][0] - p[3][1] for p in sel], [p[3][2] - p[3][0] for p in sel]]
            ax.errorbar(xs, ys, yerr=es, color=[BLUE, ORANGE][mi % 2], marker="o",
                        markerfacecolor="white", markeredgewidth=1.1, label=short(model))
        ax.axhline(0, color=DARK, lw=0.9)
        ax.text(1250, 0.003, "level of a random editor", fontsize=6, color=GREY,
                va="bottom", ha="right")
        ax.set_xscale("log"); ax.set_xlim(33, 1300)
        ax.set_xticks([40, 60, 120, 300, 1000])
        ax.set_xticklabels(["40", "60", "120", "300", "1000"])
        ax.set_xlabel("observational sample size")
        ax.set_ylabel("F1 gain over a\nrandom editor")
        ax.legend(loc="upper right"); ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title("(a)  anonymized variables: an edge only at $n{=}40$", loc="left",
                 fontweight="bold", fontsize=7)

    # (b) named variables on published structures
    ax = axes[1]
    df = load_merged("semantic_n60")
    if not df.empty:
        print("  [f5b] semantic_n60")
        p = df[df["arm"] == "probe"].copy()
        p["proposed"] = p["repair_remove"] + p["repair_add"]
        p["correct"] = p["edits_correct_remove"] + p["edits_correct_add"]
        rnd = df[df["arm"] == "probe_random_edits"]
        rnd_idx = dict(zip(zip(rnd["graph"], rnd["seed"]), rnd["directed_f1"]))
        rows = []
        for model, g in p.groupby("model_tag"):
            entry = {"model": short(model)}
            for cond in ("anon", "named"):
                gg = g[g["condition"] == cond]
                if gg.empty:
                    continue
                entry[f"prec_{cond}"] = gg["correct"].sum() / max(gg["proposed"].sum(), 1)
                keys = list(zip(gg["graph"], gg["seed"]))
                d = np.array([f - rnd_idx[k] for f, k in zip(gg["directed_f1"], keys)
                              if k in rnd_idx])
                entry[f"gain_{cond}"] = d.mean()
                entry[f"err_{cond}"] = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
            rows.append(entry)
        rows = [r for r in rows if "gain_named" in r and "gain_anon" in r]
        rows.sort(key=lambda r: r["prec_named"])
        xs = np.arange(len(rows))
        ax.bar(xs - 0.19, [r["gain_anon"] for r in rows], 0.36, yerr=[r["err_anon"] for r in rows],
               color=LIGHT, edgecolor="white", linewidth=0.4, label="X0 … Xd",
               error_kw=dict(ecolor=GREY, elinewidth=0.7))
        ax.bar(xs + 0.19, [r["gain_named"] for r in rows], 0.36, yerr=[r["err_named"] for r in rows],
               color=PURPLE, edgecolor="white", linewidth=0.4, label="real variable names",
               error_kw=dict(ecolor=GREY, elinewidth=0.7))
        for x, r in zip(xs, rows):
            ax.text(x, -0.006, f"{r['prec_named']:.0%}", ha="center", va="top", fontsize=5.8,
                    color=PURPLE, transform=ax.get_xaxis_transform())
        ax.text(-0.9, -0.006, "edits\ncorrect", ha="center", va="top", fontsize=5.6,
                color=PURPLE, transform=ax.get_xaxis_transform(), linespacing=1.1)
        ax.axhline(0, color=DARK, lw=0.9)
        ax.set_xticks(xs)
        ax.set_xticklabels([r["model"] for r in rows], fontsize=6, rotation=16, ha="right")
        ax.tick_params(axis="x", pad=13)
        ax.set_ylabel("F1 gain over a\nrandom editor")
        ax.set_ylim(-0.10, 0.13)
        ax.legend(loc="upper left", fontsize=6)
        ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title("(b)  real names, scarce data: it finally does", loc="left",
                 fontweight="bold", fontsize=7)
    fig.tight_layout(w_pad=1.6)
    save(fig, out_dir, "nemchua_f5_when")


def main():
    global STUDY_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-dir", default="study2_new")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()
    STUDY_DIR = args.study_dir
    os.makedirs(args.out_dir, exist_ok=True)
    fig_main(args.out_dir)
    fig_credit(args.out_dir)
    fig_mechanism(args.out_dir)
    fig_when(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
