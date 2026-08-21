#!/usr/bin/env python
"""NemChua figures, revision round.

    python scripts/make_nemchua_v2.py --study-dir study2_new --ab-dir study2b --out-dir figures

Three things changed after the first draft, and each needs its own picture:

  * the random-editor control now matches each model's realized edit counts instead of the
    cap, which is what `analysis/permutation.csv` holds;
  * a data-only ranker executes the rule we hand the proposer, which turns "does the model
    know anything" into a measurable contrast rather than a rhetorical one;
  * the proposer prompt asserted a false adjacency criterion, and the corrected interface
    is a paired A/B in `study2b/`.

Main:      f2 ladder, f3 credit, f4 mechanism, f5 when
Appendix:  a1 edits, a2 chain, a3 semantic, a4 ladder-grid, a5 cost,
           a6 permnull, a7 promptab, a8 ranker, a9 ceiling
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

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

STUDY_DIR, AB_DIR = "study2_new", "study2b"
SHORT = {
    "qwen3-coder-30b-a3b-instruct": "qwen3-30b", "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
    "claude-haiku-4.5": "haiku-4.5", "gemini-3-flash-preview": "gemini-3-flash",
    "gpt-5.4-mini": "gpt-5.4-mini",
}
# ordered weakest to strongest by edit precision under the executed prompt
MODEL_ORDER = ["qwen3-30b", "gpt-4o-mini", "gpt-5.4-mini", "haiku-4.5", "gemini-3-flash"]


def short(tag: str) -> str:
    return SHORT.get(tag, tag[:14])


def exists(path: str) -> bool:
    return os.path.exists(path)


def load_merged(run: str, root: str | None = None) -> pd.DataFrame:
    """`<run>_fix` for every arm it re-ran, plus the arms only `<run>` has."""
    root = root or STUDY_DIR
    frames = []
    fix = os.path.join(root, f"{run}_fix", "episodes.csv")
    if exists(fix):
        f = pd.read_csv(fix)
        frames.append(f[f["status"] == "success"])
    base = os.path.join(root, run, "episodes.csv")
    if exists(base):
        b = pd.read_csv(base)
        b = b[b["status"] == "success"]
        if frames:
            b = b[~b["arm"].isin(set(frames[0]["arm"]))]
        frames.append(b)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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


def mean_ci(values):
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if len(v) < 2:
        return (float(v.mean()) if len(v) else np.nan), 0.0
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))


def arm_mean(df, arm, metric="directed_f1"):
    """Mean over instances; LLM arms are averaged within instance across models first."""
    sub = df[df["arm"] == arm]
    if sub.empty:
        return np.nan, 0.0
    g = sub.groupby(["level", "seed"])[metric].mean()
    return mean_ci(g.values)


def paired(df, arm_a, arm_b, metric="directed_f1"):
    a = df[df["arm"] == arm_a].groupby(["level", "seed"])[metric].mean()
    b = df[df["arm"] == arm_b].groupby(["level", "seed"])[metric].mean()
    keys = a.index.intersection(b.index)
    if len(keys) < 6:
        return None
    d = (a.loc[keys] - b.loc[keys]).values
    nz = d[np.abs(d) > 1e-12]
    p = float(wilcoxon(nz).pvalue) if len(nz) >= 6 else 1.0
    return float(d.mean()), p, len(d)


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def perm_table(path):
    if not exists(path):
        return pd.DataFrame()
    d = pd.read_csv(path)
    d["model"] = d["model_tag"].map(short)
    return d


def audit_table(path):
    if not exists(path):
        return pd.DataFrame()
    d = pd.read_csv(path)
    d["model"] = d["model_tag"].map(short)
    return d


def audit_summary(d, arm=None):
    """Per-model edit precision and the share of wrong additions that are co-parent pairs."""
    if d.empty:
        return pd.DataFrame()
    if arm is not None and "arm" in d.columns:
        d = d[d["arm"] == arm]
    if d.empty:
        return pd.DataFrame()
    g = d.groupby("model").agg(
        n_remove=("n_remove", "sum"), correct_remove=("correct_remove", "sum"),
        n_add=("n_add", "sum"), correct_add=("correct_add", "sum"),
        spouse=("add_spouse", "sum"), other=("add_other", "sum"),
        chance_spouse=("chance_spouse", "mean"),
        per_inst=("n_add", "size"),
    ).reset_index()
    g["proposed"] = g.n_remove + g.n_add
    g["correct"] = g.correct_remove + g.correct_add
    g["precision"] = g.correct / g.proposed.clip(lower=1)
    wrong = (g.spouse + g.other).clip(lower=1)
    g["spouse_share"] = g.spouse / wrong
    g["spouse_lift"] = g.spouse_share / g.chance_spouse.clip(lower=1e-9)
    g["edits_per_instance"] = g.proposed / g.per_inst.clip(lower=1)
    return g


def order_models(g):
    g = g.copy()
    g["k"] = g["model"].apply(lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99)
    return g.sort_values("k")


def style(ax, ylab=None, xlab=None, title=None, grid="y"):
    if grid:
        ax.grid(axis=grid, alpha=0.9, zorder=0)
        ax.set_axisbelow(True)
    if ylab:
        ax.set_ylabel(ylab)
    if xlab:
        ax.set_xlabel(xlab)
    if title:
        ax.set_title(title, loc="left", pad=4)


# --------------------------------------------------------------------------- #
# F2 — the ladder: where the accuracy actually comes from
# --------------------------------------------------------------------------- #
LADDER = [
    ("pc_greedy_meek", "PC + greedy + Meek", GREY),
    ("probe_skel_only", "no edits", GREY),
    ("__perm__", "random edits, count-matched", LIGHT),
    ("probe", "LLM edits", BLUE),
    ("probe_stat_edits", "statistical ranker", ORANGE),
    ("probe_oracle_edits", "perfect edits", GREEN),
]


def fig_ladder(out_dir):
    cohorts = [("main_v2", "$n_{\\mathrm{obs}}=300$, $d\\in\\{4,6,8,10\\}$", "main n=300"),
               ("models_n60", "$n_{\\mathrm{obs}}=60$, $d\\in\\{6,8,10\\}$", "models n=60")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55))
    for ax, (run, title, _) in zip(axes, cohorts):
        df = load_merged(run)
        perm = perm_table(os.path.join(STUDY_DIR, run, "analysis", "permutation.csv"))
        labels, means, errs, colors = [], [], [], []
        for arm, lab, col in LADDER:
            if arm == "__perm__":
                if perm.empty:
                    continue
                m, e = mean_ci(perm["random_mean_f1"].values)
            else:
                m, e = arm_mean(df, arm)
            if not np.isfinite(m):
                continue
            labels.append(lab); means.append(m); errs.append(e); colors.append(col)
        y = np.arange(len(labels))[::-1]
        ax.barh(y, means, xerr=errs, color=colors, height=0.62, zorder=3,
                error_kw=dict(ecolor=DARK, lw=0.7, capsize=2))
        for yi, m, e in zip(y, means, errs):
            ax.text(m + e + 0.007, yi, f"{m:.3f}", va="center", ha="left",
                    fontsize=6.2, color=DARK)
        ax.set_yticks(y); ax.set_yticklabels(labels)
        lo = min(m - e for m, e in zip(means, errs)) - 0.02
        hi = max(m + e for m, e in zip(means, errs))
        ax.set_xlim(max(0.0, lo), hi + 0.075)
        style(ax, xlab="directed-edge F1", title=title, grid="x")
    fig.suptitle("The proposal channel is real, and a two-line ranker exploits it better than any model",
                 x=0.012, ha="left", fontsize=8.2, y=1.045)
    fig.tight_layout()
    save(fig, out_dir, "nemchua_f2_main")


# --------------------------------------------------------------------------- #
# F3 — credit: the matched-count permutation, and the flat capability curve
# --------------------------------------------------------------------------- #
def fig_credit(out_dir):
    fig = plt.figure(figsize=(7.0, 2.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.42)

    # (a) LLM vs its own count-matched null, per model, two cohorts.
    # Plotted as the percentile the proposal reaches inside its own null rather than the
    # raw F1 difference: that difference is dominated by ties and a few large wins, so a
    # normal interval on it badly understates what the rank test sees.
    ax = fig.add_subplot(gs[0, 0])
    rows = []
    for run, lab, col in (("models_n60", "$n_{\\mathrm{obs}}{=}60$", ORANGE),
                          ("models_n300", "$n_{\\mathrm{obs}}{=}300$", BLUE)):
        pt = perm_table(os.path.join(STUDY_DIR, run, "analysis", "permutation.csv"))
        if pt.empty:
            continue
        recs = []
        for mname, sub in pt.groupby("model"):
            v = sub["percentile"].values
            d = (sub["llm_f1"] - sub["random_mean_f1"]).values
            nz = d[np.abs(d) > 1e-12]
            pw = float(wilcoxon(nz).pvalue) if len(nz) >= 6 else 1.0
            recs.append({"model": mname, "pct": v.mean(),
                         "e": 1.96 * v.std(ddof=1) / np.sqrt(len(v)), "p": pw})
        rows.append((order_models(pd.DataFrame(recs)), lab, col))
    width, top = 0.36, 0.5
    for i, (g, lab, col) in enumerate(rows):
        x = np.arange(len(g)) + (i - 0.5) * width
        ax.bar(x, g["pct"] - 0.5, width, bottom=0.5, yerr=g["e"], color=col, label=lab,
               zorder=3, error_kw=dict(ecolor=DARK, lw=0.6, capsize=1.6))
        for xi, (_, r) in zip(x, g.iterrows()):
            top = max(top, r["pct"] + r["e"])
            mark = stars(r["p"]).replace("n.s.", "")
            if mark:
                ax.text(xi, r["pct"] + r["e"] + 0.004, mark, ha="center",
                        fontsize=5.6, color=DARK)
    if rows:
        ax.set_xticks(np.arange(len(rows[0][0])))
        ax.set_xticklabels(rows[0][0]["model"], rotation=30, ha="right")
    ax.axhline(0.5, color=DARK, lw=0.9, ls=":")
    ax.set_ylim(0.47, top + 0.045)
    ax.legend(loc="upper left", ncol=2, columnspacing=0.9)
    style(ax, ylab="percentile in own null",
          title="(a) every model beats its own volume")

    # (b) edit precision does not buy final accuracy
    ax = fig.add_subplot(gs[0, 1])
    df = load_merged("models_n60")
    aud = audit_summary(audit_table(os.path.join(STUDY_DIR, "models_n60", "analysis", "edit_audit.csv")))
    xs, ys, names = [], [], []
    for _, r in aud.iterrows():
        sub = df[(df["arm"] == "probe") & (df["model_tag"].map(short) == r["model"])]
        if sub.empty:
            continue
        xs.append(r["precision"]); ys.append(sub["directed_f1"].mean()); names.append(r["model"])
    ax.scatter(xs, ys, s=26, color=BLUE, zorder=3)
    for x, y, n in zip(xs, ys, names):
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=5.8, color=GREY)
    stat_m, _ = arm_mean(df, "probe_stat_edits")
    ax.axhline(stat_m, color=ORANGE, lw=1.1, ls="--", zorder=2)
    ax.text(0.99, stat_m, "statistical ranker  ", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=6, color=ORANGE)
    if len(xs) > 2:
        rho, p = spearmanr(xs, ys)
        ax.text(0.03, 0.04, f"$\\rho={rho:.2f}$, {stars(p)}", transform=ax.transAxes, fontsize=6.2)
    pad = (max(ys) - min(ys)) or 0.01
    ax.set_ylim(min(min(ys), stat_m) - 0.4 * pad, max(max(ys), stat_m) + 1.1 * pad)
    style(ax, xlab="edit precision", ylab="directed-edge F1",
          title="(b) $4.2\\times$ precision, no gain")

    # (c) one-at-a-time sensitivities
    ax = fig.add_subplot(gs[0, 2])
    dm = load_merged("main_v2")
    items = [("likelihood update", paired(dm, "probe", "probe_no_update")),
             ("BIC weights", paired(dm, "probe", "probe_no_bic")),
             ("candidate guard", paired(dm, "probe", "probe_noreserve")),
             ("EIG vs random", paired(dm, "probe", "probe_random_sel")),
             ("LLM vs ranker", paired(dm, "probe", "probe_stat_edits"))]
    items = [(k, v) for k, v in items if v]
    y = np.arange(len(items))[::-1]
    vals = [v[0] for _, v in items]
    cols = [RED if v < 0 else (GREEN if p < 0.05 else GREY) for (_, (v, p, _)) in items]
    ax.barh(y, vals, color=cols, height=0.6, zorder=3)
    for yi, (k, (v, p, _)) in zip(y, items):
        mark = stars(p).replace("n.s.", "")
        txt = f"{v:+.3f}{(' ' + mark) if mark else ''}"
        # a negative bar has clear space to the right of zero; putting its label on the
        # left instead would run it into the category tick label
        xpos = v + 0.006 if v >= 0 else 0.006
        ax.text(xpos, yi, txt, va="center", ha="left", fontsize=6, color=DARK)
    ax.set_yticks(y); ax.set_yticklabels([k for k, _ in items])
    ax.tick_params(axis="y", pad=1)
    ax.axvline(0, color=DARK, lw=0.8)
    lo, hi = min(vals), max(vals)
    span = hi - lo
    ax.set_xlim(lo - 0.09 * span, hi + 0.34 * span)
    style(ax, xlab="F1 change when removed", title="(c) one-at-a-time effects", grid="x")
    fig.tight_layout()
    save(fig, out_dir, "nemchua_f3_credit")


# --------------------------------------------------------------------------- #
# F4 — mechanism: the co-parent artifact, and what correcting the prompt does
# --------------------------------------------------------------------------- #
def fig_mechanism(out_dir):
    mis = audit_summary(audit_table(os.path.join(STUDY_DIR, "models_n60", "analysis", "edit_audit.csv")))
    ab = audit_table(os.path.join(AB_DIR, "sepset_n60", "analysis", "edit_audit.csv"))
    ab_mis, ab_cor = audit_summary(ab, arm="probe"), audit_summary(ab, arm="probe_sepset")

    fig = plt.figure(figsize=(7.0, 2.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.42)

    # (a) the better the model, the more its errors concentrate on co-parents
    ax = fig.add_subplot(gs[0, 0])
    g = order_models(mis)
    ax.scatter(g["precision"], g["spouse_lift"], s=28, color=PURPLE, zorder=3)
    # the two weakest models nearly coincide just above the chance line, so their labels
    # are pushed apart by hand rather than both centred above the marker
    nudge = {"qwen3-30b": (9, -2, "left"), "gpt-4o-mini": (0, 7, "center")}
    for _, r in g.iterrows():
        dx, dy, ha = nudge.get(r["model"], (0, 6, "center"))
        ax.annotate(r["model"], (r["precision"], r["spouse_lift"]), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=5.8, color=GREY)
    ax.axhline(1.0, color=DARK, lw=0.8, ls=":")
    ax.text(0.995, 1.0, "chance ", transform=ax.get_yaxis_transform(), va="bottom",
            ha="right", fontsize=5.8, color=GREY)
    if len(g) > 2:
        rho, p = spearmanr(g["precision"], g["spouse_lift"])
        ax.text(0.03, 0.93, f"$\\rho={rho:.2f}$, {stars(p)}", transform=ax.transAxes, fontsize=6.2)
    ax.set_ylim(0, g["spouse_lift"].max() * 1.35)
    xlo, xhi = g["precision"].min(), g["precision"].max()
    ax.set_xlim(xlo - 0.28 * (xhi - xlo), xhi + 0.16 * (xhi - xlo))
    style(ax, xlab="edit precision", ylab="co-parent enrichment ($\\times$ chance)",
          title="(a) capability buys compliance")

    # (b) stating the rule correctly removes the artifact
    ax = fig.add_subplot(gs[0, 1])
    m = order_models(ab_mis).set_index("model")["spouse_lift"]
    c = ab_cor.set_index("model")["spouse_lift"].reindex(m.index)
    x = np.arange(len(m))
    ax.bar(x - 0.19, m.values, 0.38, color=RED, label="stated wrongly", zorder=3)
    ax.bar(x + 0.19, c.values, 0.38, color=GREEN, label="stated correctly", zorder=3)
    ax.axhline(1.0, color=DARK, lw=0.8, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(m.index, rotation=30, ha="right")
    ax.legend(loc="upper left")
    ax.set_ylim(0, max(m.max(), np.nanmax(c.values)) * 1.42)
    style(ax, ylab="co-parent enrichment ($\\times$ chance)",
          title="(b) the artifact is ours, not theirs")

    # (c) ... and it buys no accuracy
    ax = fig.add_subplot(gs[0, 2])
    dfab = pd.read_csv(os.path.join(AB_DIR, "sepset_n60", "episodes.csv"))
    dfab = dfab[dfab["status"] == "success"]
    labs, vals, errs = [], [], []
    for mod in [x for x in MODEL_ORDER if x in set(dfab["model_tag"].map(short))]:
        s = dfab[dfab["model_tag"].map(short) == mod]
        a = s[s["arm"] == "probe_sepset"].set_index(["level", "seed"])["directed_f1"]
        b = s[s["arm"] == "probe"].set_index(["level", "seed"])["directed_f1"]
        k = a.index.intersection(b.index)
        d = (a.loc[k] - b.loc[k]).values
        labs.append(mod); vals.append(d.mean())
        errs.append(1.96 * d.std(ddof=1) / np.sqrt(len(d)))
    x = np.arange(len(labs))
    cols = [GREEN if v > 0 else RED for v in vals]
    ax.bar(x, vals, 0.6, yerr=errs, color=cols, zorder=3,
           error_kw=dict(ecolor=DARK, lw=0.6, capsize=1.8))
    ax.axhline(0, color=DARK, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labs, rotation=30, ha="right")
    style(ax, ylab="F1, corrected $-$ misspecified",
          title="(c) correcting it does not help")
    fig.tight_layout()
    save(fig, out_dir, "nemchua_f4_mechanism")


# --------------------------------------------------------------------------- #
# F5 — when a proposer is worth anything
# --------------------------------------------------------------------------- #
def fig_when(out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45))

    # (a) the crossover in sample size, LLM and ranker against the same no-edit baseline
    ax = axes[0]
    ns = [40, 60, 120, 300, 1000]
    got, cur = [], {"probe": [], "probe_stat_edits": []}
    err = {"probe": [], "probe_stat_edits": []}
    for n in ns:
        df = load_merged(f"ladder_n{n}")
        if df.empty or "probe_stat_edits" not in set(df["arm"]):
            continue
        got.append(n)
        for arm in cur:
            r = paired(df, arm, "probe_skel_only")
            if r is None:
                cur[arm].append(np.nan); err[arm].append(0.0); continue
            a = df[df["arm"] == arm].groupby(["level", "seed"])["directed_f1"].mean()
            b = df[df["arm"] == "probe_skel_only"].groupby(["level", "seed"])["directed_f1"].mean()
            k = a.index.intersection(b.index)
            d = (a.loc[k] - b.loc[k]).values
            cur[arm].append(d.mean())
            err[arm].append(1.96 * d.std(ddof=1) / np.sqrt(len(d)))
    for arm, lab, col in (("probe", "LLM edits", BLUE),
                          ("probe_stat_edits", "statistical ranker", ORANGE)):
        ax.errorbar(got, cur[arm], yerr=err[arm], fmt="o-", color=col, label=lab, zorder=3,
                    elinewidth=0.7, capsize=1.8)
    ax.axhline(0, color=DARK, lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_xticks(got)
    ax.set_xticklabels([str(n) for n in got])
    ax.legend(loc="upper right")
    style(ax, xlab="observational rows $n_{\\mathrm{obs}}$", ylab="F1 gain over no edits",
          title="(a) only where the estimator is underpowered")

    # (b) named vs anonymized, and the ranker that sees neither
    ax = axes[1]
    rows = []
    for run, lab in (("semantic_n60", "$n_{\\mathrm{obs}}{=}60$"), ("semantic", "$n_{\\mathrm{obs}}{=}300$")):
        p = os.path.join(STUDY_DIR, run, "episodes.csv")
        if not exists(p):
            continue
        d = pd.read_csv(p); d = d[d["status"] == "success"]
        pr = d[d["arm"] == "probe"]
        a = pr[pr["condition"] == "named"].set_index(["graph", "seed", "model_tag"])["directed_f1"]
        b = pr[pr["condition"] == "anon"].set_index(["graph", "seed", "model_tag"])["directed_f1"]
        k = a.index.intersection(b.index)
        dd = (a.loc[k] - b.loc[k]).values
        st = d[d["arm"] == "probe_stat_edits"].groupby(["graph", "seed"])["directed_f1"].mean()
        sk = d[d["arm"] == "probe_skel_only"].groupby(["graph", "seed"])["directed_f1"].mean()
        kk = st.index.intersection(sk.index)
        sd = (st.loc[kk] - sk.loc[kk]).values
        p_ = wilcoxon(dd[np.abs(dd) > 1e-12]).pvalue if np.any(dd) else 1.0
        ps = wilcoxon(sd[np.abs(sd) > 1e-12]).pvalue if np.any(sd) else 1.0
        rows.append((lab, dd.mean(), 1.96 * dd.std(ddof=1) / np.sqrt(len(dd)), p_,
                     sd.mean(), 1.96 * sd.std(ddof=1) / np.sqrt(len(sd)), ps))
    x = np.arange(len(rows))
    ax.bar(x - 0.19, [r[1] for r in rows], 0.38, yerr=[r[2] for r in rows], color=PURPLE,
           label="names, LLM", zorder=3, error_kw=dict(ecolor=DARK, lw=0.6, capsize=1.8))
    ax.bar(x + 0.19, [r[4] for r in rows], 0.38, yerr=[r[5] for r in rows], color=ORANGE,
           label="ranker, no names", zorder=3, error_kw=dict(ecolor=DARK, lw=0.6, capsize=1.8))
    for i, r in enumerate(rows):
        ax.text(i - 0.19, r[1] + r[2] + 0.004, stars(r[3]), ha="center", fontsize=6, color=DARK)
        ax.text(i + 0.19, r[4] + r[5] + 0.004, stars(r[6]), ha="center", fontsize=6, color=DARK)
    ax.axhline(0, color=DARK, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows])
    ax.legend(loc="upper right")
    style(ax, ylab="F1 gain", title="(b) names help, statistics help more")
    fig.tight_layout()
    save(fig, out_dir, "nemchua_f5_when")


# --------------------------------------------------------------------------- #
# A6 — the null distribution the LLM is being placed inside
# --------------------------------------------------------------------------- #
def fig_permnull(out_dir):
    runs = [("models_n60", "$n_{\\mathrm{obs}}{=}60$"), ("models_n300", "$n_{\\mathrm{obs}}{=}300$"),
            ("main_v2", "main, $n_{\\mathrm{obs}}{=}300$")]
    runs = [(r, l) for r, l in runs if exists(os.path.join(STUDY_DIR, r, "analysis", "permutation.csv"))]
    fig, axes = plt.subplots(1, len(runs) + 1, figsize=(7.0, 2.2))
    for ax, (run, lab) in zip(axes, runs):
        d = perm_table(os.path.join(STUDY_DIR, run, "analysis", "permutation.csv"))
        ax.hist(d["percentile"], bins=20, range=(0, 1), color=BLUE, alpha=0.85, zorder=3)
        ax.axvline(0.5, color=DARK, lw=0.9, ls=":")
        mu = d["percentile"].mean()
        ax.axvline(mu, color=RED, lw=1.2)
        # the mean can sit almost on the null line and on the tallest bar, so the label is
        # pinned to the panel corner rather than to the line it describes
        ax.text(0.97, 0.95, f"mean {mu:.3f}", transform=ax.transAxes, fontsize=6.2,
                color=RED, va="top", ha="right",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
        style(ax, xlab="percentile in own null", title=lab)
        if ax is axes[0]:
            ax.set_ylabel("instances $\\times$ models")
    # the corrected interface, for contrast
    ax = axes[-1]
    p = os.path.join(AB_DIR, "sepset_n60", "analysis", "permutation.csv")
    if exists(p):
        d = perm_table(p)
        ax.hist(d["percentile"], bins=20, range=(0, 1), color=GREEN, alpha=0.85, zorder=3)
        ax.axvline(0.5, color=DARK, lw=0.9, ls=":")
        mu = d["percentile"].mean()
        ax.axvline(mu, color=RED, lw=1.2)
        ax.text(0.97, 0.95, f"mean {mu:.3f}", transform=ax.transAxes, fontsize=6.2,
                color=RED, va="top", ha="right",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    style(ax, xlab="percentile in own null", title="corrected prompt, $n_{\\mathrm{obs}}{=}60$")
    fig.suptitle("Where each proposal falls among 200 random proposals of exactly its own size",
                 x=0.012, ha="left", fontsize=8, y=1.06)
    fig.tight_layout()
    save(fig, out_dir, "nemchua_a6_permnull")


# --------------------------------------------------------------------------- #
# A7 — the prompt A/B in full
# --------------------------------------------------------------------------- #
def fig_promptab(out_dir):
    ab = audit_table(os.path.join(AB_DIR, "sepset_n60", "analysis", "edit_audit.csv"))
    mis, cor = audit_summary(ab, arm="probe"), audit_summary(ab, arm="probe_sepset")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35))
    order = [m for m in MODEL_ORDER if m in set(mis["model"])]
    x = np.arange(len(order))
    for ax, col, ylab, title in (
            (axes[0], "precision", "edit precision", "(a) precision converges"),
            (axes[1], "edits_per_instance", "edits per instance", "(b) volume"),
            (axes[2], "correct", "correct edits (total)", "(c) correct edits found")):
        a = mis.set_index("model")[col].reindex(order)
        b = cor.set_index("model")[col].reindex(order)
        ax.bar(x - 0.19, a.values, 0.38, color=RED, label="stated wrongly", zorder=3)
        ax.bar(x + 0.19, b.values, 0.38, color=GREEN, label="stated correctly", zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(order, rotation=32, ha="right")
        style(ax, ylab=ylab, title=title)
        if ax is axes[0]:
            ax.legend(loc="upper left")
    fig.suptitle("Correcting the adjacency rule removes the artifact and the signal together",
                 x=0.012, ha="left", fontsize=8, y=1.05)
    fig.tight_layout()
    save(fig, out_dir, "nemchua_a7_promptab")


# --------------------------------------------------------------------------- #
# A8 — the ranker across every cohort
# --------------------------------------------------------------------------- #
def fig_ranker(out_dir):
    cohorts = [("main_v2", "main\n$n{=}300$"), ("models_n60", "models\n$n{=}60$"),
               ("models_n300", "models\n$n{=}300$"), ("robust_d12", "$d{=}12$")]
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    arms = [("probe_skel_only", "no edits", GREY), ("probe_random_edits", "random (cap)", LIGHT),
            ("probe", "LLM", BLUE), ("probe_stat_edits", "ranker", ORANGE),
            ("probe_oracle_edits", "perfect", GREEN)]
    width = 0.16
    got = []
    for ci, (run, lab) in enumerate(cohorts):
        df = load_merged(run)
        if df.empty:
            continue
        got.append(lab)
        for ai, (arm, alab, col) in enumerate(arms):
            m, e = arm_mean(df, arm)
            if not np.isfinite(m):
                continue
            ax.bar(len(got) - 1 + (ai - 2) * width, m, width, yerr=e, color=col, zorder=3,
                   label=alab if ci == 0 else None,
                   error_kw=dict(ecolor=DARK, lw=0.5, capsize=1.4))
    ax.set_xticks(np.arange(len(got))); ax.set_xticklabels(got)
    ax.set_ylim(0.75, 1.02)
    # the bars fill the panel, so the key goes above it rather than on top of the data
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=5, columnspacing=1.4)
    style(ax, ylab="directed-edge F1")
    ax.set_title("The ranker leads everywhere the exact posterior can be enumerated, and only there",
                 loc="left", pad=18)
    fig.tight_layout()
    save(fig, out_dir, "nemchua_a8_ranker")


# --------------------------------------------------------------------------- #
# A9 — adjacency ceiling vs orientation ceiling
# --------------------------------------------------------------------------- #
def fig_ceiling(out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45))
    ax = axes[0]
    cohorts = [("main_v2", "main $n{=}300$"), ("models_n60", "models $n{=}60$"),
               ("models_n300", "models $n{=}300$"), ("robust_d12", "$d{=}12$")]
    labs, oe, ts = [], [], []
    for run, lab in cohorts:
        df = load_merged(run)
        a, _ = arm_mean(df, "probe_oracle_edits")
        b, _ = arm_mean(df, "probe_true_skeleton")
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        labs.append(lab); oe.append(a); ts.append(b)
    x = np.arange(len(labs))
    ax.bar(x - 0.19, oe, 0.38, color=GREEN, label="perfect edits ($\\leq 4$)", zorder=3)
    ax.bar(x + 0.19, ts, 0.38, color=PURPLE, label="true skeleton (any distance)", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(labs, rotation=18, ha="right")
    ax.set_ylim(0.8, 1.02); ax.legend(loc="lower left")
    style(ax, ylab="directed-edge F1", title="(a) synthetic: the edit budget never binds")

    ax = axes[1]
    labs, oe, ts, sk = [], [], [], []
    for run, lab in (("semantic_n60", "$n{=}60$"), ("semantic", "$n{=}300$")):
        p = os.path.join(STUDY_DIR, run, "episodes.csv")
        if not exists(p):
            continue
        d = pd.read_csv(p); d = d[d["status"] == "success"]
        f = lambda arm: d[d["arm"] == arm].groupby(["graph", "seed"])["directed_f1"].mean().mean()
        labs.append(lab); oe.append(f("probe_oracle_edits")); ts.append(f("probe_true_skeleton"))
        sk.append(f("probe_skel_only"))
    x = np.arange(len(labs))
    ax.bar(x - 0.26, sk, 0.26, color=GREY, label="no edits", zorder=3)
    ax.bar(x, oe, 0.26, color=GREEN, label="perfect edits ($\\leq 4$)", zorder=3)
    ax.bar(x + 0.26, ts, 0.26, color=PURPLE, label="true skeleton", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(labs)
    ax.set_ylim(0.5, 1.05); ax.legend(loc="lower left", ncol=1)
    style(ax, ylab="directed-edge F1", title="(b) published networks: it binds hard")
    fig.tight_layout()
    save(fig, out_dir, "nemchua_a9_ceiling")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-dir", default="study2_new")
    ap.add_argument("--ab-dir", default="study2b")
    ap.add_argument("--out-dir", default="figures")
    a = ap.parse_args()
    global STUDY_DIR, AB_DIR
    STUDY_DIR, AB_DIR = a.study_dir, a.ab_dir
    os.makedirs(a.out_dir, exist_ok=True)
    for fn in (fig_ladder, fig_credit, fig_mechanism, fig_when,
               fig_permnull, fig_promptab, fig_ranker, fig_ceiling):
        try:
            fn(a.out_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {fn.__name__}: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
