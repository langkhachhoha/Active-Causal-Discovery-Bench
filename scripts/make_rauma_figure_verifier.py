"""RauMa figure: how separable the orientation evidence actually is.

    python scripts/make_rauma_figure_verifier.py --study-dir study1 --out-dir figures

Panel (a) plots every local orientation decision the mean-shift rule made, as the
two-sample |Z| for the neighbour under test, split by the ground-truth direction. The two
classes do not overlap: the rule is not adjudicating a close call. Panel (b) hands the same
one-edge decision to an LLM under three prompt conditions and shows where the readout
failure actually lives -- not in the causal rule, but in the numeric reduction that turns
two means into a verdict.

Inputs are produced by:
    run_study1_localreadout.py --mode mechanical --selector {oracle,random,maxdeg,eig}
    run_study1_localreadout.py --mode llm --prompt {stats,rule,rule_z} --models ...
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#c8352b"
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
    "lines.linewidth": 1.4, "lines.markersize": 4,
    "errorbar.capsize": 2, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def load_decisions(study_dir: str) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(os.path.join(study_dir, "localreadout", "mechanical*", "decisions.csv"))):
        frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit("no mechanical decision logs; run run_study1_localreadout.py --mode mechanical")
    return pd.concat(frames, ignore_index=True)


def load_ladder(result_dir: str) -> pd.DataFrame:
    """Per-decision accuracy of each readout condition, on the matched first-round decisions.

    A local decision changes the belief graph, so later rounds diverge between conditions.
    Round 1 is the one point where every condition faces an identical state, which makes it
    the honest comparison; the on-policy totals are reported in the paper's appendix.
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(result_dir, "localreadout", "*", "decisions.csv"))):
        name = os.path.basename(os.path.dirname(path))
        d = pd.read_csv(path)
        d1 = d[d.step == 1]
        if name.startswith("mechanical"):
            model, prompt = "mechanical", "mechanical"
        else:
            model, _, prompt = name.partition("_")
        rows.append({"config": name, "model": model, "prompt": prompt,
                     "n": len(d1), "acc": d1.correct.mean(),
                     "acc_all": d.correct.mean()})
    if not rows:
        raise SystemExit("no decision logs; run run_study1_localreadout.py")
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-dir", default="study1")
    ap.add_argument("--result-dir", default="result")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()

    dec = load_decisions(args.study_dir)
    lad = load_ladder(args.result_dir)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(5.5, 2.1),
                                   gridspec_kw={"width_ratios": [1.1, 1.0], "wspace": 0.34})

    # ---------------------------------------------------------------- panel a
    rng = np.random.default_rng(0)
    groups = [("truly out of the target", "a_to_b", ORANGE, 1.0, "right"),
              ("truly into the target", "b_to_a", BLUE, 0.02, "left")]
    for i, (label, truth, colour, xpos, ha) in enumerate(groups):
        z = dec.loc[dec.truth == truth, "z"].to_numpy()
        y = i + rng.uniform(-0.22, 0.22, size=z.size)
        ax0.scatter(np.maximum(z, 0.02), y, s=4.5, color=colour, alpha=0.55,
                    linewidths=0, zorder=3)
        ax0.text(85 if ha == "right" else 0.025, i + 0.30, f"{label}  (n={z.size})",
                 fontsize=6.3, color=colour, va="bottom", ha=ha)

    lo = dec.loc[dec.truth == "b_to_a", "z"].max()
    hi = dec.loc[dec.truth == "a_to_b", "z"].min()
    ax0.axvspan(lo, hi, color=LIGHT, alpha=0.32, zorder=1)
    ax0.axvline(1.96, color=RED, lw=0.9, ls=(0, (3, 2)), zorder=4)
    ax0.text(1.75, -0.30, "rule's cut 1.96", color=RED, fontsize=6.2, ha="right", va="center")
    ax0.text(np.sqrt(lo * hi), 1.60, f"nothing lands in\n[{lo:.1f}, {hi:.1f}]",
             fontsize=6.2, color=GREY, ha="center", va="center")
    ax0.set_xscale("log")
    ax0.set_xlim(0.02, 90)
    ax0.set_ylim(-0.55, 1.95)
    ax0.set_yticks([])
    ax0.set_xlabel(r"mean-shift statistic $|Z|$ (log scale)")
    ax0.set_title("(a) the evidence is not a close call", loc="left")
    ax0.xaxis.grid(True)
    ax0.set_axisbelow(True)
    ax0.spines["left"].set_visible(False)

    # ---------------------------------------------------------------- panel b
    conds = [("stats", "numbers\nonly"), ("rule", "+ causal\nrule"), ("rule_z", "+ computed\n$Z$")]
    models = [("qwen3-coder-30b", "Qwen3-Coder-30B", BLUE), ("gpt-4o-mini", "GPT-4o-mini", "#7fb2ea")]
    width = 0.36
    for mi, (mkey, mlabel, colour) in enumerate(models):
        ys = [float(lad[(lad.model == mkey) & (lad.prompt == c)].acc.iloc[0]) for c, _ in conds]
        xs = np.arange(len(conds)) + (mi - 0.5) * width
        ax1.bar(xs, ys, width * 0.92, color=colour, edgecolor="white", linewidth=0.4, label=mlabel)
        for x, y in zip(xs, ys):
            ax1.text(x, y + 0.018, f"{y:.0%}", ha="center", fontsize=6.0, color=DARK)
    # every LLM ladder run used --selector oracle, so the like-for-like mechanical
    # reference is that same arm, not the average over selectors.
    ref = float(lad[lad.config == "mechanical"].acc.iloc[0])
    ax1.axhline(ref, color=ORANGE, lw=1.0, ls=(0, (4, 2)))
    ax1.text(1.0, ref + 0.015, f"mean-shift rule ({ref:.0%})", transform=ax1.get_yaxis_transform(),
             ha="right", va="bottom", fontsize=6.2, color=ORANGE)
    ax1.axhline(0.5, color=GREY, lw=0.8, ls=(0, (1, 1.6)))
    ax1.text(0.995, 0.505, "chance", transform=ax1.get_yaxis_transform(), ha="right",
             va="bottom", fontsize=6.0, color=GREY)
    ax1.set_xlim(-0.55, 2.95)
    ax1.set_xticks(range(len(conds)))
    ax1.set_xticklabels([lbl for _, lbl in conds], fontsize=6.4)
    ax1.set_ylim(0, 1.14)
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_ylabel("one-edge decisions correct")
    ax1.set_title("(b) what the LLM is actually missing", loc="left")
    ax1.yaxis.grid(True)
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper left", handlelength=1.0, labelspacing=0.22, fontsize=6.0,
               borderpad=0.2, bbox_to_anchor=(-0.02, 1.02))

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out_dir, f"rauma_f6_verifier.{ext}"), dpi=300)
    plt.close(fig)
    print(f"[written] {args.out_dir}/rauma_f6_verifier.pdf")

    print(f"decisions={len(dec)}  accuracy={dec.correct.mean():.4f}  errors={int((1 - dec.correct).sum())}")
    print(f"separating gap: [{lo:.2f}, {hi:.2f}]")
    print(lad.sort_values(["model", "prompt"]).round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
