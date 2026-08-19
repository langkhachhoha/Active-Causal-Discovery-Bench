"""RauMa figure: how separable the orientation evidence actually is.

    python scripts/make_rauma_figure_verifier.py --study-dir study1 --out-dir figures

Panel (a) plots every local orientation decision the mean-shift rule made, as the
two-sample |Z| for the neighbour under test, split by the ground-truth direction. The two
classes do not overlap: the rule is not adjudicating a close call. Panel (b) sweeps the
decision threshold and shows that the whole residual error of the rule is type-I error at
the conventional 1.96 cut, and that abstaining in the low-|Z| band makes things worse.

Inputs are produced by (both are model-free and cost nothing to rerun):
    run_study1_localreadout.py --mode mechanical --selector {oracle,random,maxdeg,eig}
    run_study1_decompose.py --meanshift-z ... [--meanshift-abstain ...]
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


def load_sweep(study_dir: str) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(study_dir, "ablation_verifier", "*", "episodes.csv"))):
        name = os.path.basename(os.path.dirname(path))
        df = pd.read_csv(path)
        df = df[df.status == "success"]
        correct, wrong = df.orientations_correct.sum(), df.orientations_wrong.sum()
        rows.append({
            "z": float(re.search(r"z([\d.]+)$", name).group(1)),
            "abstain": name.startswith("abstain"),
            "f1": df.directed_f1.mean(),
            "err": wrong / max(correct + wrong, 1),
        })
    if not rows:
        raise SystemExit("no threshold sweep; run run_study1_decompose.py --meanshift-z ...")
    return pd.DataFrame(rows).sort_values("z")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-dir", default="study1")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()

    dec = load_decisions(args.study_dir)
    sweep = load_sweep(args.study_dir)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(5.5, 2.0),
                                   gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.42})

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
    ax0.set_xlabel(r"mean-shift statistic $|Z|$ at the neighbour (log scale)")
    ax0.set_title("(a) the evidence is not a close call", loc="left")
    ax0.xaxis.grid(True)
    ax0.set_axisbelow(True)
    for side in ("left",):
        ax0.spines[side].set_visible(False)

    # ---------------------------------------------------------------- panel b
    forced = sweep[~sweep.abstain]
    absta = sweep[sweep.abstain]
    ax0b = ax1.twinx()
    ax1.plot(forced.z, forced.f1, color=ORANGE, marker="o", markersize=3.4,
             label="forced orientation")
    ax1.plot(absta.z, absta.f1, color=BLUE, marker="s", markersize=3.4, ls=(0, (3, 1.6)),
             label="abstain when $1.0 \\leq |Z| <$ cut")
    ax0b.plot(forced.z, 100 * forced.err, color=GREY, marker="^", markersize=3.0, lw=0.9,
              ls=(0, (1, 1.4)))
    ax0b.set_ylabel("orientation errors (%)", color=GREY)
    ax0b.tick_params(axis="y", colors=GREY)
    ax0b.spines["right"].set_visible(True)
    ax0b.spines["right"].set_color(AXIS)
    ax0b.set_ylim(0, 14)
    ax1.axvline(1.96, color=RED, lw=0.9, ls=(0, (3, 2)))
    ax1.set_xlabel(r"decision threshold on $|Z|$")
    ax1.set_ylabel("directed-edge F1")
    ax1.set_ylim(0.78, 0.89)
    ax1.set_title("(b) 1.96 is not the right cut", loc="left")
    ax1.yaxis.grid(True)
    ax1.set_axisbelow(True)
    ax1.legend(loc="lower right", handlelength=1.3, labelspacing=0.25, fontsize=6.2)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out_dir, f"rauma_f6_verifier.{ext}"), dpi=300)
    plt.close(fig)
    print(f"[written] {args.out_dir}/rauma_f6_verifier.pdf")

    print(f"decisions={len(dec)}  accuracy={dec.correct.mean():.4f}  errors={int((1 - dec.correct).sum())}")
    print(f"separating gap: [{lo:.2f}, {hi:.2f}]")
    for z in (1.282, 1.645, 1.96, 2.576, 3.291):
        pred = np.where(dec.z > z, "a_to_b", "b_to_a")
        print(f"  z={z:<6} local accuracy={np.mean(pred == dec.truth):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
