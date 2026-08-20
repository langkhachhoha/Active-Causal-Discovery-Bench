"""Turn a study's raw CSVs into paper-ready tables and figures.

    python scripts/analyze.py --study 1 --run-dir traces/study1/main
    python scripts/analyze.py --study 2 --run-dir traces/study2/main

Writes everything into `<run-dir>/analysis/`:
    tables.md         all tables in markdown, ready to paste into the paper
    *.csv             the same tables as CSV
    *.png             figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "episodes.csv")
    numeric = [
        c for c in df.columns
        if c not in {"run_id", "timestamp_utc", "study", "arm", "selector", "inferencer",
                     "model", "model_tag", "status", "error", "infer_rule"}
    ]
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    failed = int((df["status"] != "success").sum())
    if failed:
        print(f"[warn] {failed}/{len(df)} episodes failed and are excluded from the tables")
        for message, count in df.loc[df["status"] != "success", "error"].value_counts().head(5).items():
            print(f"       {count:4d} x {str(message)[:110]}")
    return df[df["status"] == "success"].copy()


def ci95(values: pd.Series) -> float:
    values = values.dropna()
    if len(values) < 2:
        return 0.0
    return float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))


def summarise(df: pd.DataFrame, by: list[str], metrics: list[str]) -> pd.DataFrame:
    metrics = [m for m in metrics if m in df.columns]
    rows = []
    for key, group in df.groupby(by, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key))
        row["n"] = len(group)
        for metric in metrics:
            row[metric] = group[metric].mean()
            row[f"{metric}_ci"] = ci95(group[metric])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(by).reset_index(drop=True)


def fmt(df: pd.DataFrame, metrics: list[str], decimals: int = 3) -> pd.DataFrame:
    out = df.copy()
    for metric in metrics:
        if metric in out.columns and f"{metric}_ci" in out.columns:
            out[metric] = [
                f"{m:.{decimals}f} ±{c:.{decimals}f}" if pd.notna(m) else "—"
                for m, c in zip(out[metric], out[f"{metric}_ci"])
            ]
            out = out.drop(columns=[f"{metric}_ci"])
    return out


class Report:
    def __init__(self, out_dir: Path, title: str) -> None:
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self.lines = [f"# {title}", ""]

    def table(self, name: str, caption: str, df: pd.DataFrame) -> None:
        df.to_csv(self.out_dir / f"{name}.csv", index=False)
        self.lines += [f"## {caption}", "", df.to_markdown(index=False), ""]
        print(f"\n=== {caption} ===")
        print(df.to_string(index=False))

    def note(self, text: str) -> None:
        self.lines += [text, ""]
        print(f"\n>>> {text}")

    def save(self) -> None:
        (self.out_dir / "tables.md").write_text("\n".join(self.lines), encoding="utf-8")
        print(f"\n[written] {self.out_dir / 'tables.md'}")


def savefig(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print(f"[written] {path}")


# --------------------------------------------------------------------------- #
# study 1
# --------------------------------------------------------------------------- #
QUALITY = ["directed_f1", "compelled_f1", "skeleton_f1", "dag_shd"]
COST = ["prompt_tokens", "completion_tokens", "total_tokens", "cost_usd", "llm_calls", "wall_sec"]


def analyze_study1(df: pd.DataFrame, report: Report) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = df[df["arm"] != "llm_e2e"].copy()

    report.table(
        "t1_main",
        "Table 1 — every arm (mean ± 95% CI over paired instances)",
        fmt(
            summarise(df, ["arm", "model_tag"], QUALITY + ["interventions_used", "efficiency"] + COST),
            QUALITY + ["interventions_used", "efficiency"] + COST,
        ),
    )

    report.table(
        "t2_selector_effect",
        "Table 2 — SELECTION quality: inference held fixed at `meek`",
        fmt(
            summarise(
                grid[grid["inferencer"] == "meek"],
                ["selector", "model_tag"],
                ["directed_f1", "selection_regret_total", "selection_regret_mean",
                 "selection_quality_mean", "eig_regret_total", "wasted_steps", "steps_taken"],
            ),
            ["directed_f1", "selection_regret_total", "selection_regret_mean",
             "selection_quality_mean", "eig_regret_total", "wasted_steps", "steps_taken"],
        ),
    )

    report.table(
        "t3_inferencer_effect",
        "Table 3 — INFERENCE quality: selection held fixed at `oracle`",
        fmt(
            summarise(
                grid[grid["selector"] == "oracle"],
                ["inferencer", "model_tag"],
                QUALITY + ["orientation_accuracy", "submit_directed", "submit_undirected"],
            ),
            QUALITY + ["orientation_accuracy", "submit_directed", "submit_undirected"],
        ),
    )

    report.table(
        "t4_full_grid",
        "Table 4 — the full selector x inferencer grid (directed F1)",
        grid.pivot_table(index="selector", columns=["inferencer", "model_tag"],
                         values="directed_f1", aggfunc="mean").round(3).reset_index(),
    )

    # gap decomposition
    rows = []
    for model, group in grid.groupby("model_tag"):
        if model == "none":
            continue
        pool = grid[grid["model_tag"].isin([model, "none"])]

        def cell(selector: str, inferencer: str) -> float:
            sub = pool[(pool["selector"] == selector) & (pool["inferencer"] == inferencer)]
            return float(sub["directed_f1"].mean()) if len(sub) else np.nan

        best = cell("oracle", "meek")
        agent = cell("llm", "llm")
        rows.append(
            {
                "model": model,
                "best_possible (oracle+meek)": round(best, 4),
                "full_llm_agent (llm+llm)": round(agent, 4),
                "total_gap": round(best - agent, 4),
                "selection_gap (oracle+meek - llm+meek)": round(best - cell("llm", "meek"), 4),
                "inference_gap (oracle+meek - oracle+llm)": round(best - cell("oracle", "llm"), 4),
                "llm_e2e (no scaffold)": round(float(df[(df["arm"] == "llm_e2e") & (df["model_tag"] == model)]["directed_f1"].mean()), 4),
            }
        )
    if rows:
        report.table("t5_gap_decomposition", "Table 5 — attributing the end-to-end gap", pd.DataFrame(rows))
        report.note(
            "Read Table 5 as: `selection_gap` is what you lose by letting the LLM pick experiments "
            "(inference held perfect); `inference_gap` is what you lose by letting the LLM read the "
            "results (selection held perfect). The larger term is the bottleneck."
        )

    report.table(
        "t6_by_level",
        "Table 6 — scaling with graph size",
        fmt(summarise(df, ["arm", "model_tag", "level"], ["directed_f1", "selection_regret_total", "cost_usd"]),
            ["directed_f1", "selection_regret_total", "cost_usd"]),
    )

    # figures
    pivot = grid.pivot_table(index="selector", columns="inferencer", values="directed_f1", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(5, 3.4))
    pivot.plot(kind="bar", ax=ax, rot=0)
    ax.set_ylabel("directed F1")
    ax.set_xlabel("experiment selector")
    ax.set_title("Selection vs inference")
    ax.legend(title="inferencer")
    savefig(fig, report.out_dir / "fig_grid.png")

    steps_path = report.out_dir.parent / "steps.csv"
    if steps_path.exists():
        steps = pd.read_csv(steps_path)
        steps["selection_regret"] = pd.to_numeric(steps["selection_regret"], errors="coerce")
        arm_to_selector = {a: a.split("+")[0] for a in steps["arm"].unique() if "+" in a}
        steps["selector"] = steps["arm"].map(arm_to_selector)
        curve = steps.dropna(subset=["selector"]).groupby(["selector", "step"])["selection_regret"].mean().unstack(0)
        fig, ax = plt.subplots(figsize=(5, 3.4))
        curve.plot(ax=ax, marker="o")
        ax.set_xlabel("intervention index")
        ax.set_ylabel("mean selection regret (edges)")
        ax.set_title("Experiment-choice regret over an episode")
        savefig(fig, report.out_dir / "fig_regret.png")
        report.table(
            "t7_regret_by_step",
            "Table 7 — mean selection regret per intervention index",
            curve.round(3).reset_index(),
        )


# --------------------------------------------------------------------------- #
# study 2
# --------------------------------------------------------------------------- #
SPACE = ["truth_in_hypotheses", "best_f1_in_hypotheses", "n_hypotheses", "truth_rank_final"]

HYPOTHESIS_SOURCE = {
    "probe": "llm_repair + pc_mec",
    "probe_repair_only": "llm_repair",
    "probe_llm_graphs": "llm whole graphs",
    "probe_skel_only": "pc_skeleton (no LLM)",
    "probe_mec_only": "pc_mec (no LLM)",
    "probe_random_hyp": "random (no LLM)",
    "probe_random_edits": "random edits (no LLM)",
    "probe_oracle_edits": "oracle edits (no LLM)",
    "probe_noreserve": "llm_repair, no guard",
    "probe_random_edits_noreserve": "random edits, no guard",
    "probe_oracle_edits_noreserve": "oracle edits, no guard",
}


def analyze_study2(df: pd.DataFrame, report: Report) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report.table(
        "t1_main",
        "Table 1 — main results (mean ± 95% CI over paired instances)",
        fmt(
            summarise(df, ["arm", "model_tag"], QUALITY + ["efficiency", "interventions_used"] + COST),
            QUALITY + ["efficiency", "interventions_used"] + COST,
        ),
    )

    space = df[df["arm"].isin(HYPOTHESIS_SOURCE)].copy()
    space["hypothesis_source"] = space["arm"].map(HYPOTHESIS_SOURCE)
    report.table(
        "t2_hypothesis_space",
        "Table 2 — hypothesis-space quality drives everything",
        fmt(
            summarise(space, ["hypothesis_source", "model_tag"], ["directed_f1"] + SPACE),
            ["directed_f1"] + SPACE,
        ),
    )
    report.note(
        "`truth_in_hypotheses` is the fraction of instances whose true DAG is in the candidate set; "
        "`best_f1_in_hypotheses` is the best directed F1 any candidate could have achieved. Together "
        "they cap what the decision layer can possibly deliver."
    )

    ablations = [a for a in ("probe", "probe_random_sel", "probe_maxdeg_sel", "probe_no_bic",
                            "probe_no_update", "probe_marginal", "probe_noreserve") if a in set(df["arm"])]
    if ablations:
        columns = QUALITY + ["interventions_used", "map_weight_final", "entropy_final_nats"]
        report.table(
            "t3_decision_layer",
            "Table 3 — decision-layer ablations (hypothesis space held fixed)",
            fmt(summarise(df[df["arm"].isin(ablations)], ["arm", "model_tag"], columns), columns),
        )

    report.table(
        "t4_by_level",
        "Table 4 — scaling with graph size",
        fmt(summarise(df, ["arm", "model_tag", "level"], ["directed_f1", "truth_in_hypotheses"]),
            ["directed_f1", "truth_in_hypotheses"]),
    )

    llm = df[df["model_tag"] != "none"].copy()
    if len(llm):
        efficiency = summarise(llm, ["arm", "model_tag"], ["directed_f1", "total_tokens", "cost_usd", "llm_calls"])
        efficiency["f1_per_1k_tokens"] = (efficiency["directed_f1"] / (efficiency["total_tokens"] / 1000)).round(4)
        report.table(
            "t5_token_efficiency",
            "Table 5 — quality per token",
            fmt(efficiency, ["directed_f1", "total_tokens", "cost_usd", "llm_calls"]),
        )

    if {"repair_remove", "repair_add"} <= set(df.columns):
        edits = df.dropna(subset=["repair_remove"])
        if len(edits):
            report.table(
                "t6_edit_behaviour",
                "Table 6 — how aggressively each model edits PC's skeleton",
                fmt(
                    summarise(edits, ["arm", "model_tag"],
                              ["repair_remove", "repair_add", "directed_f1", "best_f1_in_hypotheses"]),
                    ["repair_remove", "repair_add", "directed_f1", "best_f1_in_hypotheses"],
                ),
            )

    # figures
    main = summarise(df, ["arm", "model_tag"], ["directed_f1"])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    pivot = main.pivot_table(index="arm", columns="model_tag", values="directed_f1").sort_values(
        by=main["model_tag"].mode().iat[0] if len(main) else "none", ascending=True
    )
    pivot.plot(kind="barh", ax=ax)
    ax.set_xlabel("directed F1")
    ax.set_ylabel("")
    ax.set_title("PROBE vs baselines and ablations")
    savefig(fig, report.out_dir / "fig_main.png")

    steps_path = report.out_dir.parent / "steps.csv"
    if steps_path.exists():
        steps = pd.read_csv(steps_path)
        for column in ("entropy_after_nats", "map_directed_f1_after", "step"):
            steps[column] = pd.to_numeric(steps[column], errors="coerce")
        subset = steps[steps["arm"].isin(["probe", "probe_random_sel", "probe_no_update"])]
        if len(subset):
            curve = subset.groupby(["arm", "step"])["entropy_after_nats"].mean().unstack(0)
            fig, ax = plt.subplots(figsize=(5, 3.4))
            curve.plot(ax=ax, marker="o")
            ax.set_xlabel("intervention index")
            ax.set_ylabel("posterior entropy (nats)")
            ax.set_title("How fast each rule kills uncertainty")
            savefig(fig, report.out_dir / "fig_entropy.png")
            report.table("t7_entropy_trajectory", "Table 7 — posterior entropy after each experiment",
                         curve.round(3).reset_index())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", choices=("1", "2"), required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    df = load(run_dir)
    if df.empty:
        print("[error] no successful episodes to analyse")
        return 1

    title = "Study 1 — selection vs inference" if args.study == "1" else "Study 2 — PROBE"
    report = Report(run_dir / "analysis", title)
    report.note(f"{len(df)} successful episodes from `{run_dir}`.")
    if args.study == "1":
        analyze_study1(df, report)
    else:
        analyze_study2(df, report)
    report.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
