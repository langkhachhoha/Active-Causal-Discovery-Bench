#!/usr/bin/env python
"""Every number the NemChua paper states, recomputed from the run tree.

    python paper/nemchua_numbers.py                # print
    python paper/nemchua_numbers.py --json out.json

Run this after any new stage lands and diff it against the paper. Nothing here is
cached: if a claim in the text disagrees with this output, the text is wrong.
"""
from __future__ import annotations

import argparse, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY = os.path.join(ROOT, "study2")


def pick(*names):
    for n in names:
        if os.path.exists(os.path.join(STUDY, n, "episodes.csv")):
            return n
    return None


def load(run):
    d = pd.read_csv(os.path.join(STUDY, run, "episodes.csv"))
    return d[d["status"] == "success"].copy()


def ser(d, arm, model=None, metric="directed_f1"):
    s = d[d["arm"] == arm]
    if model and (s["model_tag"] != "none").any():
        s = s[s["model_tag"] == model]
    return {(int(r.level), int(r.seed)): float(getattr(r, metric)) for r in s.itertuples()}


def contrast(a, b):
    k = sorted(set(a) & set(b))
    if not k:
        return None
    x = np.array([a[i] for i in k]); y = np.array([b[i] for i in k]); d = x - y
    rng = np.random.default_rng(0)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)])
    return dict(n=len(d), a=round(x.mean(), 4), b=round(y.mean(), 4), diff=round(d.mean(), 4),
                lo=round(float(np.percentile(bs, 2.5)), 4), hi=round(float(np.percentile(bs, 97.5)), 4),
                p=float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0,
                win=int((d > 1e-9).sum()), loss=int((d < -1e-9).sum()), tie=int((np.abs(d) <= 1e-9).sum()))


def mci(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if len(v) < 2:
        return dict(mean=round(float(v.mean()), 4) if len(v) else None, ci=0.0, n=len(v))
    return dict(mean=round(float(v.mean()), 4), ci=round(float(1.96 * v.std(ddof=1) / np.sqrt(len(v))), 4), n=len(v))


def audit(run):
    path = os.path.join(STUDY, run, "analysis", "edit_audit.csv")
    if not os.path.exists(path):
        return {}
    a = pd.read_csv(path); out = {}
    for model, g in a.groupby("model_tag"):
        proposed = int((g["n_remove"] + g["n_add"]).sum())
        correct = int((g["correct_remove"] + g["correct_add"]).sum())
        share_add = g["n_add"].sum() / max(proposed, 1)
        chance = share_add * g["chance_add"].mean() + (1 - share_add) * g["chance_remove"].mean()
        out[model] = dict(
            n=len(g), proposed=proposed, correct=correct,
            precision=round(correct / max(proposed, 1), 4), chance=round(float(chance), 4),
            lift=round(correct / max(proposed, 1) / max(float(chance), 1e-9), 2),
            n_remove=int(g["n_remove"].sum()), correct_remove=int(g["correct_remove"].sum()),
            n_add=int(g["n_add"].sum()), correct_add=int(g["correct_add"].sum()),
            pc_fp=int(g["pc_fp"].sum()), pc_fn=int(g["pc_fn"].sum()),
            fn_recall=round(g["correct_add"].sum() / max(g["pc_fn"].sum(), 1), 4),
            pc_skeleton_f1=round(float(g["pc_skeleton_f1"].mean()), 4),
        )
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json", default=""); a = ap.parse_args()
    out = {}
    MAIN = pick("main_v2", "main")
    df = load(MAIN)
    mods = sorted(m for m in df["model_tag"].unique() if m != "none")
    out["main_run"] = MAIN
    out["models"] = mods
    out["n_episodes"] = int(len(df))

    # arm means
    arms = {}
    for arm in sorted(df["arm"].unique()):
        sub = df[df["arm"] == arm]
        if (sub["model_tag"] != "none").any():
            for m in mods:
                s = sub[sub["model_tag"] == m]
                arms[f"{arm}|{m}"] = {k: mci(s[k]) for k in
                                      ("directed_f1", "compelled_f1", "skeleton_f1", "dag_shd",
                                       "interventions_used", "total_tokens", "cost_usd", "llm_calls",
                                       "truth_in_hypotheses", "best_f1_in_hypotheses",
                                       "entropy_final_nats", "map_weight_final")}
        else:
            arms[f"{arm}|none"] = {k: mci(sub[k]) for k in
                                   ("directed_f1", "compelled_f1", "skeleton_f1", "dag_shd",
                                    "interventions_used", "truth_in_hypotheses", "best_f1_in_hypotheses")}
    out["main_arms"] = arms

    # headline contrasts
    cons = {}
    for m in mods:
        for other in ("pc_greedy_meek", "pc_greedy", "probe_skel_only", "probe_mec_only",
                      "probe_llm_graphs", "llm_e2e", "probe_no_update", "probe_random_sel",
                      "probe_maxdeg_sel", "probe_no_bic", "probe_repair_only", "probe_marginal"):
            if other not in set(df["arm"]):
                continue
            c = contrast(ser(df, "probe", m), ser(df, other, m))
            if c:
                cons[f"probe[{m}] - {other}"] = c
    for arm in ("probe_random_edits", "probe_oracle_edits"):
        c = contrast(ser(df, arm), ser(df, "probe_skel_only"))
        if c:
            cons[f"{arm} - probe_skel_only"] = c
    out["main_contrasts"] = cons
    out["main_edit_audit"] = audit(MAIN)

    # sample-size ladder
    ladder = {}
    for n in (40, 60, 120, 300, 1000):
        run = pick(f"ladder_n{n}", f"n_obs_{n}")
        if not run:
            continue
        d = load(run)
        entry = {"run": run, "arms": {}, "contrasts": {}, "audit": audit(run)}
        for arm in sorted(d["arm"].unique()):
            sub = d[d["arm"] == arm]
            if (sub["model_tag"] != "none").any():
                for m in sorted(x for x in sub["model_tag"].unique() if x != "none"):
                    entry["arms"][f"{arm}|{m}"] = mci(sub[sub["model_tag"] == m]["directed_f1"])
            else:
                entry["arms"][f"{arm}|none"] = mci(sub["directed_f1"])
        base = ser(d, "probe_skel_only")
        for m in sorted(x for x in d["model_tag"].unique() if x != "none"):
            c = contrast(ser(d, "probe", m), base)
            if c:
                entry["contrasts"][f"probe[{m}] - skel_only"] = c
        for arm in ("probe_random_edits", "probe_oracle_edits"):
            c = contrast(ser(d, arm), base)
            if c:
                entry["contrasts"][f"{arm} - skel_only"] = c
        ladder[n] = entry
    out["sample_size_ladder"] = ladder

    # entropy and accuracy trajectories
    st_path = os.path.join(STUDY, MAIN, "steps.csv")
    if os.path.exists(st_path):
        st = pd.read_csv(st_path)
        traj = {}
        for arm in ("probe", "probe_random_sel", "probe_maxdeg_sel", "probe_no_bic",
                    "probe_no_update", "pc_greedy_meek"):
            sub = st[st["arm"] == arm]
            if sub.empty:
                continue
            rec = {}
            if "entropy_after_nats" in sub and sub["entropy_after_nats"].notna().any():
                w = sub.pivot_table(index=["level", "seed", "model_tag"], columns="step",
                                    values="entropy_after_nats")
                w = w.reindex(columns=sorted(w.columns)).ffill(axis=1)
                rec["entropy_start"] = round(float(sub[sub["step"] == 1]["entropy_before_nats"].mean()), 4)
                rec["entropy_after"] = {int(c): round(float(w[c].mean()), 4) for c in w.columns}
            if "map_directed_f1_after" in sub and sub["map_directed_f1_after"].notna().any():
                w = sub.pivot_table(index=["level", "seed", "model_tag"], columns="step",
                                    values="map_directed_f1_after")
                w = w.reindex(columns=sorted(w.columns)).ffill(axis=1)
                rec["f1_after"] = {int(c): round(float(w[c].mean()), 4) for c in w.columns}
            traj[arm] = rec
        out["trajectories"] = traj

    # semantic study
    sem = pick("semantic_n60", "semantic")
    if sem:
        d = pd.read_csv(os.path.join(STUDY, sem, "episodes.csv"))
        d = d[d["status"] == "success"]
        rec = {"run": sem, "by_model": {}}
        for (model, cond), g in d[d["arm"] == "probe"].groupby(["model_tag", "condition"]):
            proposed = int((g["repair_remove"] + g["repair_add"]).sum())
            correct = int((g["edits_correct_remove"] + g["edits_correct_add"]).sum())
            rec["by_model"][f"{model}|{cond}"] = dict(
                n=len(g), proposed=proposed, correct=correct,
                precision=round(correct / max(proposed, 1), 4),
                directed_f1=mci(g["directed_f1"]),
            )
        for arm in ("probe_skel_only", "probe_oracle_edits", "probe_random_edits", "pc_greedy_meek"):
            g = d[d["arm"] == arm]
            if len(g):
                rec[arm] = mci(g["directed_f1"])
        out["semantic"] = rec

    # capability sweep
    for tag in ("models_n60", "models_n300"):
        if not pick(tag):
            continue
        d = load(tag)
        rec = {"audit": audit(tag), "gain_over_no_llm": {}, "arms": {}}
        base = ser(d, "probe_skel_only")
        for m in sorted(x for x in d["model_tag"].unique() if x != "none"):
            c = contrast(ser(d, "probe", m), base)
            if c:
                rec["gain_over_no_llm"][m] = c
        for arm in sorted(d["arm"].unique()):
            sub = d[d["arm"] == arm]
            if (sub["model_tag"] != "none").any():
                for m in sorted(x for x in sub["model_tag"].unique() if x != "none"):
                    rec["arms"][f"{arm}|{m}"] = mci(sub[sub["model_tag"] == m]["directed_f1"])
            else:
                rec["arms"][f"{arm}|none"] = mci(sub["directed_f1"])
        out[tag] = rec

    text = json.dumps(out, indent=1, default=float)
    if a.json:
        open(a.json, "w").write(text)
        print(f"[written] {a.json}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
