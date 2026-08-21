#!/usr/bin/env python
"""Check every number hard-coded in the paper against the runs it came from.

    python paper/nemchua_verify.py

Tables are written by hand, so a transcription slip is invisible until a reader
recomputes it. This recomputes each cited value and prints a line per check. Anything
marked MISMATCH is a number in the .tex that the data does not support.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from importlib.machinery import SourceFileLoader  # noqa: E402

M = SourceFileLoader("m", str(REPO / "scripts" / "make_nemchua_v2.py")).load_module()
M.STUDY_DIR, M.AB_DIR = str(REPO / "study2_new"), str(REPO / "study2b")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FAILS = 0


def check(label, got, want, tol=0.0015):
    global FAILS
    ok = got is not None and np.isfinite(got) and abs(got - want) <= tol
    if not ok:
        FAILS += 1
    got_s = f"{got:.4f}" if got is not None and np.isfinite(got) else "n/a"
    print(f"  {'ok  ' if ok else 'MISMATCH'}  {label:48s} paper={want:.3f}  data={got_s}")


print("Table 1 — main cohort, n_obs=300")
dm = M.load_merged("main_v2")
for arm, want in (("oracle", 1.000), ("pc_greedy", 0.868), ("pc_greedy_meek", 0.874),
                  ("probe_skel_only", 0.901), ("probe_random_edits", 0.951),
                  ("probe", 0.954), ("probe_stat_edits", 0.975),
                  ("probe_oracle_edits", 0.999), ("probe_llm_graphs", 0.658),
                  ("llm_e2e", 0.101)):
    check(f"{arm} directed F1", M.arm_mean(dm, arm)[0], want)
for arm, want in (("probe_skel_only", 0.487), ("probe_random_edits", 0.738),
                  ("probe", 0.681), ("probe_stat_edits", 0.875)):
    check(f"{arm} truth-in-H", M.arm_mean(dm, arm, "truth_in_hypotheses")[0], want)

print("\nSection 5.1 — ablation sensitivities")
for a, b, want in (("probe", "probe_no_update", 0.285), ("probe", "probe_no_bic", 0.021),
                   ("probe", "probe_noreserve", 0.051), ("probe", "probe_random_sel", 0.062),
                   ("probe", "probe_stat_edits", -0.021),
                   ("probe_stat_edits", "probe_skel_only", 0.074),
                   ("probe_oracle_edits", "probe_skel_only", 0.097)):
    r = M.paired(dm, a, b)
    check(f"{a} - {b}", r[0] if r else None, want)
for a, b, want in (("probe_random_edits", "probe_random_edits_noreserve", 0.056),
                   ("probe_oracle_edits", "probe_oracle_edits_noreserve", 0.000)):
    r = M.paired(dm, a, b)
    check(f"guard: {a}", r[0] if r else None, want)

print("\nTable 5 (permutation) — pooled deltas and percentiles")
for run, dcol, want_d, want_p in (("main_v2", None, 0.014, 0.532),
                                  ("models_n60", None, 0.030, 0.591),
                                  ("models_n300", None, 0.017, 0.562)):
    t = M.perm_table(REPO / "study2_new" / run / "analysis" / "permutation.csv")
    check(f"{run} delta", float((t.llm_f1 - t.random_mean_f1).mean()), want_d)
    check(f"{run} percentile", float(t.percentile.mean()), want_p)
t = M.perm_table(REPO / "study2b" / "sepset_n60" / "analysis" / "permutation.csv")
check("corrected-prompt delta", float((t.llm_f1 - t.random_mean_f1).mean()), 0.008)
check("corrected-prompt percentile", float(t.percentile.mean()), 0.515)

print("\nSection 5.2 — ranker vs LLM in each cohort")
for run, want in (("models_n60", -0.028), ("models_n300", -0.028)):
    d = M.load_merged(run)
    r = M.paired(d, "probe", "probe_stat_edits")
    check(f"{run}: probe - ranker", r[0] if r else None, want)

print("\nSection 5.3 — the co-parent audit and the prompt A/B")
aud = M.audit_summary(M.audit_table(REPO / "study2_new" / "models_n60" / "analysis" / "edit_audit.csv"))
g = aud.set_index("model")
for m, want in (("qwen3-30b", 1.2), ("gemini-3-flash", 6.0)):
    check(f"{m} co-parent lift", float(g.loc[m, "spouse_lift"]), want, tol=0.15)
check("gemini edit precision", float(g.loc["gemini-3-flash", "precision"]), 0.328, tol=0.004)
check("gpt-4o-mini edit precision", float(g.loc["gpt-4o-mini", "precision"]), 0.079, tol=0.004)
ab = M.audit_table(REPO / "study2b" / "sepset_n60" / "analysis" / "edit_audit.csv")
mis = M.audit_summary(ab, arm="probe").set_index("model")
cor = M.audit_summary(ab, arm="probe_sepset").set_index("model")
for m, w_mis, w_cor in (("gemini-3-flash", 6.1, 1.4), ("haiku-4.5", 3.7, 1.4)):
    check(f"{m} lift, misspecified", float(mis.loc[m, "spouse_lift"]), w_mis, tol=0.15)
    check(f"{m} lift, corrected", float(cor.loc[m, "spouse_lift"]), w_cor, tol=0.15)
check("gemini correct edits, misspecified", float(mis.loc["gemini-3-flash", "correct"]), 58, tol=0.5)
check("gemini correct edits, corrected", float(cor.loc["gemini-3-flash", "correct"]), 16, tol=0.5)

dab = pd.read_csv(REPO / "study2b" / "sepset_n60" / "episodes.csv")
dab = dab[dab["status"] == "success"]
a = dab[dab["arm"] == "probe_sepset"].set_index(["model_tag", "level", "seed"])["directed_f1"]
b = dab[dab["arm"] == "probe"].set_index(["model_tag", "level", "seed"])["directed_f1"]
k = a.index.intersection(b.index)
check("corrected - misspecified, pooled", float((a.loc[k] - b.loc[k]).mean()), -0.009)
for m, want in (("claude-haiku-4.5", -0.051), ("gemini-3-flash-preview", -0.025)):
    kk = [x for x in k if x[0] == m]
    check(f"corrected - misspecified, {M.short(m)}", float((a.loc[kk] - b.loc[kk]).mean()), want)

print("\nSection 5.4 / Table 9 — semantic")
for run, want_named, want_anon in (("semantic_n60", 0.700, 0.679), ("semantic", 0.821, 0.818)):
    d = pd.read_csv(REPO / "study2_new" / run / "episodes.csv")
    d = d[(d["status"] == "success") & (d["arm"] == "probe")]
    for cond, want in (("named", want_named), ("anon", want_anon)):
        check(f"{run} probe {cond}", float(d[d["condition"] == cond]["directed_f1"].mean()), want)
    sh = pd.read_csv(REPO / "study2_new" / run / "episodes.csv")
    sh = sh[(sh["status"] == "success") & (sh["condition"] == "shared")]
    for arm, lab in (("probe_stat_edits", "ranker"), ("probe_true_skeleton", "true skeleton"),
                     ("probe_skel_only", "no edits")):
        v = sh[sh["arm"] == arm].groupby(["graph", "seed"])["directed_f1"].mean().mean()
        print(f"  info      {run} {lab:16s} = {v:.4f}")

print("\nAppendix — d=12 saturation")
d12 = M.load_merged("robust_d12")
for arm, want in (("probe_true_skeleton", 0.883), ("probe", 0.930), ("probe_stat_edits", 0.858)):
    check(f"d=12 {arm}", M.arm_mean(d12, arm)[0], want, tol=0.004)
check("d=12 true-skeleton best-in-H", M.arm_mean(d12, "probe_true_skeleton", "best_f1_in_hypotheses")[0], 0.918, tol=0.004)
check("d=12 probe best-in-H", M.arm_mean(d12, "probe", "best_f1_in_hypotheses")[0], 0.939, tol=0.004)

print("\nAppendix — Sachs adjacency vs orientation")
sg = pd.read_csv(REPO / "study2_new" / "semantic_n60" / "episodes.csv")
sg = sg[(sg["status"] == "success") & (sg["condition"] == "shared") & (sg["graph"] == "sachs")]
for arm, want in (("probe_skel_only", 0.411), ("probe_oracle_edits", 0.467),
                  ("probe_true_skeleton", 0.828)):
    check(f"sachs {arm}", float(sg[sg["arm"] == arm]["directed_f1"].mean()), want, tol=0.002)

print(f"\n{'ALL CHECKS PASS' if not FAILS else f'{FAILS} MISMATCH(ES) — fix the .tex'}")
sys.exit(1 if FAILS else 0)
