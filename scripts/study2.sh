#!/usr/bin/env bash
# Study 2 — PROBE: LLM-proposed hypothesis spaces + exact Bayesian experimental design.
#
#   bash scripts/study2.sh smoke     # ~3 min,  ~$0.02  — check everything works
#   bash scripts/study2.sh main      # ~2-4 h,  ~$2-5   — the paper's main table + all ablations
#   bash scripts/study2.sh ablation  # ~2-3 h,  ~$2-4   — sample-size sweep (the headline ablation)
#   bash scripts/study2.sh all       # main + ablation
#
# Safe to re-run: every stage checkpoints and resumes.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME="${ACDB_ENV:-acdb-active}"
PY="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)' 2>/dev/null || echo python)"
MODELS="${ACDB_MODELS:-qwen3-coder-30b,gpt-4o-mini}"
WORKERS="${ACDB_WORKERS:-6}"
STAGE="${1:-smoke}"

run() {
    local name="$1"; shift
    local out="traces/study2/${name}"
    echo
    echo "=============================================================="
    echo " study2 :: ${name}"
    echo " out    :: ${out}"
    echo "=============================================================="
    "$PY" run_study2_probe.py --out-dir "$out" --models "$MODELS" --workers "$WORKERS" --resume "$@"
    "$PY" scripts/analyze.py --study 2 --run-dir "$out"
}

case "$STAGE" in
smoke)
    run smoke --levels 0,1 --seeds-per-level 2 --n-obs 300 --n-int 150
    ;;
main)
    # every arm: baselines + PROBE + hypothesis-space ablations + decision-layer ablations
    run main --levels 0,1,2,3 --seeds-per-level 10 --n-obs 300 --n-int 150
    ;;
ablation)
    # The headline ablation: the LLM proposer earns its keep exactly where PC's skeleton
    # is unreliable, i.e. when observational data is scarce.
    for N in 40 60 120 300 1000; do
        run "n_obs_${N}" --levels 1,2 --seeds-per-level 10 --n-obs "$N" --n-int $(( N > 200 ? 150 : N )) \
            --arms oracle,pc_greedy,pc_greedy_meek,probe,probe_repair_only,probe_skel_only,probe_mec_only,probe_llm_graphs
    done
    # How many skeleton edits should the LLM be allowed to make?
    for E in 2 4 8; do
        run "edits_${E}" --levels 1,2 --seeds-per-level 8 --n-obs 60 --n-int 40 \
            --max-skeleton-edits "$E" --arms probe,probe_repair_only,probe_skel_only
    done
    # Largest graph size.
    run d12 --levels 4 --seeds-per-level 8 --n-obs 300 --n-int 150
    ;;
all)
    bash "$0" main
    bash "$0" ablation
    ;;
*)
    echo "unknown stage '$STAGE' (expected: smoke | main | ablation | all)" >&2
    exit 1
    ;;
esac

echo
echo "[study2] done. Tables and figures are in traces/study2/*/analysis/"
