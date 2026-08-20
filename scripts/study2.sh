#!/usr/bin/env bash
# Study 2 — NemChua: LLM-proposed hypothesis spaces + exact Bayesian experimental design.
#
#   bash scripts/study2.sh smoke     # ~4 min,  ~$0.05  — check everything works
#   bash scripts/study2.sh main      # ~2-3 h,  ~$3-6   — headline table + every ablation
#   bash scripts/study2.sh models    # ~2-3 h,  ~$5-12  — capability sweep across 5 proposers
#   bash scripts/study2.sh ladder    # ~1-2 h,  ~$2-4   — proposal-content ladder vs sample size
#   bash scripts/study2.sh semantic  # ~40 min, ~$2-4   — named vs anonymized variables
#   bash scripts/study2.sh robust    # ~1 h,    ~$1-2   — budget, alpha and d=12 robustness
#   bash scripts/study2.sh all       # everything, in dependency order
#
# Safe to re-run: every stage checkpoints and resumes.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME="${ACDB_ENV:-acdb-active}"
PY="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)' 2>/dev/null || echo python)"
CORE_MODELS="${ACDB_MODELS:-qwen3-coder-30b,gpt-4o-mini}"
# The capability sweep. Any OpenRouter id works; these five span roughly two orders of
# magnitude of price and capability, which is the axis the paper needs.
SWEEP_MODELS="${ACDB_SWEEP_MODELS:-qwen3-coder-30b,gpt-4o-mini,gpt-5.4-mini,haiku-4.5,gemini-3-flash}"
WORKERS="${ACDB_WORKERS:-6}"
SEEDS="${ACDB_SEEDS:-20}"
OUT_ROOT="${ACDB_OUT_ROOT:-study2}"
STAGE="${1:-smoke}"

# The proposal-content ladder: no edits -> random edits -> LLM edits -> perfect edits.
LADDER_ARMS="oracle,pc_greedy_meek,probe_skel_only,probe_random_edits,probe,probe_oracle_edits"

run() {
    local name="$1"; shift
    local out="${OUT_ROOT}/${name}"
    echo
    echo "=============================================================="
    echo " study2 :: ${name}"
    echo " out    :: ${out}"
    echo "=============================================================="
    "$PY" run_study2_probe.py --out-dir "$out" --workers "$WORKERS" --resume "$@"
    "$PY" scripts/analyze.py --study 2 --run-dir "$out" || true
}

run_semantic() {
    local name="$1"; shift
    local out="${OUT_ROOT}/${name}"
    echo
    echo "=============================================================="
    echo " study2 :: ${name} (semantic)"
    echo " out    :: ${out}"
    echo "=============================================================="
    "$PY" run_study2_semantic.py --out-dir "$out" --workers "$WORKERS" --resume "$@"
}

case "$STAGE" in
smoke)
    run smoke --models "$CORE_MODELS" --levels 0,1 --seeds-per-level 2 --n-obs 300 --n-int 150
    run_semantic semantic_smoke --models "gpt-4o-mini" --graphs asia --seeds 2
    ;;

main)
    # The headline table: every arm on every instance, so all 20 arms are paired.
    run main_v2 --models "$CORE_MODELS" --levels 0,1,2,3 --seeds-per-level "$SEEDS" \
        --n-obs 300 --n-int 150
    ;;

models)
    # Does the story depend on using weak proposers? Same instances, five proposers,
    # in both the data-poor regime (where the proposal channel is live) and the
    # data-rich one (where PC is already right).
    run models_n60 --models "$SWEEP_MODELS" --levels 1,2,3 --seeds-per-level "$SEEDS" \
        --n-obs 60 --n-int 60 \
        --arms "${LADDER_ARMS},probe_llm_graphs,probe_noreserve,llm_e2e"
    run models_n300 --models "$SWEEP_MODELS" --levels 1,2,3 --seeds-per-level "$SEEDS" \
        --n-obs 300 --n-int 150 \
        --arms "${LADDER_ARMS},probe_llm_graphs,probe_noreserve"
    ;;

ladder)
    # The crossover: the proposal channel pays off only where the statistical front-end
    # is underpowered. `probe_random_edits` and `probe_oracle_edits` bracket the LLM.
    for N in 40 60 120 300 1000; do
        run "ladder_n${N}" --models "$CORE_MODELS" --levels 1,2 --seeds-per-level "$SEEDS" \
            --n-obs "$N" --n-int $(( N > 200 ? 150 : N )) \
            --arms "${LADDER_ARMS},probe_llm_graphs,probe_mec_only"
    done
    # Is the guard that makes a wrong proposal free actually load-bearing? Same instances,
    # same proposals, only the reserved share of the hypothesis budget changes.
    for N in 60 300; do
        run "reserve_n${N}" --models "$CORE_MODELS" --levels 1,2 --seeds-per-level "$SEEDS" \
            --n-obs "$N" --n-int $(( N > 200 ? 150 : N )) \
            --arms "probe,probe_noreserve,probe_skel_only,probe_random_edits,probe_random_edits_noreserve,probe_oracle_edits,probe_oracle_edits_noreserve"
    done
    # How many edits should a proposer be allowed to make?
    for E in 2 4 8; do
        run "edits_e${E}" --models "$CORE_MODELS" --levels 1,2 --seeds-per-level "$SEEDS" \
            --n-obs 60 --n-int 60 --max-skeleton-edits "$E" \
            --arms probe,probe_skel_only,probe_random_edits
    done
    ;;

semantic)
    # Anonymized variables strip exactly what an LLM is supposed to know. Put the names
    # back, on published structures, and measure the proposal channel both ways.
    run_semantic semantic --models "$SWEEP_MODELS" \
        --graphs cancer,earthquake,survey,asia,sachs --seeds 12 --n-obs 300 --n-int 150
    # Data-poor version: this is where the proposal channel matters most.
    run_semantic semantic_n60 --models "$SWEEP_MODELS" \
        --graphs cancer,earthquake,survey,asia,sachs --seeds 12 --n-obs 60 --n-int 60
    ;;

robust)
    # (a) tight budget: every experiment must count, so the selector becomes load-bearing.
    run robust_tightbudget --models "$CORE_MODELS" --levels 0,1,2,3 --seeds-per-level "$SEEDS" \
        --n-obs 300 --n-int 150 --budget-slack 0 \
        --arms oracle,pc_greedy_meek,probe,probe_random_sel,probe_maxdeg_sel,probe_skel_only
    # (b) is the crossover an artefact of PC's significance threshold?
    for A in 0.01 0.10; do
        run "robust_alpha${A}" --models "$CORE_MODELS" --levels 1,2 --seeds-per-level "$SEEDS" \
            --n-obs 60 --n-int 60 --alpha "$A" --arms "$LADDER_ARMS"
    done
    # (c) the largest graph the exact posterior can still be enumerated on.
    run robust_d12 --models "$CORE_MODELS" --levels 4 --seeds-per-level 10 \
        --n-obs 300 --n-int 150 --arms "${LADDER_ARMS},probe_llm_graphs,llm_e2e"
    ;;

all)
    bash "$0" main
    bash "$0" ladder
    bash "$0" models
    bash "$0" semantic
    bash "$0" robust
    ;;
*)
    echo "unknown stage '$STAGE' (expected: smoke | main | models | ladder | semantic | robust | all)" >&2
    exit 1
    ;;
esac

echo
echo "[study2] done. Tables and figures are in ${OUT_ROOT}/*/analysis/"
