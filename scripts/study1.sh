#!/usr/bin/env bash
# Study 1 — decomposing active causal discovery into selection and inference.
#
#   bash scripts/study1.sh smoke     # ~2 min,  ~$0.01  — check everything works
#   bash scripts/study1.sh main      # ~2-4 h,  ~$3-6   — the paper's main table
#   bash scripts/study1.sh ablation  # ~1-2 h,  ~$2-4   — sample-size + evidence-format ablations
#   bash scripts/study1.sh verifier  # ~2 min,  free     — mean-shift threshold / abstention sweep
#   bash scripts/study1.sh local     # ~15 min, ~$0.10   — per-edge readout: is it interpretation?
#   bash scripts/study1.sh all       # main + ablation
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
    local out="traces/study1/${name}"
    echo
    echo "=============================================================="
    echo " study1 :: ${name}"
    echo " out    :: ${out}"
    echo "=============================================================="
    "$PY" run_study1_decompose.py --out-dir "$out" --models "$MODELS" --workers "$WORKERS" --resume "$@"
    "$PY" scripts/analyze.py --study 1 --run-dir "$out"
}

case "$STAGE" in
smoke)
    run smoke --levels 0,1 --seeds-per-level 2 --n-obs 300 --n-int 150
    ;;
main)
    # 4 graph sizes x 10 paired instances x (5 selectors x 2 inferencers + end-to-end)
    run main --levels 0,1,2,3 --seeds-per-level 10 --n-obs 300 --n-int 150
    ;;
ablation)
    # (a) TIGHT BUDGET. In `main` the budget is |I*| + 1, so one wasted experiment is still
    # recoverable and every selector lands on the same score. At slack 0 the budget is exactly
    # |I*|: a wasted experiment is unrecoverable and the selection axis becomes load-bearing.
    # Same seeds and same graphs as `main` (slack does not enter the rejection policy), so
    # these rows pair instance-by-instance with the main run.
    run ablation_tightbudget --levels 0,1,2,3 --seeds-per-level 10 --n-obs 300 --n-int 150 \
        --budget-slack 0 --inferencers meek --no-e2e
    # (b) does the diagnosis survive when observational data is scarce / plentiful?
    run ablation_n60  --levels 1,2 --seeds-per-level 10 --n-obs 60  --n-int 40
    run ablation_n1000 --levels 1,2 --seeds-per-level 10 --n-obs 1000 --n-int 500
    # (c) does the LLM inferencer improve when it sees raw rows instead of sufficient statistics?
    run ablation_rawevidence --levels 1,2 --seeds-per-level 10 --n-obs 300 --n-int 150 \
        --evidence-mode raw --selectors oracle,eig,llm --inferencers llm --no-e2e
    # (d) is the diagnosis an artefact of PC's significance threshold?
    for A in 0.01 0.10; do
        run "ablation_alpha${A}" --levels 1,2 --seeds-per-level 10 --n-obs 300 --n-int 150 \
            --alpha "$A" --no-e2e
    done
    # (e) the largest graph size
    run ablation_d12 --levels 4 --seeds-per-level 8 --n-obs 300 --n-int 150
    ;;
verifier)
    # Model-free, deterministic, seconds per run: how much of the mean-shift rule's residual
    # error is the threshold rather than the evidence, and does abstaining help?
    for Z in 1.282 1.645 1.960 2.576 3.291; do
        "$PY" run_study1_decompose.py --out-dir "study1/ablation_verifier/z${Z}" \
            --levels 0,1,2,3 --seeds-per-level 10 --n-obs 300 --n-int 150 \
            --selectors random,maxdeg,eig,oracle --inferencers meek --no-e2e --models "" \
            --meanshift-z "$Z"
    done
    for Z in 1.645 1.960 2.576; do
        "$PY" run_study1_decompose.py --out-dir "study1/ablation_verifier/abstain_z${Z}" \
            --levels 0,1,2,3 --seeds-per-level 10 --n-obs 300 --n-int 150 \
            --selectors random,maxdeg,eig,oracle --inferencers meek --no-e2e --models "" \
            --meanshift-z "$Z" --meanshift-abstain 1.0
    done
    # per-decision |Z| log behind the reliability curve (also model-free)
    for SEL in oracle random maxdeg eig; do
        SUFFIX=""; [ "$SEL" = oracle ] || SUFFIX="_$SEL"
        "$PY" run_study1_localreadout.py --mode mechanical --selector "$SEL" \
            --seed-map-from study1/main/run_manifest.json \
            --out-dir "study1/localreadout/mechanical${SUFFIX}"
    done
    "$PY" scripts/make_rauma_figure_verifier.py --study-dir study1 --out-dir figures
    ;;
local)
    # Does the readout gap survive when the LLM only has to decide ONE edge from ONE tuple?
    # Same instances, same targets, same neighbourhoods as the mechanical arm; the LLM is
    # substituted at the single point where evidence becomes an arrow. Three prompt conditions
    # vary how much of the causal rule it is handed.
    for M in qwen3-coder-30b gpt-4o-mini; do
        for PROMPT in stats rule rule_z; do
            "$PY" run_study1_localreadout.py --mode llm --prompt "$PROMPT" --models "$M" \
                --seed-map-from study1/main/run_manifest.json --workers "$WORKERS" --resume \
                --out-dir "study1/localreadout/${M}_${PROMPT}"
        done
    done
    ;;
rawevidence-paired)
    # The evidence-format ablation, re-run on the MAIN study's instances so the summary-vs-raw
    # contrast is paired instance by instance instead of comparing two separate draws.
    run ablation_rawevidence_paired --levels 1,2 --seeds-per-level 10 --n-obs 300 --n-int 150 \
        --seed-map-from study1/main/run_manifest.json \
        --evidence-mode raw --selectors oracle,eig,llm --inferencers llm --no-e2e
    ;;
all)
    bash "$0" main
    bash "$0" ablation
    ;;
*)
    echo "unknown stage '$STAGE' (expected: smoke | main | ablation | verifier | local | rawevidence-paired | all)" >&2
    exit 1
    ;;
esac

echo
echo "[study1] done. Tables and figures are in traces/study1/*/analysis/"
