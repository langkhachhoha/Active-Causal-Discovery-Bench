#!/usr/bin/env bash
# Study 2, revision round — the three experiments the review asks for.
#
#   bash scripts/study2b.sh smoke    # ~3 min, ~$0.05 — check everything works
#   bash scripts/study2b.sh perm     # ~1-2 h,  $0     — count-matched permutation controls
#   bash scripts/study2b.sh ranker   # ~30 min, $0     — statistical ranker + true-skeleton ceiling
#   bash scripts/study2b.sh sepset   # ~1-2 h,  ~$5    — corrected-prompt A/B
#   bash scripts/study2b.sh all      # everything, in dependency order
#
# `perm` and `ranker` make no API calls at all. Safe to re-run: everything resumes.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME="${ACDB_ENV:-acdb-active}"
PY="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)' 2>/dev/null || echo python)"
CORE_MODELS="${ACDB_MODELS:-qwen3-coder-30b,gpt-4o-mini}"
SWEEP_MODELS="${ACDB_SWEEP_MODELS:-qwen3-coder-30b,gpt-4o-mini,gpt-5.4-mini,haiku-4.5,gemini-3-flash}"
WORKERS="${ACDB_WORKERS:-12}"
PERM_WORKERS="${ACDB_PERM_WORKERS:-24}"
DRAWS="${ACDB_DRAWS:-200}"
SRC_ROOT="${ACDB_SRC_ROOT:-study2_new}"     # where the recorded proposals live
OUT_ROOT="${ACDB_OUT_ROOT:-study2b}"
STAGE="${1:-smoke}"

# The full ladder, now with two extra rungs: a data-only ranker that executes our own
# stated rule, and the true adjacency set at any edit distance.
LADDER="probe_skel_only,probe_random_edits,probe_stat_edits,probe_oracle_edits,probe_true_skeleton"

banner() {
    echo
    echo "=============================================================="
    echo " study2b :: $1"
    echo "=============================================================="
}

case "$STAGE" in
smoke)
    banner "smoke (all three stages, tiny)"
    "$PY" run_study2_probe.py --out-dir "${OUT_ROOT}/smoke" --workers 2 --resume \
        --models gpt-4o-mini --levels 1 --seeds-per-level 2 --n-obs 60 --n-int 60 \
        --arms "probe,probe_sepset,${LADDER}"
    "$PY" scripts/nemchua_permutation.py "${OUT_ROOT}/smoke" --draws 5 --workers 4
    "$PY" scripts/nemchua_edit_audit.py "${OUT_ROOT}/smoke"
    ;;

perm)
    # Review point 1. The random-editor arm in the paper draws a fixed four removals and
    # four additions, which is the cap rather than what any model actually does; models
    # that propose less were being compared against a control that proposes more. This
    # rebuilds the control at each model's *realized* counts, many times over, so the
    # LLM's score can be placed inside the null distribution its own volume generates.
    # No API calls: every proposal is replayed from the recorded events.
    banner "permutation control :: main cohort (n_obs=300)"
    "$PY" scripts/nemchua_permutation.py "${SRC_ROOT}/main_v2" \
        --draws "$DRAWS" --workers "$PERM_WORKERS"
    banner "permutation control :: five-model cohort (n_obs=60)"
    "$PY" scripts/nemchua_permutation.py "${SRC_ROOT}/models_n60" \
        --draws "$DRAWS" --workers "$PERM_WORKERS"
    banner "permutation control :: data-rich five-model cohort (n_obs=300)"
    "$PY" scripts/nemchua_permutation.py "${SRC_ROOT}/models_n300" \
        --draws "$DRAWS" --workers "$PERM_WORKERS"
    ;;

ranker)
    # Two new rungs on the ladder, neither of which needs a model:
    #   probe_stat_edits    — our own instruction, executed mechanically by a ranker with
    #                         no world knowledge. Separates "the model knows something"
    #                         from "the rule we handed it is worth following".
    #   probe_true_skeleton — the true adjacency set at any edit distance, so whatever
    #                         accuracy is still missing is orientation error, not a budget.
    # These resume into the existing directories, so they are paired with everything there.
    for RUN in main_v2_fix models_n60_fix models_n300_fix robust_d12_fix; do
        [ -d "${SRC_ROOT}/${RUN}" ] || { echo "[skip] ${SRC_ROOT}/${RUN}"; continue; }
        banner "ladder rungs :: ${RUN}"
        "$PY" run_study2_probe.py --out-dir "${SRC_ROOT}/${RUN}" --workers "$WORKERS" --resume \
            --arms probe_stat_edits,probe_true_skeleton
    done
    # The Sachs question: is the oracle's failure there an orientation problem or an
    # edit-budget problem? `probe_true_skeleton` answers it directly.
    for RUN in semantic semantic_n60; do
        [ -d "${SRC_ROOT}/${RUN}" ] || { echo "[skip] ${SRC_ROOT}/${RUN}"; continue; }
        banner "ladder rungs :: ${RUN} (semantic)"
        "$PY" run_study2_semantic.py --out-dir "${SRC_ROOT}/${RUN}" --workers "$WORKERS" --resume \
            --arms probe_stat_edits,probe_true_skeleton
    done
    "$PY" scripts/nemchua_edit_audit.py "${SRC_ROOT}"/main_v2 "${SRC_ROOT}"/models_n60 \
        "${SRC_ROOT}"/models_n300 || true
    ;;

sepset)
    # Review point 2. The proposer prompt states that a nonzero partial correlation given
    # all other variables means adjacency. That is false for a DAG: the support of the
    # precision matrix is the *moral* graph, so two parents of a common child look
    # connected. The audit shows models' wrong additions concentrate on exactly those
    # pairs, and the better the model the more they concentrate — so the interface, not
    # the model, may be the binding constraint.
    #
    # This runs both prompts on the same instances with the same models. `probe` is rerun
    # here rather than reused, so both conditions see the current prompt text and the
    # only difference between them is the statistics and the adjacency rule.
    banner "corrected-prompt A/B :: five models, n_obs=60"
    "$PY" run_study2_probe.py --out-dir "${OUT_ROOT}/sepset_n60" --workers "$WORKERS" --resume \
        --models "$SWEEP_MODELS" --levels 1,2,3 --seeds-per-level 20 \
        --n-obs 60 --n-int 60 --preflight-seed 20260816 \
        --arms "probe,probe_sepset,${LADDER}"
    banner "corrected-prompt A/B :: two models, n_obs=300"
    "$PY" run_study2_probe.py --out-dir "${OUT_ROOT}/sepset_main" --workers "$WORKERS" --resume \
        --models "$CORE_MODELS" --levels 0,1,2,3 --seeds-per-level 20 \
        --n-obs 300 --n-int 150 --preflight-seed 20260816 \
        --arms "probe,probe_sepset,${LADDER}"
    banner "audit + permutation on the corrected interface"
    "$PY" scripts/nemchua_edit_audit.py "${OUT_ROOT}/sepset_n60" "${OUT_ROOT}/sepset_main"
    "$PY" scripts/nemchua_permutation.py "${OUT_ROOT}/sepset_n60" \
        --draws "$DRAWS" --workers "$PERM_WORKERS" --arm probe_sepset
    ;;

all)
    bash "$0" ranker
    bash "$0" sepset
    bash "$0" perm
    ;;
*)
    echo "unknown stage '$STAGE' (expected: smoke | perm | ranker | sepset | all)" >&2
    exit 1
    ;;
esac

echo
echo "[study2b] done."
