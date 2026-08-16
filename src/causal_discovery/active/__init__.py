"""Active-experiment extensions to ACDB.

Two studies live here:

* **Study 1** (`run_study1_decompose.py`) factorises an active causal-discovery agent
  into an *experiment selector* and a *structure inferencer* and runs the full
  cross-product, so the end-to-end gap can be attributed to one or the other.
* **Study 2** (`run_study2_probe.py`) is PROBE: the LLM proposes a hypothesis space,
  and exact Bayesian experimental design chooses and interprets the experiments.
"""

from causal_discovery.active.gaussian import (
    GaussianParams,
    bic_score,
    fit_linear_gaussian,
    implied_mean_cov,
    interventional_loglik,
)
from causal_discovery.active.levels import LEVELS, LevelSpec, build_instance, build_seed_map
from causal_discovery.active.llm_client import (
    OpenRouterClient,
    UsageTotals,
    resolve_api_key,
    resolve_model,
    short_model_name,
)
from causal_discovery.active.mec import enumerate_mec, mec_entropy
from causal_discovery.active.pdag import (
    intervention_value,
    meek_closure,
    open_targets,
    orient_from_intervention,
    orient_from_truth,
)
from causal_discovery.active.state import BeliefState, Evidence

__all__ = [
    "BeliefState",
    "Evidence",
    "GaussianParams",
    "LEVELS",
    "LevelSpec",
    "OpenRouterClient",
    "UsageTotals",
    "bic_score",
    "build_instance",
    "build_seed_map",
    "enumerate_mec",
    "fit_linear_gaussian",
    "implied_mean_cov",
    "intervention_value",
    "interventional_loglik",
    "mec_entropy",
    "meek_closure",
    "open_targets",
    "orient_from_intervention",
    "orient_from_truth",
    "resolve_api_key",
    "resolve_model",
    "short_model_name",
]
