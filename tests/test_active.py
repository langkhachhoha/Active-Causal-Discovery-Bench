"""Correctness tests for the active-experiment layer.

Run with:  pytest tests/test_active.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_discovery import make_v1_config, build_benchmark_instance  # noqa: E402
from causal_discovery.core import DAG  # noqa: E402
from causal_discovery.equivalence import CPDAG, dag_to_cpdag  # noqa: E402
from causal_discovery.active.episode import run_pc, skeleton_ceiling_f1, truth_in_class  # noqa: E402
from causal_discovery.active.gaussian import (  # noqa: E402
    bic_score,
    fit_linear_gaussian,
    implied_mean_cov,
    interventional_loglik,
    normalise_log_weights,
)
from causal_discovery.active.levels import LEVELS, build_instance  # noqa: E402
from causal_discovery.active.mec import edge_marginals, enumerate_mec, mec_entropy  # noqa: E402
from causal_discovery.active.pdag import (  # noqa: E402
    dag_v_structures,
    intervention_value,
    meek_closure,
    open_targets,
    orient_from_intervention,
    orient_from_truth,
    v_structures,
)
from causal_discovery.active.probe import (  # noqa: E402
    build_posterior,
    expected_information_gain,
    submission_from_posterior,
)
from causal_discovery.active.selectors import (  # noqa: E402
    MaxDegreeSelector,
    OracleSelector,
    RandomSelector,
    EIGSelector,
    mec_target_entropy,
    true_gain_per_target,
)
from causal_discovery.active.state import BeliefState  # noqa: E402
from causal_discovery.sampling import sample_interventional_data, sample_observational_data  # noqa: E402


# --------------------------------------------------------------------------- #
# MEC enumeration
# --------------------------------------------------------------------------- #
def test_mec_of_chain_has_three_members():
    """X0 -> X1 -> X2 has MEC {chain, reverse chain, fork} — the classic size-3 class."""
    chain = DAG.from_edges(3, [(0, 1), (1, 2)])
    cpdag = dag_to_cpdag(chain)
    members, exhaustive = enumerate_mec(cpdag)
    assert exhaustive
    assert len(members) == 3
    edge_sets = {frozenset(m.edges) for m in members}
    assert frozenset({(0, 1), (1, 2)}) in edge_sets  # chain
    assert frozenset({(2, 1), (1, 0)}) in edge_sets  # reversed chain
    assert frozenset({(1, 0), (1, 2)}) in edge_sets  # fork


def test_mec_of_collider_is_singleton():
    """A v-structure is fully identified by observational data."""
    collider = DAG.from_edges(3, [(0, 1), (2, 1)])
    cpdag = dag_to_cpdag(collider)
    assert cpdag.num_undirected_edges == 0
    members, exhaustive = enumerate_mec(cpdag)
    assert exhaustive and len(members) == 1
    assert members[0].edges == collider.edges


def test_every_mec_member_shares_skeleton_and_v_structures():
    for seed in range(6):
        instance = build_instance(LEVELS[1], 1000 + seed, n_obs=100, n_int=50)
        cpdag = instance.observational_ceiling
        members, _ = enumerate_mec(cpdag, max_members=256)
        assert members, "MEC must be non-empty"
        reference = v_structures(cpdag)
        skeleton = {tuple(sorted(e)) for e in cpdag.undirected_edges} | {
            tuple(sorted(e)) for e in cpdag.directed_edges
        }
        for member in members:
            assert {tuple(sorted(e)) for e in member.edges} == skeleton
            assert dag_v_structures(member) == reference
        # the true DAG must be one of them
        assert any(m.edges == instance.true_dag.edges for m in members)


# --------------------------------------------------------------------------- #
# orientation / regret bookkeeping
# --------------------------------------------------------------------------- #
def test_oracle_intervention_set_fully_resolves():
    """Intervening on every element of I* must leave zero undirected edges."""
    for seed in range(6):
        instance = build_instance(LEVELS[1], 2000 + seed, n_obs=100, n_int=50)
        pdag = instance.observational_ceiling
        for target in instance.optimal_intervention_set:
            pdag, _ = orient_from_truth(pdag, target, instance.true_dag)
        assert pdag.num_undirected_edges == 0
        assert pdag.directed_edges == instance.true_dag.edges


def test_true_gain_is_nonnegative_and_bounded():
    instance = build_instance(LEVELS[2], 4242, n_obs=100, n_int=50)
    pdag = instance.observational_ceiling
    gains = true_gain_per_target(pdag, instance.true_dag)
    assert set(gains) == set(open_targets(pdag))
    for target, gain in gains.items():
        assert 0 <= gain <= pdag.num_undirected_edges


def test_mean_shift_orientation_recovers_truth_with_clean_data():
    """With plenty of interventional samples the mean-shift rule should be right."""
    instance = build_instance(LEVELS[1], 777, n_obs=2000, n_int=2000)
    rng = np.random.default_rng(0)
    obs = sample_observational_data(instance.scm, 2000, rng)
    pdag = instance.observational_ceiling
    for target in instance.optimal_intervention_set:
        value = intervention_value(obs, target)
        data = sample_interventional_data(instance.scm, target, value, 2000, rng)
        pdag, _ = orient_from_intervention(pdag, target, obs, data)
    assert pdag.num_undirected_edges == 0
    assert pdag.directed_edges == instance.true_dag.edges


def test_eig_prefers_informative_targets():
    """On a 3-chain, intervening on the middle node resolves everything at once."""
    chain = DAG.from_edges(3, [(0, 1), (1, 2)])
    cpdag = dag_to_cpdag(chain)
    eig, size, entropy, exhaustive = mec_target_entropy(cpdag)
    assert size == 3 and exhaustive
    assert entropy == pytest.approx(np.log(3))
    # X1 touches both undirected edges and separates all three members
    assert eig[1] == pytest.approx(np.log(3))
    assert eig[1] > eig[0] and eig[1] > eig[2]


def test_eig_selector_matches_manual_argmax():
    instance = build_instance(LEVELS[1], 31337, n_obs=100, n_int=50)
    state = BeliefState.create(instance.observational_ceiling, instance.observational_data, 3)
    selector = EIGSelector(np.random.default_rng(0))
    choice = selector.choose(state)
    eig, _, _, _ = mec_target_entropy(instance.observational_ceiling)
    assert eig[choice.target] == pytest.approx(max(eig.values()))


# --------------------------------------------------------------------------- #
# linear-Gaussian scoring
# --------------------------------------------------------------------------- #
def test_fit_recovers_weights_on_large_sample():
    instance = build_instance(LEVELS[1], 55, n_obs=100, n_int=50)
    rng = np.random.default_rng(3)
    data = sample_observational_data(instance.scm, 20000, rng)
    params = fit_linear_gaussian(data, instance.true_dag)
    for src, dst in instance.true_dag.edges:
        assert params.weights[src, dst] == pytest.approx(instance.scm.weights[src, dst], abs=0.05)


def test_implied_moments_match_empirical():
    instance = build_instance(LEVELS[0], 91, n_obs=100, n_int=50)
    rng = np.random.default_rng(5)
    data = sample_observational_data(instance.scm, 40000, rng)
    params = fit_linear_gaussian(data, instance.true_dag)
    mean, cov = implied_mean_cov(params)
    assert np.allclose(mean, data.mean(axis=0), atol=0.05)
    assert np.allclose(cov, np.cov(data, rowvar=False), atol=0.1)


def test_intervened_coordinate_is_deterministic_in_implied_moments():
    instance = build_instance(LEVELS[0], 92, n_obs=100, n_int=50)
    data = sample_observational_data(instance.scm, 2000, np.random.default_rng(1))
    params = fit_linear_gaussian(data, instance.true_dag)
    mean, cov = implied_mean_cov(params, intervene=(1, 7.5))
    assert mean[1] == pytest.approx(7.5)
    assert np.allclose(cov[1, :], 0.0) and np.allclose(cov[:, 1], 0.0)


def test_true_dag_wins_on_interventional_likelihood():
    """The Bayes update must be able to separate Markov-equivalent hypotheses."""
    wins = 0
    trials = 0
    for seed in range(8):
        instance = build_instance(LEVELS[1], 6000 + seed, n_obs=400, n_int=400)
        rng = np.random.default_rng(seed)
        obs = sample_observational_data(instance.scm, 400, rng)
        members, _ = enumerate_mec(instance.observational_ceiling, max_members=64)
        if len(members) < 2:
            continue
        params = [fit_linear_gaussian(obs, dag) for dag in members]
        log_weights = np.zeros(len(members))
        for target in instance.optimal_intervention_set:
            value = intervention_value(obs, target)
            int_data = sample_interventional_data(instance.scm, target, value, 400, rng)
            log_weights = log_weights + np.array(
                [interventional_loglik(int_data, p, target, value) for p in params]
            )
        best = int(np.argmax(log_weights))
        trials += 1
        wins += int(members[best].edges == instance.true_dag.edges)
    assert trials >= 4
    assert wins == trials, f"interventional likelihood picked the wrong DAG in {trials - wins}/{trials} cases"


def test_bic_prefers_the_true_dag_over_a_denser_one():
    instance = build_instance(LEVELS[1], 4321, n_obs=100, n_int=50)
    data = sample_observational_data(instance.scm, 3000, np.random.default_rng(2))
    true_bic = bic_score(data, instance.true_dag)
    extra = set(instance.true_dag.edges)
    for i in range(instance.config.d):
        for j in range(i + 1, instance.config.d):
            if (i, j) not in extra and (j, i) not in extra:
                candidate = DAG.from_edges(instance.config.d, extra | {(i, j)})
                assert bic_score(data, candidate) > true_bic
                return
    pytest.skip("graph is complete; no denser alternative exists")


def test_normalise_log_weights_is_stable():
    weights = normalise_log_weights(np.array([-1e5, -1e5 + 1.0, -1e5 + 2.0]))
    assert np.isclose(weights.sum(), 1.0)
    assert weights[2] > weights[1] > weights[0]


# --------------------------------------------------------------------------- #
# PROBE plumbing
# --------------------------------------------------------------------------- #
def test_posterior_and_map_submission_are_valid():
    instance = build_instance(LEVELS[1], 8080, n_obs=200, n_int=100)
    obs = instance.observational_data
    members, _ = enumerate_mec(instance.observational_ceiling, max_members=32)
    posterior = build_posterior(list(members), obs, use_bic=True)
    assert np.isclose(posterior.weights.sum(), 1.0)
    assert posterior.rank_of(instance.true_dag) >= 1
    submission = submission_from_posterior(posterior, instance.config.d, "map")
    assert submission.undirected_edges == frozenset()
    marginal = submission_from_posterior(posterior, instance.config.d, "marginal")
    assert marginal.num_nodes == instance.config.d


def test_expected_information_gain_is_nonnegative():
    instance = build_instance(LEVELS[1], 9090, n_obs=200, n_int=100)
    obs = instance.observational_data
    members, _ = enumerate_mec(instance.observational_ceiling, max_members=16)
    posterior = build_posterior(list(members), obs, use_bic=True)
    rng = np.random.default_rng(0)
    for target in open_targets(instance.observational_ceiling):
        value = intervention_value(obs, target)
        gain = expected_information_gain(posterior, target, value, 100, rng, num_outcomes=8)
        assert gain >= -1e-6
        assert gain <= posterior.entropy + 1e-6


def test_edge_marginals_sum_to_weights():
    chain = DAG.from_edges(3, [(0, 1), (1, 2)])
    members, _ = enumerate_mec(dag_to_cpdag(chain))
    weights = np.full(len(members), 1.0 / len(members))
    marginals = edge_marginals(members, weights, 3)
    # every member has exactly 2 edges, so the total mass is 2
    assert marginals.sum() == pytest.approx(2.0)


def test_mec_entropy_matches_log_of_uniform_size():
    assert mec_entropy(np.array([0.25, 0.25, 0.25, 0.25])) == pytest.approx(np.log(4))
    assert mec_entropy(np.array([1.0, 0.0])) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# PC front-end helpers
# --------------------------------------------------------------------------- #
def test_pc_front_end_returns_a_valid_cpdag():
    instance = build_instance(LEVELS[1], 12345, n_obs=500, n_int=200)
    pdag = run_pc(instance.observational_data, alpha=0.05)
    assert isinstance(pdag, CPDAG)
    assert pdag.num_nodes == instance.config.d
    assert 0.0 <= skeleton_ceiling_f1(pdag, instance.true_dag) <= 1.0


def test_truth_in_class_is_true_for_the_true_cpdag():
    instance = build_instance(LEVELS[1], 4711, n_obs=200, n_int=100)
    assert truth_in_class(instance.observational_ceiling, instance.true_dag)
    wrong = CPDAG(num_nodes=instance.config.d, directed_edges=frozenset(), undirected_edges=frozenset())
    assert not truth_in_class(wrong, instance.true_dag)


def test_meek_closure_is_idempotent():
    instance = build_instance(LEVELS[2], 606, n_obs=100, n_int=50)
    once = meek_closure(instance.observational_ceiling)
    twice = meek_closure(once)
    assert once.directed_edges == twice.directed_edges
    assert once.undirected_edges == twice.undirected_edges


# --------------------------------------------------------------------------- #
# selectors (non-LLM) run end to end
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("factory", [
    lambda inst: RandomSelector(np.random.default_rng(0)),
    lambda inst: MaxDegreeSelector(),
    lambda inst: EIGSelector(np.random.default_rng(0)),
    lambda inst: OracleSelector(inst.true_dag),
])
def test_selectors_return_valid_targets(factory):
    instance = build_instance(LEVELS[2], 202, n_obs=200, n_int=100)
    state = BeliefState.create(instance.observational_ceiling, instance.observational_data, 3)
    selector = factory(instance)
    choice = selector.choose(state)
    assert 0 <= choice.target < instance.config.d
