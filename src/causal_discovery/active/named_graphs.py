"""Real-world DAG structures with their published variable names.

The semantic experiment needs graphs where the variable names carry genuine domain
meaning, so that hiding the names removes real information rather than nothing. We take
the *structure and node names* of the small networks in the bnlearn repository and
re-parameterize them as linear-Gaussian SCMs, exactly as the synthetic ladder does. The
conditional distributions are ours; the graph and the vocabulary are the published ones.

Only the naming condition differs between the two arms of the experiment: identical DAG,
identical parameters, identical samples, identical node indices. Anything that moves is
attributable to the words alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from causal_discovery.config import make_v1_config
from causal_discovery.core import DAG, Permutation
from causal_discovery.benchmark.instance import BenchmarkInstance
from causal_discovery.equivalence import (
    compute_minimum_intervention_set,
    dag_to_cpdag,
)
from causal_discovery.sampling import sample_observational_data
from causal_discovery.scm import parameterize_linear_gaussian_scm


@dataclass(frozen=True, slots=True)
class NamedGraph:
    name: str
    domain: str
    nodes: tuple[str, ...]
    arcs: tuple[tuple[str, str], ...]

    @property
    def d(self) -> int:
        return len(self.nodes)

    @property
    def k(self) -> int:
        return len(self.arcs)

    def to_dag(self) -> DAG:
        index = {name: i for i, name in enumerate(self.nodes)}
        return DAG.from_edges(self.d, {(index[a], index[b]) for a, b in self.arcs})


NAMED_GRAPHS: dict[str, NamedGraph] = {
    "cancer": NamedGraph(
        name="cancer",
        domain="a small medical model of lung cancer risk and its symptoms",
        nodes=("Pollution", "Smoker", "Cancer", "Xray", "Dyspnoea"),
        arcs=(
            ("Pollution", "Cancer"),
            ("Smoker", "Cancer"),
            ("Cancer", "Xray"),
            ("Cancer", "Dyspnoea"),
        ),
    ),
    "earthquake": NamedGraph(
        name="earthquake",
        domain="a home burglar-alarm model",
        nodes=("Burglary", "Earthquake", "Alarm", "JohnCalls", "MaryCalls"),
        arcs=(
            ("Burglary", "Alarm"),
            ("Earthquake", "Alarm"),
            ("Alarm", "JohnCalls"),
            ("Alarm", "MaryCalls"),
        ),
    ),
    "survey": NamedGraph(
        name="survey",
        domain="a transport-usage survey of commuters",
        nodes=("Age", "Sex", "Education", "Occupation", "Residence", "Travel"),
        arcs=(
            ("Age", "Education"),
            ("Sex", "Education"),
            ("Education", "Occupation"),
            ("Education", "Residence"),
            ("Occupation", "Travel"),
            ("Residence", "Travel"),
        ),
    ),
    "asia": NamedGraph(
        name="asia",
        domain="a chest-clinic model of lung disease and its symptoms",
        nodes=("asia", "tub", "smoke", "lung", "bronc", "either", "xray", "dysp"),
        arcs=(
            ("asia", "tub"),
            ("smoke", "lung"),
            ("smoke", "bronc"),
            ("tub", "either"),
            ("lung", "either"),
            ("either", "xray"),
            ("either", "dysp"),
            ("bronc", "dysp"),
        ),
    ),
    "sachs": NamedGraph(
        name="sachs",
        domain="a protein-signalling network measured by flow cytometry",
        nodes=("Akt", "Erk", "Jnk", "Mek", "P38", "PIP2", "PIP3", "PKA", "PKC", "Plcg", "Raf"),
        arcs=(
            ("PKC", "Jnk"), ("PKC", "P38"), ("PKC", "Raf"), ("PKC", "Mek"), ("PKC", "PKA"),
            ("PKA", "Jnk"), ("PKA", "P38"), ("PKA", "Raf"), ("PKA", "Mek"),
            ("PKA", "Erk"), ("PKA", "Akt"),
            ("Raf", "Mek"), ("Mek", "Erk"), ("Erk", "Akt"),
            ("Plcg", "PIP2"), ("Plcg", "PIP3"), ("PIP3", "PIP2"),
        ),
    ),
}

DEFAULT_GRAPHS = ("cancer", "earthquake", "survey", "asia", "sachs")


def parse_graph_names(text: str) -> list[str]:
    names = [x.strip() for x in text.split(",") if x.strip()]
    if not names:
        raise ValueError("no graphs selected")
    for name in names:
        if name not in NAMED_GRAPHS:
            raise ValueError(f"unknown graph {name!r}; available: {sorted(NAMED_GRAPHS)}")
    return names


def build_named_instance(
    graph: NamedGraph,
    seed: int,
    n_obs: int,
    n_int: int,
    budget_slack: int = 1,
) -> tuple[BenchmarkInstance, tuple[str, ...]]:
    """One instance on a fixed published structure, with a fresh parameterization.

    The node order is permuted per seed so that a bare index carries no information about
    the published ordering; the returned name tuple is permuted with it, so name `i`
    always labels column `i` of the data. Returns `(instance, variable_names)`.
    """
    rng = np.random.default_rng(seed)
    config = make_v1_config(
        d=graph.d, k=graph.k, n_obs=n_obs, n_int=n_int, budget_slack=budget_slack
    )
    dag_internal = graph.to_dag()

    permutation = Permutation.from_mapping(rng.permutation(graph.d))
    dag_public = dag_internal.relabel(permutation)
    names_public = [""] * graph.d
    for old_index, name in enumerate(graph.nodes):
        names_public[permutation.apply_node(old_index)] = name

    scm_public = parameterize_linear_gaussian_scm(
        dag_public, config.weight_range, config.noise_var, rng
    )
    cpdag_public = dag_to_cpdag(dag_public)
    intervention_set = compute_minimum_intervention_set(dag_public, cpdag_public)
    observational_data = sample_observational_data(scm_public, config.n_obs, rng)

    instance = BenchmarkInstance(
        config=config,
        true_dag=dag_public,
        observational_ceiling=cpdag_public,
        scm=scm_public,
        optimal_intervention_set=intervention_set,
        observational_data=observational_data,
        intervention_budget=len(intervention_set) + budget_slack,
        label_permutation=permutation,
    )
    return instance, tuple(names_public)
