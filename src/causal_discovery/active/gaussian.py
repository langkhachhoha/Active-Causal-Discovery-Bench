"""Exact linear-Gaussian scoring for candidate DAGs.

Used by PROBE (study 2) to turn a set of LLM-proposed graphs into a *posterior*,
and to compute the exact interventional likelihood that drives the Bayes update.

Model
-----
    X_j = c_j + sum_{i in Pa(j)} B[i, j] X_i + eps_j ,   eps_j ~ N(0, s2_j)

In matrix form `X = B^T X + c + eps`, hence

    mu    = (I - B^T)^{-1} c
    Sigma = (I - B^T)^{-1} diag(s2) (I - B^T)^{-T}

A hard intervention `do(a = v)` mutilates the model: column `a` of `B` is zeroed,
`c_a = v` and `s2_a = 0`. The likelihood is then evaluated on the coordinates
`!= a` only (coordinate `a` is deterministic).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from causal_discovery.core import DAG

_JITTER = 1e-9
_LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True, slots=True)
class GaussianParams:
    """Maximum-likelihood parameters of a linear-Gaussian SCM on a fixed DAG."""

    weights: np.ndarray  # B, shape (d, d), B[i, j] = coefficient of X_i in eq. of X_j
    intercepts: np.ndarray  # c, shape (d,)
    noise_var: np.ndarray  # s2, shape (d,)
    loglik: float
    num_params: int


def fit_linear_gaussian(data: np.ndarray, dag: DAG, *, ridge: float = 1e-8) -> GaussianParams:
    """Per-node OLS fit; returns MLE parameters plus the observational log-likelihood."""
    x = np.asarray(data, dtype=float)
    n, d = x.shape
    if d != dag.num_nodes:
        raise ValueError(f"data has {d} columns but DAG has {dag.num_nodes} nodes")

    weights = np.zeros((d, d), dtype=float)
    intercepts = np.zeros(d, dtype=float)
    noise_var = np.zeros(d, dtype=float)
    loglik = 0.0
    num_params = 2 * d  # intercepts + variances

    for node in range(d):
        parents = dag.parents(node)
        y = x[:, node]
        if parents:
            design = np.hstack([np.ones((n, 1)), x[:, list(parents)]])
        else:
            design = np.ones((n, 1))
        gram = design.T @ design + ridge * np.eye(design.shape[1])
        coeffs = np.linalg.solve(gram, design.T @ y)
        residual = y - design @ coeffs
        sigma2 = float(residual @ residual) / n
        sigma2 = max(sigma2, 1e-10)

        intercepts[node] = float(coeffs[0])
        for slot, parent in enumerate(parents, start=1):
            weights[parent, node] = float(coeffs[slot])
        noise_var[node] = sigma2
        num_params += len(parents)
        loglik += -0.5 * n * (_LOG_2PI + np.log(sigma2) + 1.0)

    return GaussianParams(
        weights=weights,
        intercepts=intercepts,
        noise_var=noise_var,
        loglik=float(loglik),
        num_params=int(num_params),
    )


def bic_score(data: np.ndarray, dag: DAG, *, params: GaussianParams | None = None) -> float:
    """BIC = -2 logL + p log n. Lower is better."""
    fitted = params if params is not None else fit_linear_gaussian(data, dag)
    n = int(np.asarray(data).shape[0])
    return float(-2.0 * fitted.loglik + fitted.num_params * np.log(max(n, 2)))


def implied_mean_cov(
    params: GaussianParams,
    *,
    intervene: tuple[int, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Model-implied mean and covariance, optionally under a hard intervention."""
    b = np.array(params.weights, dtype=float, copy=True)
    c = np.array(params.intercepts, dtype=float, copy=True)
    s2 = np.array(params.noise_var, dtype=float, copy=True)
    d = b.shape[0]

    if intervene is not None:
        target, value = intervene
        b[:, target] = 0.0
        c[target] = float(value)
        s2[target] = 0.0

    inverse = np.linalg.inv(np.eye(d) - b.T)
    mean = inverse @ c
    cov = inverse @ np.diag(s2) @ inverse.T
    return mean, cov


def _gaussian_loglik(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """Sum of log N(x_row; mean, cov) over rows, with a PSD-safe fallback."""
    n, k = x.shape
    cov = 0.5 * (cov + cov.T) + _JITTER * np.eye(k)
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues = np.clip(eigenvalues, 1e-10, None)
        cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        chol = np.linalg.cholesky(cov)
    log_det = 2.0 * float(np.sum(np.log(np.diag(chol))))
    delta = (x - mean).T
    solved = np.linalg.solve(chol, delta)
    quad = float(np.sum(solved * solved))
    return -0.5 * (n * (k * _LOG_2PI + log_det) + quad)


def interventional_loglik(
    int_data: np.ndarray,
    params: GaussianParams,
    target: int,
    value: float,
) -> float:
    """log p(interventional sample | DAG, params, do(target = value)).

    Evaluated on the non-intervened coordinates, which is where all the
    discriminative signal lives (the intervened column is deterministic).
    """
    x = np.asarray(int_data, dtype=float)
    d = x.shape[1]
    keep = [j for j in range(d) if j != target]
    mean, cov = implied_mean_cov(params, intervene=(target, float(value)))
    return _gaussian_loglik(x[:, keep], mean[keep], cov[np.ix_(keep, keep)])


def sample_interventional(
    params: GaussianParams,
    target: int,
    value: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a predictive interventional batch (non-intervened coordinates only)."""
    d = params.weights.shape[0]
    keep = [j for j in range(d) if j != target]
    mean, cov = implied_mean_cov(params, intervene=(target, float(value)))
    sub = cov[np.ix_(keep, keep)]
    sub = 0.5 * (sub + sub.T) + _JITTER * np.eye(len(keep))
    return rng.multivariate_normal(mean[keep], sub, size=n_samples, method="cholesky")


def loglik_of_batch(
    batch: np.ndarray,
    params: GaussianParams,
    target: int,
    value: float,
) -> float:
    """log-likelihood of an already-reduced batch (columns != target, in order)."""
    d = params.weights.shape[0]
    keep = [j for j in range(d) if j != target]
    mean, cov = implied_mean_cov(params, intervene=(target, float(value)))
    return _gaussian_loglik(batch, mean[keep], cov[np.ix_(keep, keep)])


class LocalBicCache:
    """Decomposable BIC with memoised local scores.

    `BIC(G) = sum_j local(j, Pa_G(j))`, so scoring thousands of orientations of a few
    skeletons costs only a few hundred regressions.
    """

    def __init__(self, data: np.ndarray, *, ridge: float = 1e-8) -> None:
        self._x = np.asarray(data, dtype=float)
        self._n = int(self._x.shape[0])
        self._d = int(self._x.shape[1])
        self._ridge = float(ridge)
        self._log_n = float(np.log(max(self._n, 2)))
        self._cache: dict[tuple[int, frozenset[int]], float] = {}

    def local(self, node: int, parents: tuple[int, ...]) -> float:
        key = (node, frozenset(parents))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        y = self._x[:, node]
        if parents:
            design = np.hstack([np.ones((self._n, 1)), self._x[:, list(parents)]])
        else:
            design = np.ones((self._n, 1))
        gram = design.T @ design + self._ridge * np.eye(design.shape[1])
        coeffs = np.linalg.solve(gram, design.T @ y)
        residual = y - design @ coeffs
        sigma2 = max(float(residual @ residual) / self._n, 1e-10)
        score = self._n * (_LOG_2PI + np.log(sigma2) + 1.0) + (len(parents) + 2) * self._log_n
        self._cache[key] = float(score)
        return float(score)

    def score(self, dag: DAG) -> float:
        parents: dict[int, list[int]] = {node: [] for node in range(self._d)}
        for src, dst in dag.edges:
            parents[dst].append(src)
        return float(sum(self.local(node, tuple(sorted(ps))) for node, ps in parents.items()))


def normalise_log_weights(log_weights: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over log-weights."""
    lw = np.asarray(log_weights, dtype=float)
    if lw.size == 0:
        return lw
    shifted = lw - float(np.max(lw))
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        return np.full(lw.shape, 1.0 / lw.size)
    return weights / total
