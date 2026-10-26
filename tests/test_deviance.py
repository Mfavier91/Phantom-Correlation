import numpy as np
import pytest
import scipy.sparse as sp
from scipy.special import expit

from phantom.deviance import deviances


def test_deviance_matches_direct_sum_over_pairs():
    rng = np.random.default_rng(0)
    U, n = 3000, 25
    h = np.repeat(rng.normal(-2, 0.8, size=n // 5), 5)        # items share log-odds in groups of 5
    alpha, w = np.array([-1.0, 0.5, 1.5]), np.array([0.5, 0.3, 0.2])
    theta = rng.choice(alpha, size=U, p=w)
    X = (rng.random((U, n)) < expit(h[None, :] + theta[:, None])).astype(float)
    X = X[:, (X.sum(axis=0) > 0) & (X.sum(axis=0) < U)]
    h = h[: X.shape[1]]
    N = X.T @ X
    p = X.mean(axis=0)
    P = expit(h[None, :] + alpha[:, None])
    mu = {"activity": U * (P * w[:, None]).T @ P, "independence": U * np.outer(p, p)}
    i, j = np.triu_indices(X.shape[1], 1)
    direct = {}
    for k, M in mu.items():
        nn, mm = N[i, j], M[i, j]
        direct[k] = 2 * np.sum(np.where(nn > 0, nn * np.log(np.where(nn > 0, nn, 1) / mm), 0.0) - (nn - mm))
    got = deviances(sp.csr_matrix(X), h, alpha, w, chunk=7)
    for k in direct:
        assert got[k] == pytest.approx(direct[k], rel=1e-9)
