import itertools

import numpy as np
import pytest
from scipy.special import expit

from phantom.model import log_elementary_symmetric_grouped
from phantom import (ActivityModel, fit_activity_model, log_elementary_symmetric,
                     observed_cooccurrence, simulate)


def test_elementary_symmetric_matches_brute_force():
    rng = np.random.default_rng(0)
    o = rng.uniform(0.1, 3.0, size=7)
    le = log_elementary_symmetric(np.log(o), len(o))
    for k in range(len(o) + 1):
        brute = sum(np.prod(c) for c in itertools.combinations(o, k)) if k else 1.0
        assert np.exp(le[k]) == pytest.approx(brute, rel=1e-10)


def test_degree_pmf_matches_brute_force():
    # small model: enumerate all 2^n baskets
    h = np.array([-1.0, 0.3, -0.5, 1.2, -2.0])
    alpha = np.array([-1.0, 0.0, 2.0])
    w = np.array([0.5, 0.2, 0.3])
    m = ActivityModel(h, alpha, w)
    pmf = np.zeros(len(h) + 1)
    for x in itertools.product([0, 1], repeat=len(h)):
        x = np.array(x)
        p = expit(h[None, :] + alpha[:, None])
        prob = w @ np.prod(np.where(x, p, 1 - p), axis=1)
        pmf[x.sum()] += prob
    assert m.degree_pmf() == pytest.approx(pmf, abs=1e-12)


def test_cooccurrence_matches_simulation():
    rng = np.random.default_rng(1)
    h = rng.normal(-2, 0.7, size=20)
    alpha = np.array([-1.5, 0.0, 1.5])
    w = np.array([0.5, 0.3, 0.2])
    m = ActivityModel(h, alpha, w)
    users = rng.choice(alpha, size=200_000, p=w)
    co_obs, p_obs = observed_cooccurrence(simulate(h, users, rng).astype(float))
    assert m.marginals() == pytest.approx(p_obs, abs=3e-3)
    assert m.cooccurrence() == pytest.approx(co_obs, abs=2e-3)


def test_fit_reproduces_item_counts_and_degrees():
    rng = np.random.default_rng(2)
    h = rng.normal(-2.5, 0.8, size=60)
    users = rng.normal(0.0, 1.0, size=30_000)
    X = simulate(h, users, rng)
    C = X.sum(axis=0)
    deg = np.bincount(X.sum(axis=1), minlength=X.shape[1] + 1)
    m = fit_activity_model(C, deg)
    # item counts are reproduced exactly at convergence
    assert m.marginals() * X.shape[0] == pytest.approx(C, rel=1e-5)
    # degree distribution is close to the empirical one
    pmf_emp = deg / deg.sum()
    assert np.abs(m.degree_pmf() - pmf_emp).sum() < 0.05
    # item log-odds recovered up to the (h, alpha) shift
    assert np.corrcoef(m.h, h)[0, 1] > 0.99


def test_grouped_elementary_symmetric_matches_ungrouped():
    rng = np.random.default_rng(4)
    lo_u = rng.normal(-1, 1, size=6)
    mult = np.array([1, 3, 40, 2, 55, 1])  # two groups larger than kmax use the binomial path
    kmax = 30
    a = log_elementary_symmetric(np.repeat(lo_u, mult), kmax)
    b = log_elementary_symmetric_grouped(lo_u, mult, kmax)
    assert b == pytest.approx(a, rel=1e-12, abs=1e-12)
