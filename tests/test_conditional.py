import itertools

import numpy as np
import pytest

import phantom.conditional as cond
from phantom.conditional import _inclusion


def brute_force_inclusion(o, k):
    """P(item i in a basket of size k) under P(x | S = k) = prod o^x / e_k(o), by enumeration."""
    baskets = np.array([b for b in itertools.product([0, 1], repeat=len(o)) if sum(b) == k])
    w = np.prod(np.where(baskets, o, 1.0), axis=1)
    return (w @ baskets) / w.sum()


def test_inclusion_matches_brute_force():
    rng = np.random.default_rng(0)
    hu = np.log(rng.uniform(0.05, 3.0, size=5))
    mult = np.array([1, 2, 1, 3, 1])                 # equal odds share one entry
    o = np.exp(np.repeat(hu, mult))
    kmax = int(mult.sum())
    pi, _, _ = _inclusion(hu, mult, kmax)
    for k in range(1, kmax + 1):
        bf = brute_force_inclusion(o, k)
        assert pi[k] == pytest.approx(bf[np.cumsum(mult) - 1], rel=1e-9)


def test_inclusion_sums_to_basket_size():
    rng = np.random.default_rng(1)
    hu = np.sort(rng.normal(-3, 1.5, size=40))
    mult = rng.integers(1, 4, size=40)
    kmax = 30
    pi, _, _ = _inclusion(hu, mult, kmax)
    assert (pi * mult).sum(axis=1) == pytest.approx(np.arange(kmax + 1), abs=1e-8)


def test_large_catalog_path_matches_standard_path(monkeypatch):
    rng = np.random.default_rng(3)
    hu = np.concatenate([rng.normal(-3, 1, size=60), [2.5, 3.0]])   # a few very popular items need the backward pass
    mult = np.ones(len(hu), dtype=int)
    mult[-2] = 2
    pi_std, _, _ = _inclusion(hu, mult, 40)
    monkeypatch.setattr(cond, "LARGE_CATALOG", 10)
    pi_big, _, _ = _inclusion(hu, mult, 40)
    assert pi_big == pytest.approx(pi_std, rel=1e-8, abs=1e-14)
