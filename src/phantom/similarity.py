"""Item-item association measures computed from joint and marginal probabilities.

All functions take co[i, j] = P(X_i = 1, X_j = 1) and p[i] = P(X_i = 1), so the same code
applies to observed data (co = D^T D / U) and to model-implied quantities.
"""

import numpy as np


def observed_cooccurrence(X):
    """co = X^T X / U for a (dense or scipy.sparse) binary U x n matrix X."""
    U = X.shape[0]
    co = (X.T @ X) / U
    co = co.toarray() if hasattr(co, "toarray") else np.asarray(co, float)
    p = np.asarray(X.sum(axis=0)).ravel() / U
    return co, p


def upper_pairs(n):
    return np.triu_indices(n, k=1)


def pair_measures(co, p, pairs=None):
    """Dict of association measures for item pairs (upper triangle by default)."""
    if pairs is None:
        pairs = upper_pairs(len(p))
    i, j = pairs
    pij = co[i, j]
    pi, pj = p[i], p[j]
    cov = pij - pi * pj
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = pij / (pi * pj)
        return {
            "cooc": pij,
            "cov": cov,
            "corr": cov / np.sqrt(pi * (1 - pi) * pj * (1 - pj)),
            "lift": lift,
            "pmi": np.log(lift),
            "cosine": pij / np.sqrt(pi * pj),
            "jaccard": pij / (pi + pj - pij),
        }
