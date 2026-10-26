"""EASE (Steck 2019) from a Gram matrix, and the choice of its regularization strength.

EASE needs only the Gram matrix G = X^T X:
    P = (G + lambda I)^{-1},   B_ij = -P_ij / P_jj (i != j),   B_jj = 0.
A model has no interaction matrix, but it has an expected Gram matrix, U p_ij (off-diagonal) and
U p_i (diagonal), so EASE can be computed on it without sampling (paper, Section 5.3.3).
"""

import numpy as np
import scipy.sparse as sp
from scipy.linalg import inv as la_inv


def ease(G, lam):
    """EASE closed form (Steck 2019, eq. 8), in float32 and in place (G is overwritten): one n x n
    matrix plus the inversion workspace, so that catalogs of ~20k items fit in memory."""
    G = G.astype(np.float32, copy=False)
    G[np.diag_indices_from(G)] += lam
    P = la_inv(G, overwrite_a=True, check_finite=False)
    d = np.diag(P).copy()
    P /= -d[None, :]
    np.fill_diagonal(P, 0.0)
    return P


def choose_lambda(X, lams, rng, k=20, holdout=0.2):
    """Hold out a share of each user's items; recall@k of the held-out items for every lambda."""
    X = X.tocsr()
    rows, cols = X.nonzero()
    test = rng.random(len(rows)) < holdout
    Xtr = sp.csr_matrix((np.ones((~test).sum()), (rows[~test], cols[~test])), shape=X.shape)
    Xte = sp.csr_matrix((np.ones(test.sum()), (rows[test], cols[test])), shape=X.shape)
    Xtr32 = Xtr.astype(np.float32)
    G = (Xtr32.T @ Xtr32).toarray()                      # float32 throughout (memory: ML-20M)
    users = np.where(np.asarray(Xte.sum(axis=1)).ravel() > 0)[0]
    res = {}
    for lam in lams:
        B = ease(G.copy(), lam)
        rec = []
        for u0 in range(0, len(users), 5000):            # score users in chunks (dense users x items)
            uu = users[u0:u0 + 5000]
            S = Xtr32[uu] @ B
            S[Xtr32[uu].nonzero()] = -np.inf             # never recommend training items
            top = np.argpartition(-S, k, axis=1)[:, :k]
            hits = np.asarray(Xte[uu][np.arange(len(uu))[:, None], top].sum(axis=1)).ravel()
            rec.append(hits / np.minimum(k, np.asarray(Xte[uu].sum(axis=1)).ravel()))
        del B
        res[lam] = float(np.mean(np.concatenate(rec)))
        print(f"  lambda {lam}: recall@{k} {res[lam]:.4f}", flush=True)
    return max(res, key=res.get), res
