"""Poisson deviance of the co-occurrence counts over all item pairs (paper, Equation (deviance)).

    D = 2 sum_{i<j} [ n_ij log(n_ij / mu_ij) - (n_ij - mu_ij) ],

n_ij observed co-occurrence count, mu_ij = U p_ij^model the expected count. Only pairs with n_ij > 0
need individual terms; sum n_ij and sum mu_ij come in closed form:
    sum_{i<j} n_ij  = sum_u S_u (S_u - 1) / 2,
    sum_{i<j} mu_ij = U/2 [ sum_g w_g tau_g^2 - sum_g w_g sum_i P(X_i=1|theta_g)^2 ],  tau_g = sum_i P(X_i=1|theta_g).
Items with the same log-odds (activity model) or the same count (independence model) share their
expected counts, so mu_ij is a lookup in a (distinct values)^2 table: this works for any catalog size.
"""

import numpy as np
from scipy.special import expit


def deviances(X, h, alpha, w, chunk=1000):
    """Deviance of the activity model (h, alpha, w) and of the independence model (popularities of X)
    on the data X (U x n, binary). Returns {"activity": D_act, "independence": D_ind}."""
    X = X.tocsc()
    U, n = X.shape
    C = np.asarray(X.sum(axis=0)).ravel()
    S = np.asarray(X.sum(axis=1)).ravel()
    hu, inv_h = np.unique(h, return_inverse=True)
    F = expit(hu[None, :] + alpha[:, None])                      # P(X_i = 1 | theta_g), per distinct h
    Ta = U * (F * w[:, None]).T @ F
    Cu, inv_c = np.unique(C, return_inverse=True)
    pu = Cu / U
    Ti = U * np.outer(pu, pu)
    mh, mc = np.bincount(inv_h), np.bincount(inv_c)
    tau, t2 = F @ mh, (F ** 2) @ mh
    smu = {"activity": U / 2 * (w @ tau ** 2 - w @ t2), "independence": U / 2 * ((pu @ mc) ** 2 - (pu ** 2) @ mc)}
    sn = float((S * (S - 1)).sum() / 2)
    nlog = {"activity": 0.0, "independence": 0.0}
    XT = X.T.tocsr()
    for j0 in range(0, n, chunk):
        A = (XT @ X[:, j0:j0 + chunk]).tocoo()                 # n x chunk, entries n_ij
        i, j, v = A.row, A.col + j0, A.data
        keep = i < j
        i, j, v = i[keep], j[keep], v[keep]
        nlog["activity"] += float(np.sum(v * np.log(v / Ta[inv_h[i], inv_h[j]])))
        nlog["independence"] += float(np.sum(v * np.log(v / Ti[inv_c[i], inv_c[j]])))
    return {k: 2 * (nlog[k] - sn + smu[k]) for k in nlog}
