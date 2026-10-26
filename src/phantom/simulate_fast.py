"""Fast exact simulation from the activity model when many items share the same log-odds h.

Items with equal h are exchangeable: given a user's activity level alpha_g, the number of consumed items among
the m_c items of group c is Binomial(m_c, logistic(h_c + alpha_g)), and which ones is a uniformly random subset
of that size. Drawing the counts per (user, group) and then the subsets is exact and much cheaper than one
Bernoulli draw per (user, item) when the catalog has few distinct h values (Amazon Books: 495k items, ~1k values).
"""

import numpy as np
import scipy.sparse as sp
from scipy.special import expit


def simulate_fast(h, alpha, w, U, rng, user_chunk=20000, exact_max_group=64):
    """Sparse U x n matrix X with X_ui ~ Bernoulli(logistic(h_i + alpha_u)), alpha_u ~ (alpha, w)."""
    h = np.asarray(h)
    n = len(h)
    hu, inv = np.unique(h, return_inverse=True)
    m = np.bincount(inv)
    order = np.argsort(inv, kind="stable")                        # items grouped by h value
    offset = np.concatenate([[0], np.cumsum(m)])
    types = np.repeat(np.arange(len(alpha)), rng.multinomial(U, w))
    rows_out, cols_out = [], []
    for u0 in range(0, U, user_chunk):
        t = types[u0:u0 + user_chunk]
        K = rng.binomial(m[None, :], expit(hu[None, :] + alpha[t][:, None]))   # users x groups
        rr, cc = np.nonzero(K)
        kk = K[rr, cc]
        exact = (m[cc] <= exact_max_group) | (kk * 4 > m[cc])
        # exact uniform subsets: random keys, keep the kk smallest (per group value, vectorized over users)
        for c in np.unique(cc[exact]):
            sel = np.where(exact & (cc == c))[0]
            keys = rng.random((len(sel), m[c]))
            rank = np.argsort(np.argsort(keys, axis=1), axis=1)
            r_idx, pos = np.nonzero(rank < kk[sel][:, None])
            rows_out.append(rr[sel][r_idx] + u0)
            cols_out.append(order[offset[c] + pos])
        # large groups, few items: draw positions, redraw duplicates within the same (user, group) until none
        big = np.where(~exact)[0]
        if len(big):
            r = np.repeat(rr[big], kk[big])
            c = np.repeat(cc[big], kk[big])
            pos = (rng.random(len(r)) * m[c]).astype(np.int64)
            while True:
                key = (r.astype(np.int64) * len(m) + c) * n + pos      # unique per (user, group, position)
                _, first = np.unique(key, return_index=True)
                dup = np.ones(len(key), bool)
                dup[first] = False
                if not dup.any():
                    break
                pos[dup] = (rng.random(dup.sum()) * m[c[dup]]).astype(np.int64)
            rows_out.append(r + u0)
            cols_out.append(order[offset[c] + pos])
    r, c = np.concatenate(rows_out), np.concatenate(cols_out)
    return sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(U, n)), types
