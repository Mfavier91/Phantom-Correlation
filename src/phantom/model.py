"""Independent-activity model.

Users consume items independently; a user's activity alpha_u shifts the log-odds of every
item by the same amount (Rasch model, Proposition 1 of the paper):

    P(X_ui = 1 | alpha_u) = sigmoid(h_i + alpha_u)

The activity distribution P(alpha) is nonparametric: weights w on a fixed grid of alpha
values (NPMLE of a mixing distribution). Because the posterior of alpha_u given x_u depends
on x_u only through the degree S_u, the model is fitted from two sufficient statistics:
the item counts C_i and the degree histogram.
"""

import warnings

import numpy as np
from scipy.special import expit, gammaln, logsumexp


def softplus(x):
    return np.logaddexp(0.0, x)


def _alpha_for_expected_degree(h, target):
    """alpha such that sum_i sigmoid(h_i + alpha) = target (bisection)."""
    lo, hi = -60.0, 60.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if expit(h + mid).sum() < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def make_alpha_grid(h, min_degree, max_degree, n_grid=200, pad=1.5):
    """Grid of alpha spanning expected degrees from below min_degree to above max_degree."""
    n = len(h)
    a_lo = _alpha_for_expected_degree(h, max(min_degree, 0.05)) - pad
    a_hi = _alpha_for_expected_degree(h, min(max(max_degree, 1.0), n - 0.05)) + pad
    return np.linspace(a_lo, a_hi, n_grid)


class ActivityModel:
    """Fitted independent-activity model: item log-odds h, activity grid alpha, weights w."""

    def __init__(self, h, alpha, w):
        self.h = np.asarray(h, float)
        self.alpha = np.asarray(alpha, float)
        self.w = np.asarray(w, float)

    @property
    def n_items(self):
        return len(self.h)

    def item_probs(self):
        """G x n matrix of P(X_i = 1 | alpha_g)."""
        return expit(self.h[None, :] + self.alpha[:, None])

    def marginals(self):
        """P(X_i = 1) = sum_g w_g sigmoid(h_i + alpha_g)."""
        return self.w @ self.item_probs()

    def cooccurrence(self, items=None):
        """Model-implied P(X_i = 1, X_j = 1) = sum_g w_g p_i(alpha_g) p_j(alpha_g).

        Items are conditionally independent given alpha, so the joint is a mixture of
        products (Proposition 2 of the paper)."""
        P = self.item_probs()
        if items is not None:
            P = P[:, items]
        co = (P * self.w[:, None]).T @ P
        np.fill_diagonal(co, self.w @ P)  # X_i^2 = X_i
        return co

    def total_pair_covariance(self):
        """sum_{i<j} Cov(X_i, X_j) under the model, without forming the n x n matrix:
        sum_{i != j} p_ij = sum_g w_g [(sum_i s_ig)^2 - sum_i s_ig^2] with s_ig = sigmoid(h_i + alpha_g)."""
        P = self.item_probs()
        joint = self.w @ (P.sum(axis=1) ** 2 - (P**2).sum(axis=1))
        p = self.w @ P
        return 0.5 * (joint - (p.sum() ** 2 - (p**2).sum()))

    def degree_pmf(self, kmax=None):
        """Model P(S = k), k = 0..kmax (default n): sum_g w_g e_k(o) gamma_g^k / prod_i (1 + gamma_g o_i).

        The cost is O(n * kmax); pass kmax = largest observed degree for large catalogs."""
        kmax = self.n_items if kmax is None else min(int(kmax), self.n_items)
        hu, mult = np.unique(self.h, return_counts=True)
        log_e = log_elementary_symmetric_grouped(hu, mult, kmax)
        A = softplus(self.h[None, :] + self.alpha[:, None]).sum(axis=1)
        k = np.arange(kmax + 1)
        with np.errstate(divide="ignore"):
            log_w = np.log(self.w)
        L = log_w[:, None] + self.alpha[:, None] * k[None, :] - A[:, None]
        return np.exp(log_e + logsumexp(L, axis=0))

    def activity_moments(self):
        """Mean and variance of alpha and of gamma = exp(alpha) under w."""
        g = np.exp(self.alpha)
        m_a = self.w @ self.alpha
        m_g = self.w @ g
        return {
            "alpha_mean": m_a,
            "alpha_var": self.w @ (self.alpha - m_a) ** 2,
            "gamma_mean": m_g,
            "gamma_cv2": (self.w @ (g - m_g) ** 2) / m_g**2,
        }


def log_elementary_symmetric(log_o, kmax):
    """log e_k(o) for k = 0..kmax via the O(n * kmax) recursion, in log-space."""
    le = np.full(kmax + 1, -np.inf)
    le[0] = 0.0
    for lo in log_o:
        le[1:] = np.logaddexp(le[1:], le[:-1] + lo)
    return le


def log_elementary_symmetric_grouped(log_o_u, mult, kmax):
    """log e_k(o), k = 0..kmax, when the odds exp(log_o_u[u]) occur mult[u] times.

    Equivalent to log_elementary_symmetric(np.repeat(log_o_u, mult), kmax). Small groups
    multiply by (1 + o t) one factor at a time; a group with m >= kmax copies multiplies by
    (1 + o t)^m = sum_j C(m, j) o^j t^j in one log-space convolution.
    """
    le = np.full(kmax + 1, -np.inf)
    le[0] = 0.0
    k = np.arange(kmax + 1)
    for lo, m in zip(log_o_u, mult):
        if m < kmax:
            for _ in range(int(m)):
                le[1:] = np.logaddexp(le[1:], le[:-1] + lo)
        else:
            log_binom = gammaln(m + 1) - gammaln(k + 1) - gammaln(m - k + 1) + k * lo  # j = 0..kmax
            shifted = np.where(k[:, None] >= k[None, :], le[np.clip(k[:, None] - k[None, :], 0, kmax)], -np.inf)
            le = logsumexp(shifted + log_binom[None, :], axis=1)
    return le


def fit_activity_model(item_counts, degree_counts, alpha_grid=None, n_grid=200,
                       max_iter=20000, tol=1e-10, verbose=False):
    """Fit (h, w) by EM on a fixed alpha grid (marginal maximum likelihood).

    item_counts: C_i, number of users who consumed item i (0 < C_i < U required).
    degree_counts: degree_counts[k] = number of users with degree k.

    E-step: posterior over grid points for a user of degree k,
        r_gk ∝ w_g exp(alpha_g k - sum_i softplus(h_i + alpha_g)).
    M-step: w_g = N_g / U with N_g = sum_k count_k r_gk, and h_i solves
        sum_g N_g sigmoid(h_i + alpha_g) = C_i   (one Newton step per iteration).
    At convergence the model reproduces every item count exactly.
    """
    C = np.asarray(item_counts, float)
    dc = np.asarray(degree_counts, float)
    U = dc.sum()
    if np.any(C <= 0) or np.any(C >= U):
        raise ValueError("every item needs 0 < count < number of users")

    k_all = np.arange(len(dc))
    nz = dc > 0
    k, cnt = k_all[nz], dc[nz]

    # Items with the same count solve the same M-step equation, so they share the same h.
    # Work with one h per distinct count (weighted by multiplicity): identical result, much faster.
    Cu, inv, mult = np.unique(C, return_inverse=True, return_counts=True)

    hu = np.log(Cu / (U - Cu))
    if alpha_grid is None:
        alpha_grid = make_alpha_grid(hu[inv], k.min(), k.max(), n_grid)
    alpha = np.asarray(alpha_grid, float)
    w = np.full(len(alpha), 1.0 / len(alpha))

    ll_old = -np.inf
    for it in range(max_iter):
        A = softplus(hu[None, :] + alpha[:, None]) @ mult
        with np.errstate(divide="ignore"):
            L = np.log(w)[:, None] + alpha[:, None] * k[None, :] - A[:, None]
        lse = logsumexp(L, axis=0)
        ll = (mult * Cu * hu).sum() + (cnt * lse).sum()  # marginal log-likelihood (up to a constant)
        N = np.exp(L - lse) @ cnt
        w = N / U

        P = expit(hu[None, :] + alpha[:, None])
        f = N @ P - Cu
        fp = N @ (P * (1 - P))
        hu = hu - np.clip(f / fp, -5, 5)

        rel_change = (ll - ll_old) / abs(ll)
        max_count_err = np.max(np.abs(f) / Cu)
        if verbose and it % 500 == 0:
            print(f"iter {it}: ll={ll:.6f} rel_change={rel_change:.2e} count_err={max_count_err:.2e}")
        if it > 10 and abs(rel_change) < tol and max_count_err < 1e-6:
            break
        ll_old = ll
    else:
        warnings.warn(f"EM did not converge in {max_iter} iterations "
                      f"(rel_change={rel_change:.2e}, count_err={max_count_err:.2e})")
    h = hu[inv]

    if w[0] > 1e-3 or w[-1] > 1e-3:
        warnings.warn(f"activity mass at grid edge (w[0]={w[0]:.2e}, w[-1]={w[-1]:.2e}); widen the grid")

    model = ActivityModel(h, alpha, w)
    model.n_iter = it + 1
    model.log_likelihood = ll
    return model


def simulate(h, alpha_users, rng):
    """Dense boolean U x n interaction matrix from the independent-activity model."""
    p = expit(np.asarray(alpha_users)[:, None] + np.asarray(h)[None, :])
    return rng.random(p.shape) < p
