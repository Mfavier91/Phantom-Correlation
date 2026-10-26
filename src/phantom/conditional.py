"""Inclusion probabilities of the items in a basket of given size (Section 5.2 of the paper).

Given its size k, a basket follows the conditional Bernoulli distribution
    P(x | S = k) = prod_i o_i^{x_i} / e_k(o),   o_i = exp(h_i),
whatever the activity distribution. The first-order inclusion probabilities
pi_i(k) = o_i e_{k-1}(o_{-i}) / e_k(o) use the exact recursion in both directions
(r_k = e_{k-1}(o)/e_k(o), increasing in k by log-concavity):
    forward   pi_i(k)   = o_i r_k (1 - pi_i(k-1)),        stable while o_i r_k < 1
    backward  pi_i(k-1) = 1 - pi_i(k) / (o_i r_k),        stable while o_i r_k > 1, pi_i(n) = 1
Used for the popularity of a user's items as a function of the number of items consumed
(Appendix B of the paper).
"""

import numpy as np

from .model import log_elementary_symmetric_grouped


def _log_esp_grouped(hu, mult, kmax):
    """log e_k(o), k = 0..kmax, for odds exp(hu) with multiplicities mult."""
    return log_elementary_symmetric_grouped(hu, mult, kmax)


def _inclusion(hu, mult, kmax, want_pairs=False):
    """Inclusion probabilities given the degree, for distinct log-odds hu (multiplicities mult).

    Returns (pi, gg, le): pi[k, u] = pi_u(k); if want_pairs, gg[k, u] = probability that two
    distinct items with odds exp(hu[u]) are both in a basket of size k; le = log e_k(o).
    Uses the forward/backward recursions described in the module docstring."""
    n = int(mult.sum())
    o = np.exp(hu)

    def ratios(K):
        le = _log_esp_grouped(hu, mult, K)
        return le, np.concatenate([[np.nan], np.exp(le[:-1] - le[1:])])  # r[k] = e_{k-1}/e_k

    # log e_k up to kmax is enough unless some item needs the backward pass (o r_k >= 1),
    # which starts from k = n; only then compute up to n.
    le, r = ratios(kmax)
    switch = np.array([np.searchsorted(o_u * r[1:kmax + 1], 1.0) + 1 for o_u in o])  # first k with o r_k >= 1
    if np.any(switch <= kmax):
        if n > LARGE_CATALOG:
            return _exact_for_unstable(hu, mult, kmax, le, r, switch <= kmax, want_pairs)
        le, r = ratios(n)

    with np.errstate(over="ignore", invalid="ignore"):  # the discarded halves may overflow
        return _passes(o, r, n, kmax, switch, want_pairs) + (le,)


LARGE_CATALOG = 100_000


def _log_times_linear(le, log_o):
    """log-coefficients of P(t) * (1 + o t), truncated to the same length."""
    out = le.copy()
    out[1:] = np.logaddexp(le[1:], le[:-1] + log_o)
    return out


def _exact_for_unstable(hu, mult, kmax, le, r, unstable, want_pairs):
    """Large catalogs: forward recursion for the stable items, exact log-space leave-one-out
    (and leave-two-out for ties) for the few unstable ones, H. We compute log e_k without
    all copies of H once, then multiply back the other factors of H; no subtractions."""
    o = np.exp(hu)
    with np.errstate(over="ignore", invalid="ignore"):
        pi, gg = _passes(o, r, int(mult.sum()), kmax, np.full(len(o), kmax + 1), want_pairs)
    H = np.where(unstable)[0]
    rest = np.ones(len(hu), bool)
    rest[H] = False
    le_rest = _log_esp_grouped(hu[rest], mult[rest], kmax)
    for u in H:
        loo = le_rest
        for v in H:
            for _ in range(mult[v] - (v == u)):
                loo = _log_times_linear(loo, hu[v])
        pi[1:, u] = np.exp(hu[u] + loo[:-1] - le[1:])
        if want_pairs and mult[u] >= 2:
            # remove one more copy of u: divide is unstable, so rebuild without two copies
            loo2 = le_rest
            for v in H:
                for _ in range(mult[v] - 2 * (v == u)):
                    loo2 = _log_times_linear(loo2, hu[v])
            gg[2:, u] = np.exp(2 * hu[u] + loo2[:-2] - le[2:])
            gg[:2, u] = 0.0
    return pi, gg, le


def _passes(o, r, n, kmax, switch, want_pairs):
    # forward pass, k = 1..kmax (values beyond each item's switch point are discarded)
    pi_f = np.zeros((kmax + 1, len(o)))
    gg_f = np.zeros((kmax + 1, len(o)))
    for k in range(1, kmax + 1):
        pi_f[k] = o * r[k] * (1 - pi_f[k - 1])
        if want_pairs:
            gg_f[k] = o * r[k] * (pi_f[k - 1] - gg_f[k - 1])
    pi, gg = pi_f, gg_f
    if np.any(switch <= kmax):
        # backward pass from k = n down to 1 for the items that need it
        need = switch <= kmax
        ob = o[need]
        p_b = np.ones(need.sum())
        g_b = np.ones(need.sum())
        pi_b = np.zeros((kmax + 1, need.sum()))
        gg_b = np.zeros((kmax + 1, need.sum()))
        for k in range(n, 0, -1):
            if k <= kmax:
                pi_b[k], gg_b[k] = p_b, g_b
            p_prev = 1 - p_b / (ob * r[k])
            if want_pairs:
                g_b = p_prev - g_b / (ob * r[k])
            p_b = p_prev
        ks = np.arange(kmax + 1)[:, None]
        use_b = ks >= switch[need][None, :]
        pi[:, need] = np.where(use_b, pi_b, pi_f[:, need])
        if want_pairs:
            gg[:, need] = np.where(use_b, gg_b, gg_f[:, need])
    return pi, gg
