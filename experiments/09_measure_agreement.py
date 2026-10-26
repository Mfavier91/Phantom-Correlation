"""Measure-by-measure agreement between the data and the models over blocks of items (paper, Section 6,
"Each measure separately" and "Popularity surface"; Tables 2 and 3).

Blocks: on MovieLens-1M (>= 4) the films with >= 50 users in blocks of 14 consecutive films (by popularity)
plus the four tail buckets (as in Figure 1); on the other datasets popularity groups with equal shares of
interactions (40 target). Block measures are ratios of sums over the pairs i != j (lift, cosine, Pearson).
Only block pairs with >= 5 expected co-occurrences under the activity model are used.
For each measure (lift and cosine on log scale, Pearson raw), with per-block noise variances sigma_b^2 from
--reps datasets simulated from the fitted activity model:
    Z = mean_b (y_b - m_b)^2 / sigma_b^2,     share = (Z_ind - Z_act) / (Z_ind - Z_bar),
Z_bar the mean Z of the simulated datasets (leave-one-out). Popularity surface: least-squares fit of
log(block measure) = a + b (x + y) + c x y + d (x + y)^2, x, y = log mean popularity of the two blocks centered
at the median, observed and under each model.

Usage: python experiments/09_measure_agreement.py --datasets ml1m_r4 ml20m_r4 lastfm2k amazon_books
Output: results/measure_agreement.json
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phantom.blocks import groups_for  # noqa: E402
from phantom.data import LOADERS  # noqa: E402
from phantom.paths import RESULTS  # noqa: E402
from phantom.runinfo import save_json  # noqa: E402
from phantom.simulate_fast import simulate_fast  # noqa: E402

OBS, ACT, IND = "observed", "activity", "independence"


def surface(vals, x, y):
    """Popularity surface (Section 6, Equation (surface)): least-squares fit of
    log(value) = a + b (x + y) + c x y + d (x + y)^2 per model, x, y = centered log popularities of the two blocks.
    Values <= 0 have no logarithm: they are left out and counted ("nonpositive")."""
    out = {}
    for k, v in vals.items():
        ok = v > 0
        if not ok.any():                              # Pearson under independence (0 everywhere)
            out[k] = None
            continue
        if np.ptp(v[ok]) < 1e-12:                     # lift under independence (1 everywhere)
            out[k] = {"nonpositive": int((~ok).sum()),
                      "quadratic": {"a": float(np.log(v[ok][0])), "b": 0.0, "c": 0.0, "d": 0.0, "R2": None}}
            continue
        s = x[ok] + y[ok]
        Q = np.c_[np.ones(ok.sum()), s, x[ok] * y[ok], s ** 2]
        cq = np.linalg.lstsq(Q, np.log(v[ok]), rcond=None)[0]
        rq = np.log(v[ok]) - Q @ cq
        out[k] = {"nonpositive": int((~ok).sum()),
                  "quadratic": {"a": float(cq[0]), "b": float(cq[1]), "c": float(cq[2]), "d": float(cq[3]),
                                "R2": float(1 - rq.var() / np.log(v[ok]).var())}}
    return out


def run(tag, reps=0, seed0=5000):
    X, _, _, _ = LOADERS[tag]()
    z = np.load(RESULTS / tag / "model.npz")
    X = X[:, z["fit_items"]].tocsr()
    U, n = X.shape
    C = np.asarray(X.sum(axis=0)).ravel()
    p = C / U
    b = groups_for(tag, C)
    K = b.max() + 1
    Ind = sp.csr_matrix((np.ones(n), (np.arange(n), b)), shape=(n, K))
    def block_co(Xm):
        nub = (Xm @ Ind).toarray()
        return (nub.T @ nub - np.diag(nub.sum(axis=0))) / U
    co = {OBS: block_co(X)}
    P = expit(z["h"][None, :] + z["alpha"][:, None])
    for name, Pm, w in [(IND, p[None, :], np.ones(1)), (ACT, P, z["w"])]:
        tau, t2 = Pm @ Ind, (Pm ** 2) @ Ind
        co[name] = (tau * w[:, None]).T @ tau - np.diag(w @ t2)

    def pair_sum(v):
        s = v @ Ind
        return np.outer(s, s) - np.diag((v ** 2) @ Ind)
    chance, sq, sdp = pair_sum(p), pair_sum(np.sqrt(p)), pair_sum(np.sqrt(p * (1 - p)))
    keep = np.triu(co[ACT] * U >= 5)
    # popularity surface: x, y = log mean popularity of the two blocks, centered at the median
    pA = np.log((p @ Ind) / np.asarray(Ind.sum(axis=0)).ravel())
    m0 = np.median(pA[np.nonzero(keep)[0].tolist() + np.nonzero(keep)[1].tolist()])
    xb = (pA[:, None] - m0 + 0 * pA[None, :])[keep]
    yb = (pA[None, :] - m0 + 0 * pA[:, None])[keep]
    raw = {"lift": lambda c: c / chance, "cosine": lambda c: c / sq, "Pearson": lambda c: (c - chance) / sdp}
    meas = {"lift": lambda c: np.log(c / chance), "cosine": lambda c: np.log(c / sq),
            "Pearson": lambda c: (c - chance) / sdp}
    out = {"blocks": int(K), "block_pairs_used": int(keep.sum())}
    # reference: datasets simulated from the fitted activity model, treated like observed data (their own popularities)
    sims = []
    for r in range(reps):
        Xs, _ = simulate_fast(z["h"], z["alpha"], z["w"], U, np.random.default_rng(seed0 + r))
        ps = np.asarray(Xs.sum(axis=0)).ravel() / U
        cs = block_co(Xs.tocsr())
        sims.append((cs, pair_sum(ps), pair_sum(np.sqrt(ps)), pair_sum(np.sqrt(ps * (1 - ps)))))
    meas_s = {"lift": lambda c, ch, q, sd: np.log(c / ch), "cosine": lambda c, ch, q, sd: np.log(c / q),
              "Pearson": lambda c, ch, q, sd: (c - ch) / sd}
    for mname, f in meas.items():
        v = {k: f(c)[keep] for k, c in co.items()}
        out.setdefault("log_trend", {})[mname] = surface({k: raw[mname](c)[keep] for k, c in co.items()}, xb, yb)
        out[mname] = {}
        if sims:
            # standardized error (Section 6): per-block noise variance from the simulations, mean over blocks of
            # (obs - model)^2 / sigma_b^2; ~1 if the data look like activity plus noise
            VS = np.array([meas_s[mname](cs, ch, q, sd)[keep] for cs, ch, q, sd in sims])
            sig2 = np.mean((VS - v[ACT]) ** 2, axis=0)
            z_act = float(np.mean((v[OBS] - v[ACT]) ** 2 / sig2))
            z_ind = float(np.mean((v[OBS] - v[IND]) ** 2 / sig2))
            z_sim = []
            for r in range(len(VS)):                          # leave-one-out noise for the simulated reference
                loo = np.mean((np.delete(VS, r, axis=0) - v[ACT]) ** 2, axis=0)
                z_sim.append(float(np.mean((VS[r] - v[ACT]) ** 2 / loo)))
            zs = float(np.mean(z_sim))
            out[mname]["standardized_error"] = {"Z_activity": z_act, "Z_independence": z_ind,
                                                "Z_simulated_mean": zs, "Z_simulated_range": [min(z_sim), max(z_sim)],
                                                "share": (z_ind - z_act) / (z_ind - zs)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ml1m_r4"])
    ap.add_argument("--reps", type=int, default=10, help="datasets simulated from the fitted activity model (reference)")
    a = ap.parse_args()
    res = {}
    for d in a.datasets:
        res[d] = run(d, a.reps)
        r = res[d]
        print(f"{d}: {r['blocks']} blocks, {r['block_pairs_used']} block pairs")
        for m in ["lift", "cosine", "Pearson"]:
            zz = r[m].get("standardized_error")
            if zz:
                print(f"  {m:8s} Z act {zz['Z_activity']:.3g}  Z ind {zz['Z_independence']:.3g}  "
                      f"Z sim {zz['Z_simulated_mean']:.2f}  share {zz['share']:.3f}")
            for side in ["observed", "activity"]:
                q = r["log_trend"][m][side]
                if q:
                    c = q["quadratic"]
                    print(f"  {m:8s} surface {side:8s} a {c['a']:+.2f} b {c['b']:+.3f} c {c['c']:+.3f} d {c['d']:+.3f}"
                          f"  (left out: {q['nonpositive']})")
    save_json(RESULTS / "measure_agreement.json", res, config={"datasets": a.datasets}, seed=None)


if __name__ == "__main__":
    main()
