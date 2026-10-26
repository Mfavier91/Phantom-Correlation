"""MovieLens-1M (ratings >= 4): every pair of films with >= --min-users users, and the tail in four buckets
(paper, Section 7.3).

Films with >= --min-users users are compared pair by pair; the rarer films are pooled into four buckets
equally spaced in log count (for 50: 1-2, 3-7, 8-18, 19-49 users). Bucket entries are ratios of sums,
e.g. lift(i, B) = sum_{j in B} p_ij / sum_{j in B} p_i p_j, computed exactly for the models from the
expected number of bucket items given the activity level. Models: independence (every user identical,
P(X_i = 1) = p_i) and the activity model (results/<d>/model.npz).

Usage: python experiments/03_movielens_pairs.py --dataset ml1m_r4 [--min-users 50]
Output: results/<d>/full_matrix.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit, xlogy
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phantom.blocks import tail_buckets  # noqa: E402
from phantom.data import LOADERS  # noqa: E402
from phantom.paths import RESULTS  # noqa: E402
from phantom.runinfo import save_json  # noqa: E402

IND, ACT = "independence", "activity model"


def load_models(tag, C, U):
    """Logits per activity level and weights: independence (one level) and the fitted activity model."""
    z = np.load(RESULTS / tag / "model.npz")
    p = C / U
    return {IND: (np.log(p / (1 - p))[None, :], np.ones(1)),      # every user identical: P(X_i = 1) = p_i
            ACT: (z["h"][None, :] + z["alpha"][:, None], z["w"])}


def run(tag, min_users):
    X, _, _, _ = LOADERS[tag]()
    fz = np.load(RESULTS / tag / "model.npz")
    X = X[:, fz["fit_items"]].tocsc()
    U = X.shape[0]
    C = np.asarray(X.sum(axis=0)).ravel()
    order = np.argsort(-C, kind="stable")
    dense = order[C[order] >= min_users]
    bucket, labels = tail_buckets(C, min_users)
    K = len(labels)
    p = C / U
    # observed co-occurrence: dense x dense, dense x bucket, bucket x bucket (ordered pairs i != j)
    Xd = X[:, dense]
    n_ub = np.stack([np.asarray(X[:, bucket == b].sum(axis=1)).ravel() for b in range(K)], axis=1)
    obs = {"dd": (Xd.T @ Xd).toarray() / U, "db": np.asarray(Xd.T @ n_ub) / U,
           "bb": (n_ub.T @ n_ub - np.diag(n_ub.sum(axis=0))) / U}
    np.fill_diagonal(obs["dd"], p[dense])
    # chance term p_i p_j (lift = co-occurrence / chance)
    pb = np.array([p[bucket == b].sum() for b in range(K)])
    p2b = np.array([(p[bucket == b] ** 2).sum() for b in range(K)])
    pd = p[dense]
    ind = {"dd": np.outer(pd, pd), "db": np.outer(pd, pb), "bb": np.outer(pb, pb) - np.diag(p2b)}
    mod = {}
    for name, (L, w) in load_models(tag, C, U).items():
        P = expit(L)                                                     # P(X_i = 1 | grid point g)
        Pd = P[:, dense]
        tau = np.stack([P[:, bucket == b].sum(axis=1) for b in range(K)], axis=1)   # E[# bucket items | g]
        t2 = np.stack([(P[:, bucket == b] ** 2).sum(axis=1) for b in range(K)], axis=1)
        dd = (Pd * w[:, None]).T @ Pd
        np.fill_diagonal(dd, w @ Pd)
        mod[name] = {"dd": dd, "db": (Pd * w[:, None]).T @ tau, "bb": (tau * w[:, None]).T @ tau - np.diag(w @ t2)}
    res = summary(obs, mod, ind, pd, U, bucket, labels, C, dense, min_users)
    # effective number of users behind the co-occurrences within each bucket:
    # (sum_u c_u)^2 / sum_u c_u^2 with c_u = n_ub (n_ub - 1), the ordered pairs user u contributes
    for b, r in enumerate(res["buckets"]):
        c_u = n_ub[:, b] * (n_ub[:, b] - 1.0)
        r["effective_users_within"] = float(c_u.sum() ** 2 / (c_u ** 2).sum()) if c_u.sum() > 0 else 0.0
        r["users_contributing_within"] = int((c_u > 0).sum())
    save_json(RESULTS / tag / "full_matrix.json", res, config={"dataset": tag, "min_users": min_users}, seed=None)
    print(tag, json.dumps({k: v for k, v in res.items() if k != "buckets"}, indent=1))


def summary(obs, mod, ind, pd, U, bucket, labels, C, dense, min_users):
    iu = np.triu_indices(len(pd), k=1)
    n = U * obs["dd"][iu]
    res = {"min_users": min_users, "dense_items": int(len(dense)), "pairs": int(len(iu[0])),
           "dense_share_of_interactions": float(C[dense].sum() / C.sum()),
           "observed": {"share_pairs_never_cooccur": float(np.mean(n == 0)),
                        "share_pairs_cov_negative": float(np.mean(obs["dd"][iu] < ind["dd"][iu]))}}
    for name, m in mod.items():
        mm = U * m["dd"][iu]
        ratio = n / mm
        res[name] = {"poisson_deviance_per_pair": float(2 * np.sum(xlogy(n, n / mm) - (n - mm)) / len(n)),
                     "median_ratio_obs_over_model": float(np.median(ratio)),
                     "ratio_quartiles_obs_over_model": [float(q) for q in np.percentile(ratio, [25, 75])],
                     "share_pairs_ratio_within_factor_2": float(np.mean((ratio >= 0.5) & (ratio <= 2))),
                     "spearman_lift_all_pairs": (None if np.ptp(m["dd"][iu] / ind["dd"][iu]) < 1e-12 else   # constant lift (independence)
                                                 float(spearmanr(obs["dd"][iu] / ind["dd"][iu], m["dd"][iu] / ind["dd"][iu]).statistic)),
                     "share_pairs_model_cov_negative": float(np.mean(m["dd"][iu] < ind["dd"][iu] * (1 - 1e-9)))}
    res["buckets"] = []
    for b, lab in enumerate(labels):
        r = {"users": lab, "items": int((bucket == b).sum()), "share_of_interactions": float(C[bucket == b].sum() / C.sum()),
             "observed_lift_within": float(obs["bb"][b, b] / ind["bb"][b, b]),
             "observed_lift_with_dense": float(obs["db"][:, b].sum() / ind["db"][:, b].sum())}
        for name, m in mod.items():
            r[f"obs_over_model_within ({name})"] = float(obs["bb"][b, b] / m["bb"][b, b])
            r[f"obs_over_model_with_dense ({name})"] = float(obs["db"][:, b].sum() / m["db"][:, b].sum())
        res["buckets"].append(r)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ml1m_r4")
    ap.add_argument("--min-users", type=int, default=50)
    a = ap.parse_args()
    run(a.dataset, a.min_users)


if __name__ == "__main__":
    main()
