"""EASE on the observed data, on the expected Gram matrices of the models and on simulated data (paper,
Section 7.2 and Table 2, column "EASE").

--reps datasets are simulated from the fitted activity model (fast exact simulation, phantom/simulate_fast.py);
EASE with the lambda of results/<d>/ease_lambda.json is computed on each. Single weights of a simulated dataset
are compared with the weights of the expected counts (rank correlation: sampling noise of single weights).
Block means of the symmetrized weights use the blocks of 09_measure_agreement.py (MovieLens-1M: blocks of --block films + four tail buckets; other datasets:
popularity groups with equal shares of interactions). The block-level share of the error of independence
removed by activity uses the per-block noise of the simulated activity datasets, as in 09.

Usage: python experiments/10_ease_simulated.py --dataset ml1m_r4 [--reps 10]
Output: results/<d>/ease_simulated.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.special import expit
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phantom.blocks import groups_for, tail_buckets  # noqa: E402
from phantom.data import LOADERS  # noqa: E402
from phantom.ease import ease  # noqa: E402
from phantom.paths import RESULTS  # noqa: E402
from phantom.runinfo import save_json  # noqa: E402
from phantom.simulate_fast import simulate_fast  # noqa: E402


OBS, ACT, IND = "observed", "activity model", "independence"


def run(tag, reps, min_users, block, seed0):
    X0, _, _, _ = LOADERS[tag]()
    z = np.load(RESULTS / tag / "model.npz")
    X0 = X0[:, z["fit_items"]].tocsc()
    U, n = X0.shape
    h, alpha, w = z["h"], z["alpha"], z["w"]
    lam = json.loads((RESULTS / tag / "ease_lambda.json").read_text())["lambda"]
    C = np.asarray(X0.sum(axis=0)).ravel()
    p = C / U
    F = expit(h[None, :] + alpha[:, None])

    order = np.argsort(-C, kind="stable")
    dense = order[C[order] >= min_users]          # items with >= min_users users: pair-level noise check
    iu = np.triu_indices(len(dense), 1)
    if tag == "ml1m_r4":
        b = np.full(n, -1)
        b[dense] = np.arange(len(dense)) // block
        nd = b.max() + 1
        tail, _ = tail_buckets(C, min_users)
        b[tail >= 0] = nd + tail[tail >= 0]
    else:                                         # popularity groups, as 09_measure_agreement.py / Table 2
        b = groups_for(tag, C)
    K = b.max() + 1
    Ind = sp.csr_matrix((np.ones(n), (np.arange(n), b)), shape=(n, K))
    size = np.bincount(b, minlength=K).astype(float)
    npairs = np.outer(size, size) - np.diag(size)

    def ease_of(G):
        # float32 throughout (20k items: one n x n matrix is 1.7 GB); symmetrized weights (B + B^T)/2 via the
        # dense submatrix and the block sums, never as a second n x n matrix; EASE has a zero diagonal
        B = ease(G, lam)
        Bd = B[np.ix_(dense, dense)].astype(np.float64)
        S = np.asarray((Ind.T @ (Ind.T @ B).T).T, dtype=np.float64)
        del B
        return ((Bd + Bd.T) / 2)[iu], (S + S.T) / 2 / npairs

    def gram(X):
        X32 = X.astype(np.float32).tocsc()
        return (X32.T @ X32).toarray()

    _, obs_blocks = ease_of(gram(X0))
    G = (U * ((F * w[:, None]).T @ F)).astype(np.float32)       # expected Gram matrix of the activity model
    G[np.diag_indices_from(G)] = U * p
    act_pairs, act_blocks = ease_of(G)
    G = (U * np.outer(p, p)).astype(np.float32)                 # expected Gram matrix of the independence model
    G[np.diag_indices_from(G)] = U * p
    _, ind_blocks = ease_of(G)
    del G

    reps_blocks, rank_corr = [], []
    for r in range(reps):
        t0 = time.time()
        X, _ = simulate_fast(h, alpha, w, U, np.random.default_rng(seed0 + r))
        v, bl = ease_of(gram(X))
        reps_blocks.append(bl)
        # single weights of one simulated dataset vs the weights of the expected counts (sampling noise)
        rank_corr.append(float(spearmanr(v, act_pairs).statistic))
        print(f"simulated dataset {r}: rank correlation with the expected-count weights {rank_corr[-1]:.3f} "
              f"({time.time() - t0:.0f} s)", flush=True)
    # standardized error at block level (as 09_measure_agreement.py): per-block noise from the simulated activity
    # datasets around the expected-Gram activity weights; block pairs with >= 5 expected co-occurrences
    Indd = Ind.toarray()
    tau, t2 = F @ Indd, (F ** 2) @ Indd
    co_act = U * ((tau * w[:, None]).T @ tau - np.diag(w @ t2))
    keep = np.triu(co_act >= 5)
    VS = np.array([bl[keep] for bl in reps_blocks])
    m_act, m_ind, y = act_blocks[keep], ind_blocks[keep], obs_blocks[keep]
    sig2 = np.mean((VS - m_act) ** 2, axis=0)
    z_act, z_ind = float(np.mean((y - m_act) ** 2 / sig2)), float(np.mean((y - m_ind) ** 2 / sig2))
    z_sim = [float(np.mean((VS[r] - m_act) ** 2 / np.mean((np.delete(VS, r, axis=0) - m_act) ** 2, axis=0)))
             for r in range(len(VS))]
    res = {"lambda": lam, "reps": reps,
           "simulated_rank_correlation_with_expected_weights": {"mean": float(np.mean(rank_corr)),
                                                                "per_replicate": rank_corr},
           "standardized_error": {"block_pairs_used": int(keep.sum()), "Z_activity": z_act, "Z_independence": z_ind,
                                  "Z_simulated_mean": float(np.mean(z_sim)), "Z_simulated_range": [min(z_sim), max(z_sim)],
                                  "share": (z_ind - z_act) / (z_ind - float(np.mean(z_sim)))}}
    print("standardized error:", res["standardized_error"])
    save_json(RESULTS / tag / "ease_simulated.json", res, config={"dataset": tag, "reps": reps, "min_users": min_users,
                                                                   "block": block}, seed=seed0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ml1m_r4")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--min-users", type=int, default=50)
    ap.add_argument("--block", type=int, default=14)
    ap.add_argument("--seed", type=int, default=3000)
    a = ap.parse_args()
    run(a.dataset, a.reps, a.min_users, a.block, a.seed)


if __name__ == "__main__":
    main()
