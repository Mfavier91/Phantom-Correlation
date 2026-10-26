"""Fit the activity model to a dataset by expectation-maximization (paper, Sections 6 and 7.1).

The model is fitted on all items (every item with 0 < C_i < U) from its sufficient statistics: the item
counts C_i and the distribution of history lengths. Saves the fitted model and the dataset statistics of
Table 1 (Var(S), sum_i p_i (1 - p_i)).

Usage: python experiments/02_fit_activity_model.py --dataset ml1m_r4
Outputs: results/<dataset>/model.npz (h, alpha, w, fit_items), results/<dataset>/summary.json
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phantom import fit_activity_model  # noqa: E402
from phantom.data import LOADERS  # noqa: E402
from phantom.paths import RESULTS  # noqa: E402
from phantom.runinfo import save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(LOADERS))
    args = ap.parse_args()
    tag = args.dataset
    out = RESULTS / tag
    t0 = time.time()

    X, item_ids, titles, info = LOADERS[tag]()
    U, n = X.shape
    C = np.asarray(X.sum(axis=0)).ravel()
    S = np.asarray(X.sum(axis=1)).ravel().astype(int)
    keep_fit = (C > 0) & (C < U)
    deg = np.bincount(S, minlength=n + 1)
    print(f"{tag}: {U} users, {n} items, {X.nnz} interactions, density {X.nnz / U / n:.5f}")

    model = fit_activity_model(C[keep_fit], deg, verbose=True)
    deg_emp = deg / U
    deg_mod = model.degree_pmf(kmax=S.max())  # model mass beyond the largest observed degree is reported below
    m_ = min(len(deg_emp), len(deg_mod))
    ks = np.max(np.abs(np.cumsum(deg_emp[:m_]) - np.cumsum(deg_mod[:m_])))
    p_all = C[keep_fit] / U
    print(f"fit: {model.n_iter} iterations, degree KS {ks:.4f} ({time.time() - t0:.0f}s)")

    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "model.npz", h=model.h, alpha=model.alpha, w=model.w, fit_items=np.where(keep_fit)[0])
    payload = {
        "data": {**info, "users": U, "items": n, "interactions": int(X.nnz),
                 "density": X.nnz / U / n, "degree_min": int(S.min()),
                 "degree_median": float(np.median(S)), "degree_max": int(S.max()),
                 "items_fitted": int(keep_fit.sum())},
        "fit": {"n_iter": model.n_iter, "log_likelihood": float(model.log_likelihood),
                "degree_ks_distance": float(ks),
                "model_mass_below_min_observed_degree": float(deg_mod[:int(S.min())].sum()),
                "model_mass_above_max_observed_degree": float(1 - deg_mod.sum()),
                "degree_variance": float(S.var()),                                   # Var(S), Table 1
                "independence_variance_sum_p1mp": float((p_all * (1 - p_all)).sum()),  # sum_i p_i (1 - p_i)
                **model.activity_moments()},
        "runtime_seconds": time.time() - t0,
    }
    path = save_json(out / "summary.json", payload, config=vars(args), seed=None)
    print(f"done in {time.time() - t0:.0f}s -> {path}")


if __name__ == "__main__":
    main()
