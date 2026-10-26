"""Figure 1 of the paper: MovieLens-1M (ratings >= 4), four item-item measures under the observed data, the
activity model and the independence model.

Every film with >= --min-users users, pooled in blocks of --block consecutive films (by popularity), and the
tail in the four buckets of phantom.blocks.tail_buckets. Measures: lift, cosine, Pearson correlation (ratios
of sums over the pairs of two blocks) and EASE weights (Steck 2019; symmetrized, mean weight per block pair,
lambda from results/<d>/ease_lambda.json, see 05_ease_lambda.py). The models enter through their expected
co-occurrences (and expected Gram matrices for EASE).

Usage: python experiments/06_main_figure.py --dataset ml1m_r4
Output: results/figures/<d>/paper_matrices_wide.pdf (and .png)
"""

import argparse
import json
import warnings
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.ticker import NullFormatter
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phantom.blocks import augment, tail_buckets  # noqa: E402
from phantom.data import LOADERS  # noqa: E402
from phantom.ease import ease as ease_weights  # noqa: E402
from phantom.paths import RESULTS, dataset_figure  # noqa: E402


OBS, IND, ACT = "observed", "independence", "activity model"
COLS = {OBS: "observed", IND: "independence", ACT: "activity model"}


def run(tag, min_users, wide=True, block=14):
    X, _, _, _ = LOADERS[tag]()
    z = np.load(RESULTS / tag / "model.npz")
    X = X[:, z["fit_items"]].tocsc()
    U, n = X.shape
    C = np.asarray(X.sum(axis=0)).ravel()
    p = C / U
    order = np.argsort(-C, kind="stable")
    dense = order[C[order] >= min_users]
    bucket, labels = tail_buckets(C, min_users)
    K = len(labels)
    masks = [np.where(bucket == b)[0] for b in range(K)]

    # co-occurrence parts (dense x dense, dense x bucket, bucket x bucket; ordered pairs i != j)
    Xd = X[:, dense]
    n_ub = np.stack([np.asarray(X[:, m].sum(axis=1)).ravel() for m in masks], axis=1)
    co = {OBS: {"dd": (Xd.T @ Xd).toarray() / U, "db": np.asarray(Xd.T @ n_ub) / U,
                "bb": (n_ub.T @ n_ub - np.diag(n_ub.sum(axis=0))) / U}}
    P_act = expit(z["h"][None, :] + z["alpha"][:, None])          # P(X_i = 1 | theta_g), activity model
    for name, P, w in [(IND, p[None, :], np.ones(1)), (ACT, P_act, z["w"])]:
        Pd = P[:, dense]
        tau = np.stack([P[:, m].sum(axis=1) for m in masks], axis=1)
        t2 = np.stack([(P[:, m] ** 2).sum(axis=1) for m in masks], axis=1)
        co[name] = {"dd": (Pd * w[:, None]).T @ Pd, "db": (Pd * w[:, None]).T @ tau,
                    "bb": (tau * w[:, None]).T @ tau - np.diag(w @ t2)}
    pd, pb = p[dense], np.array([p[m].sum() for m in masks])
    p2b = np.array([(p[m] ** 2).sum() for m in masks])
    sb = np.array([np.sqrt(p[m]).sum() for m in masks])
    sdv = np.sqrt(p * (1 - p))
    sdd, sdb = sdv[dense], np.array([sdv[m].sum() for m in masks])
    vb = np.array([(p[m] * (1 - p[m])).sum() for m in masks])
    chance = {"dd": np.outer(pd, pd), "db": np.outer(pd, pb), "bb": np.outer(pb, pb) - np.diag(p2b)}
    sq = {"dd": np.sqrt(np.outer(pd, pd)), "db": np.outer(np.sqrt(pd), sb), "bb": np.outer(sb, sb) - np.diag(pb)}
    sdp = {"dd": np.outer(sdd, sdd), "db": np.outer(sdd, sdb), "bb": np.outer(sdb, sdb) - np.diag(vb)}

    # EASE weights from the observed and the expected Gram matrices (lambda from 05_ease_lambda.py)
    lam = json.loads((RESULTS / tag / "ease_lambda.json").read_text())["lambda"]
    gram = {OBS: (X.T @ X).toarray().astype(np.float32)}
    for name, P, w in [(IND, p[None, :], np.ones(1)), (ACT, P_act, z["w"])]:
        G = (U * ((P * w[:, None]).T @ P)).astype(np.float32)
        G[np.diag_indices_from(G)] = U * p
        gram[name] = G
    ease = {}
    for k in [OBS, IND, ACT]:
        B = ease_weights(gram.pop(k), lam)
        Bd = B[np.ix_(dense, dense)].astype(float)
        ease[k] = {"dd": (Bd + Bd.T) / 2,
                   "db": np.stack([(B[np.ix_(dense, m)].mean(axis=1) + B[np.ix_(m, dense)].mean(axis=0)) / 2
                                   for m in masks], axis=1),
                   "bb": np.array([[(B[np.ix_(a, b)].mean() + B[np.ix_(b, a)].mean()) / 2 for b in masks]
                                   for a in masks])}
        del B

    if block > 1:       # pool the dense films into blocks of `block` consecutive films (by popularity)
        co = {k: block_sum(v, block) for k, v in co.items()}
        chance, sq, sdp = block_sum(chance, block), block_sum(sq, block), block_sum(sdp, block)
        ease = {k: block_mean(v, block) for k, v in ease.items()}
    rows = [("lift", {k: {q: co[k][q] / chance[q] for q in chance} for k in co}, "log"),
            ("cosine", {k: {q: co[k][q] / sq[q] for q in sq} for k in co}, "log"),
            ("Pearson correlation", {k: {q: (co[k][q] - chance[q]) / sdp[q] for q in sdp} for k in co}, "div"),
            (f"EASE weight ($\\lambda$ = {lam})", ease, "div")]
    figure(tag, rows, labels, min_users, len(dense) if block <= 1 else -(-len(dense) // block), wide=wide,
           blocked=block > 1)


def block_sum(M, f):
    """Sums over blocks of f consecutive dense films, pairs i != j only: block measures are then ratios of
    these sums (lift_AB = sum p_ij / sum p_i p_j, as for the tail buckets). Strips: sums along the film axis."""
    n = M["dd"].shape[0]
    m = -(-n // f) * f
    dd = np.zeros((m, m))
    dd[:n, :n] = M["dd"]
    dd[np.arange(n), np.arange(n)] = 0.0
    db = np.zeros((m, M["db"].shape[1]))
    db[:n] = M["db"]
    return {"dd": dd.reshape(m // f, f, m // f, f).sum(axis=(1, 3)), "db": db.reshape(m // f, f, -1).sum(axis=1),
            "bb": M["bb"]}


def block_mean(M, f):
    """Mean EASE weight over blocks of f consecutive dense films, pairs i != j (EASE weights are not ratios of
    pair sums, so a block gets the mean weight, as the tail buckets do). Pooling values instead of letting the
    viewer resample the image avoids mixing the colors of alternating red/blue weights into grey."""
    if f <= 1:
        return M
    n = M["dd"].shape[0]
    m = -(-n // f) * f
    dd = np.full((m, m), np.nan)
    dd[:n, :n] = M["dd"]
    dd[np.arange(n), np.arange(n)] = np.nan
    db = np.full((m, M["db"].shape[1]), np.nan)
    db[:n] = M["db"]
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dd = np.nanmean(dd.reshape(m // f, f, m // f, f), axis=(1, 3))
        db = np.nanmean(db.reshape(m // f, f, -1), axis=1)
    return {"dd": dd, "db": db, "bb": M["bb"]}


def figure(tag, rows, labels, min_users, n, wide=False, blocked=False):
    """tall: rows = measures, columns = models (paper_matrices.pdf); wide: rows = models, columns =
    measures, one horizontal colorbar per measure (paper_matrices_wide.pdf)."""
    width = max(4, n // 16)
    models = [OBS, ACT, IND]                   # panel order
    if wide:
        fig, axes = plt.subplots(3, len(rows), figsize=(7.0, 5.6), layout="constrained")
        axes = axes.T                                  # axes[measure, model]
    else:
        fig, axes = plt.subplots(len(rows), 3, figsize=(7.0, 9.0), layout="constrained")
    for r, (mname, mats, scale) in enumerate(rows):
        imgs = {}
        for k in models:
            with np.errstate(divide="ignore", invalid="ignore"):
                A, centers = augment(mats[k], width)
            if not blocked:
                A[:n, :n][np.eye(n, dtype=bool)] = np.nan
            if scale == "log":
                A[A <= 0] = np.nan
            imgs[k] = A
        vals = np.concatenate([a[np.isfinite(a)] for a in imgs.values()])
        if scale == "log":
            nrm, cmap = LogNorm(*np.percentile(vals[vals > 0], [2, 98])), "viridis"
        else:                                          # can be zero or negative: diverging, centered at 0
            v = np.percentile(np.abs(vals), 98)
            nrm, cmap = TwoSlopeNorm(0, -v, v), "RdBu_r"
        for c, k in enumerate(models):
            ax = axes[r, c]
            im = ax.imshow(imgs[k], norm=nrm, cmap=cmap, interpolation="nearest" if blocked else "antialiased")
            first_row = (c == 0) if wide else (r == 0)        # top row of the figure
            last_row = (c == 2) if wide else (r == len(rows) - 1)
            first_col = (r == 0) if wide else (c == 0)
            if first_row:
                ax.set_title(mname if wide else COLS[k], fontsize=9 if not wide else 8)
            ax.set_xticks(centers)
            ax.set_xticklabels(labels if last_row else [], rotation=90, fontsize=5)
            ax.set_yticks(centers if first_col else [])
            ax.set_yticklabels(labels if first_col else [], fontsize=5)
            ax.tick_params(length=0, pad=1)
            if wide and first_col:
                ax.set_ylabel(COLS[k], fontsize=8)
        if wide:
            cb = fig.colorbar(im, ax=axes[r, :], orientation="horizontal", shrink=0.9, aspect=18)
            if scale == "log" and nrm.vmax / nrm.vmin < 30:     # narrow log range: plain ticks, no minor labels
                t = [x for x in [1, 1.5, 2, 3, 5, 7, 10, 15, 20] if nrm.vmin <= x <= nrm.vmax]
                cb.set_ticks(t, labels=[f"{x:g}" for x in t])
                cb.ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            axes[r, 0].set_ylabel(mname, fontsize=8)
            cb = fig.colorbar(im, ax=axes[r, :], shrink=0.85)
        cb.ax.tick_params(labelsize=6)
    out = dataset_figure(tag, "paper_matrices_wide.pdf" if wide else "paper_matrices.pdf")
    fig.savefig(out, dpi=300)
    if wide:
        fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ml1m_r4")
    ap.add_argument("--min-users", type=int, default=50)
    ap.add_argument("--tall", action="store_true", help="rows = measures, columns = models (default: the paper's layout)")
    ap.add_argument("--block", type=int, default=14, help="pool films in blocks of this size (paper: 14)")
    a = ap.parse_args()
    run(a.dataset, a.min_users, wide=not a.tall, block=a.block)


if __name__ == "__main__":
    main()
