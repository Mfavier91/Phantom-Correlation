"""Whole-catalog comparison in popularity groups for the datasets without a full matrix (paper, Appendix B).

All items, grouped so that every group holds about the same share of interactions
(blocks.popularity_groups_equal_mass); block lift, cosine and Pearson correlation are ratios of sums
(exact). Grey in the figures: fewer than 5 expected co-occurrences under the activity model.
Also: the mean popularity of a user's items as a function of the number of items consumed (observed,
activity model, independence), which needs the inclusion probabilities given the basket size.

Usage: python experiments/04_catalog_groups.py --datasets ml20m_r4 lastfm2k amazon_books [--groups 40]
Outputs: results/<d>/catalog.json; results/figures/<d>/catalog_block_matrices.pdf and
catalog_profile_and_curves.pdf (copies in figures/)
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.ticker import NullFormatter
from scipy.special import expit, logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phantom.blocks import block_sums_from_distinct, popularity_groups_equal_mass  # noqa: E402
from phantom.conditional import _inclusion  # noqa: E402
from phantom.data import DISPLAY_NAMES, LOADERS  # noqa: E402
from phantom.model import softplus  # noqa: E402
from phantom.paths import RESULTS, dataset_figure  # noqa: E402
from phantom.runinfo import save_json  # noqa: E402

IND, ACT = "independence", "activity model"
NAMES = ["observed", ACT, IND]
STYLE = {"observed": "o-", IND: "k--", ACT: "D-."}


def profile(hu, mult, fu, alpha, w, kmax):
    """E[mean f of the basket | S = k] under logit P(X_i = 1 | theta) = h_i + theta, per distinct count
    (hu, fu, mult). Given S = k, the basket does not depend on theta (conditional Bernoulli distribution),
    so the inclusion probabilities are computed once and averaged over the posterior of theta."""
    k = np.arange(kmax + 1)
    A = softplus(hu[None, :] + alpha[:, None]) @ mult
    pi, _, le = _inclusion(hu, mult, kmax)
    with np.errstate(invalid="ignore", divide="ignore"):
        cond = (pi * mult) @ fu / k
        logpost = np.where(w[:, None] > 0, np.log(w)[:, None] + le[None, : kmax + 1] + alpha[:, None] * k[None, :]
                           - A[:, None], -np.inf)
    post = np.exp(logpost - logsumexp(logpost, axis=0))
    return post.sum(axis=0) * cond


def run(tag, n_groups):
    out = RESULTS / tag
    X, _, _, _ = LOADERS[tag]()
    fz = np.load(out / "model.npz")
    X = X[:, fz["fit_items"]].tocsr()
    U = X.shape[0]
    C = np.asarray(X.sum(axis=0)).ravel()
    S = np.asarray(X.sum(axis=1)).ravel().astype(int)
    logC = np.log(C)
    f = logC - (C @ logC) / C.sum()                       # centered log popularity (only for the profile)
    Cu, idx, inv, mult = np.unique(C, return_index=True, return_inverse=True, return_counts=True)
    p_u = Cu / U
    # (log-odds per distinct count, activity grid, weights); independence: every user identical, P(X_i = 1) = p_i
    models = {IND: (np.log(p_u / (1 - p_u)), np.zeros(1), np.ones(1)),
              ACT: (fz["h"][idx], fz["alpha"], fz["w"])}
    res = {}

    # popularity of a user's items vs the number of items consumed
    shift = (C @ logC) / C.sum()
    to_log10p = lambda mf: (mf + shift) / np.log(10) - np.log10(U)  # noqa: E731
    user_mean = np.asarray(X @ f).ravel() / S
    curves = {k: profile(h, mult, f[idx], a, w, int(S.max())) for k, (h, a, w) in models.items()}
    edges = np.unique(np.round(np.logspace(np.log10(max(S.min(), 1)), np.log10(S.max() + 1), 13)).astype(int))
    prof = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (S >= lo) & (S < hi)
        if s.any():
            r = {"degree_from": int(lo), "degree_to": int(hi - 1), "users": int(s.sum()),
                 "observed": float(to_log10p(user_mean[s].mean()))}
            r.update({k: float(to_log10p(np.mean(v[S[s]]))) for k, v in curves.items()})
            prof.append(r)
    res["popularity_profile_log10"] = prof
    res["profile_mean_abs_error_log10"] = {k: float(np.average([abs(r[k] - r["observed"]) for r in prof],
                                                               weights=[r["users"] for r in prof])) for k in models}

    # blocks over equal-interaction groups
    grp = popularity_groups_equal_mass(C, n_groups)
    G = grp.max() + 1
    vg = np.zeros(len(Cu), int)
    vg[inv] = grp
    onehot = sp.csr_matrix((np.ones(len(grp)), (np.arange(len(grp)), grp)), shape=(len(grp), G))
    n_ug = (X @ onehot).toarray()
    sums = {"observed": (n_ug.T @ n_ug - np.diag(n_ug.sum(axis=0))) / U}
    for name, (h, a, w) in models.items():
        P = expit(h[None, :] + a[:, None])                # P(X_i = 1 | grid point), per distinct count
        F = (P * w[:, None]).T @ P
        sums[name] = block_sums_from_distinct(F, np.diag(F), vg, mult, G)
    ind = block_sums_from_distinct(np.outer(p_u, p_u), p_u**2, vg, mult, G)
    sqs = block_sums_from_distinct(np.outer(np.sqrt(p_u), np.sqrt(p_u)), p_u, vg, mult, G)
    sdu = np.sqrt(p_u * (1 - p_u))
    sdp = block_sums_from_distinct(np.outer(sdu, sdu), sdu**2, vg, mult, G)
    lift = {k: v / ind for k, v in sums.items()}
    cosine = {k: v / sqs for k, v in sums.items()}
    pearson = {k: (v - ind) / sdp for k, v in sums.items()}     # ratio of sums: sum cov / sum sd_i sd_j
    reliable = U * sums[ACT] >= 5
    gpop = np.bincount(grp, weights=C / U) / np.bincount(grp)
    share = np.bincount(grp, weights=C) / C.sum()
    res["groups"] = {"n": int(G), "items_min_max": [int(np.bincount(grp).min()), int(np.bincount(grp).max())],
                     "interaction_share_min_max": [float(share.min()), float(share.max())],
                     "least_popular_group_max_users": int(C[grp == G - 1].max())}
    save_json(out / "catalog.json", res, config={"dataset": tag, "groups": n_groups, "binning": "equal interactions"},
              seed=None)
    figures(tag, prof, lift, cosine, pearson, reliable, gpop)
    print(tag, {k: res[k] for k in ["profile_mean_abs_error_log10", "groups"]}, flush=True)


def figures(tag, prof, lift, cosine, pearson, reliable, gpop):
    name = DISPLAY_NAMES[tag]
    measures = [("lift", lift), ("cosine", cosine), ("Pearson correlation", pearson)]
    # popularity profile and within-group curves
    fig, axes = plt.subplots(1, 4, figsize=(22, 4.6), layout="constrained")
    mid = [np.sqrt(r["degree_from"] * (r["degree_to"] + 1)) for r in prof]
    for k in NAMES:
        axes[0].plot(mid, [r[k] for r in prof], STYLE[k], label=k)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("user degree (items consumed)")
    axes[0].set_ylabel("mean log10 popularity of the user's items")
    axes[0].set_title("Popularity of a user's items vs degree")
    axes[0].legend(frameon=False, fontsize=8)
    d = np.diag_indices(len(gpop))
    ok = reliable[d]
    for ax, (mname, mats) in zip(axes[1:], measures):
        for k in NAMES:
            ax.plot(gpop[ok], mats[k][d][ok], STYLE[k], ms=3, label=k)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-3) if mname == "Pearson correlation" else ax.set_yscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("group mean popularity (share of users)")
        ax.set_ylabel(f"within-group block {mname}")
        ax.set_title(f"{mname} within popularity groups")
    fig.suptitle(f"{name}: observed, independence and activity model, all items; "
                 f"groups hold equal shares of interactions")
    out = dataset_figure(tag, "catalog_profile_and_curves.pdf")
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)        # for display in notebooks/figures.ipynb
    plt.close(fig)

    def strip(ax):
        s = ax.inset_axes([0, -0.08, 1, 0.05])
        im = s.imshow(gpop[None, :], norm=LogNorm(gpop.min(), gpop.max()), cmap="Oranges", aspect="auto",
                      interpolation="nearest")
        s.set_xticks([]); s.set_yticks([])
        s.set_xlabel("groups, most → least popular", fontsize=8)
        return im

    # block matrices: rows = measures, columns = observed, activity model, independence
    fig, axes = plt.subplots(3, len(NAMES), figsize=(15, 15), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.3)
    for r, (mname, mats) in enumerate(measures):
        vals = np.concatenate([np.where(reliable, mats[k], np.nan).ravel() for k in NAMES])
        vals = vals[np.isfinite(vals)]
        if mname == "Pearson correlation":           # can be zero or negative: diverging, centered at 0
            v = np.percentile(np.abs(vals), 98)
            nrm, cmap = TwoSlopeNorm(0, -v, v), "RdBu_r"
        else:
            nrm, cmap = LogNorm(*np.percentile(vals[vals > 0], [2, 98])), "viridis"
        for c, k in enumerate(NAMES):
            ax = axes[r, c]
            im = ax.imshow(np.where(reliable, mats[k], np.nan), norm=nrm, cmap=cmap, interpolation="nearest")
            if r == 0:
                ax.set_title(k, fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
            strip(ax)
        axes[r, 0].set_ylabel(f"{mname}\npopularity groups, least ← most popular", fontsize=10)
        fig.colorbar(im, ax=axes[r, :], shrink=0.8, label=f"block {mname}")
    fig.suptitle(f"{name}, whole catalog in groups with equal shares of interactions "
                 f"(grey = fewer than 5 expected co-occurrences)", fontsize=13)
    out = dataset_figure(tag, "catalog_block_matrices.pdf")
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".png"), dpi=150)        # for display in notebooks/figures.ipynb
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ml20m_r4", "lastfm2k", "amazon_books"])
    ap.add_argument("--groups", type=int, default=40)
    args = ap.parse_args()
    for tag in args.datasets:
        run(tag, args.groups)


if __name__ == "__main__":
    main()
