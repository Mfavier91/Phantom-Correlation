"""Pooling of items into blocks (paper, Section 7.2: "Circumventing computational limitations").

MovieLens-1M: every film with >= min_users users is kept individually (or pooled in blocks of
consecutive films by popularity), and the rarer films are pooled into four buckets equally spaced
in log count. Larger catalogs: popularity groups holding equal shares of all interactions.
Block measures are ratios of sums over the pairs i in A, j in B, i != j.
"""

import numpy as np


def tail_buckets(C, min_users, k=4):
    """Bucket index (0 = most popular bucket) for items with C < min_users, -1 otherwise.
    For min_users = 50 the buckets hold items with 1-2, 3-7, 8-18 and 19-49 users."""
    lower = np.unique(np.ceil(np.logspace(0, np.log10(min_users), k + 1)).astype(int))   # 1, 3, 8, 19, 50
    b = np.full(len(C), -1)
    for j in range(len(lower) - 1):
        b[(C >= lower[j]) & (C < lower[j + 1])] = len(lower) - 2 - j
    labels = [f"{lower[j]}–{lower[j + 1] - 1}" if lower[j + 1] - 1 > lower[j] else f"{lower[j]}"
              for j in range(len(lower) - 2, -1, -1)]
    return b, labels


def popularity_groups_equal_mass(C, n_groups):
    """Group index per item: groups holding (about) equal shares of all interactions, most popular
    group first. Items with equal counts always share a group (needed by block_sums_from_distinct),
    so a group can exceed its share when one count value is heavy."""
    C = np.asarray(C)
    Cu, inv = np.unique(C, return_inverse=True)
    mass = np.bincount(inv, weights=C)[::-1]               # per distinct count, most popular first
    before = np.cumsum(mass) - mass                        # interaction mass of more popular items
    g_desc = np.minimum((before / mass.sum() * n_groups).astype(int), n_groups - 1)
    used, g_desc = np.unique(g_desc, return_inverse=True)  # drop empty groups, keep order
    return g_desc[::-1][inv]


def groups_for(tag, C, min_users=50, block=14, n_groups=40):
    """Blocks used for the measure-by-measure comparison (Table 2 of the paper): on MovieLens-1M the films
    with >= min_users users in blocks of `block` consecutive films plus the four tail buckets (as in
    Figure 1); on the other datasets the popularity groups with equal shares of interactions."""
    if tag == "ml1m_r4":
        order = np.argsort(-C, kind="stable")
        dense = order[C[order] >= min_users]
        b = np.full(len(C), -1)
        b[dense] = np.arange(len(dense)) // block
        nd = b.max() + 1
        tail, _ = tail_buckets(C, min_users)
        used, idx = np.unique(tail[tail >= 0], return_inverse=True)
        b[tail >= 0] = nd + idx
        return b
    return popularity_groups_equal_mass(C, n_groups)


def block_sums_from_distinct(F, self_term, value_group, mult, G):
    """sum_{i in A, j in B, i != j} f(i, j) when f depends only on the items' distinct values.

    F[u, v]: value for two different items with values u, v; self_term[u]: f for a pair of
    different items with the same value u (= F[u, u]); value_group[u]: group of value u."""
    W = np.zeros((len(mult), G))
    W[np.arange(len(mult)), value_group] = mult       # items of each value in each group
    total = W.T @ F @ W
    total[np.diag_indices(G)] -= np.bincount(value_group, weights=mult * self_term, minlength=G)
    return total


def augment(M, width):
    """One image from the parts of a matrix: dense block ("dd"), then a white gap, then each tail bucket
    ("db", "bb") repeated `width` pixels. Returns the image and the pixel centers of the buckets."""
    n, K = M["db"].shape
    gap = max(2, width // 4)
    N = n + gap + K * width
    A = np.full((N, N), np.nan)
    A[:n, :n] = M["dd"]
    rep = np.repeat(M["db"], width, axis=1)
    A[:n, n + gap:] = rep
    A[n + gap:, :n] = rep.T
    A[n + gap:, n + gap:] = np.repeat(np.repeat(M["bb"], width, axis=0), width, axis=1)
    centers = n + gap + width * (np.arange(K) + 0.5)
    return A, centers
