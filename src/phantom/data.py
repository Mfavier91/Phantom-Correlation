"""Dataset loaders. Raw data live in data/ (not included; see README.md for the download links).

Every loader returns (X, item_ids, titles, info): X is a CSR user x item 0/1 matrix
(repeated interactions counted once), titles is an array of item names (or ids), and
info describes the source and preprocessing.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

DATA = Path(__file__).resolve().parents[2] / "data"


def _binary_matrix(users, items):
    """CSR 0/1 matrix from parallel arrays of user and item identifiers."""
    u_idx, u_ids = pd.factorize(users, sort=True)
    i_idx, i_ids = pd.factorize(items, sort=True)
    i_ids = np.asarray(i_ids)
    X = sp.csr_matrix((np.ones(len(u_idx)), (u_idx, i_idx)), shape=(len(u_ids), len(i_ids)))
    X.data[:] = 1.0  # duplicates were summed by the constructor
    return X, i_ids


def load_movielens_1m(min_rating=4):
    """MovieLens-1M, https://files.grouplens.org/datasets/movielens/ml-1m.zip.
    A rating counts as a consumption if it is at least min_rating."""
    r = pd.read_csv(DATA / "ml-1m" / "ratings.dat", sep="::", engine="python",
                    names=["user", "item", "rating", "ts"])
    r = r[r.rating >= min_rating]
    X, items = _binary_matrix(r.user.values, r.item.values)
    movies = pd.read_csv(DATA / "ml-1m" / "movies.dat", sep="::", engine="python",
                         names=["item", "title", "genres"], encoding="latin-1").set_index("item")
    titles = movies.title.reindex(items).fillna("?").values
    info = {"source": "MovieLens-1M (GroupLens)", "interaction": f"rating >= {min_rating}"}
    return X, items, titles, info


def load_movielens_20m(min_rating=4):
    """MovieLens-20M, https://files.grouplens.org/datasets/movielens/ml-20m.zip.
    A rating counts as a consumption if it is at least min_rating."""
    r = pd.read_csv(DATA / "ml-20m" / "ratings.csv", usecols=["userId", "movieId", "rating"])
    r = r[r.rating >= min_rating]
    X, items = _binary_matrix(r.userId.values, r.movieId.values)
    movies = pd.read_csv(DATA / "ml-20m" / "movies.csv").set_index("movieId")
    titles = movies.title.reindex(items).fillna("?").values
    info = {"source": "MovieLens-20M (GroupLens)", "interaction": f"rating >= {min_rating}"}
    return X, items, titles, info


def load_lastfm_2k():
    """Last.fm HetRec 2011, https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip.
    user_artists.dat keeps each user's top-50 most listened artists, so degrees are <= 50."""
    r = pd.read_csv(DATA / "lastfm-2k" / "user_artists.dat", sep="\t")
    X, items = _binary_matrix(r.userID.values, r.artistID.values)
    artists = pd.read_csv(DATA / "lastfm-2k" / "artists.dat", sep="\t", usecols=["id", "name"],
                          encoding="utf-8", on_bad_lines="skip").set_index("id")
    titles = artists.name.reindex(items).fillna("?").values
    info = {"source": "Last.fm HetRec 2011", "interaction": "artist among the user's top-50 listened"}
    return X, items, titles, info


def load_amazon_books():
    """Amazon Reviews 2023, Books, 5-core rating-only benchmark split,
    https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Books.csv.gz"""
    r = pd.read_csv(DATA / "amazon2023" / "Books.csv.gz", usecols=["user_id", "parent_asin"])
    X, items = _binary_matrix(r.user_id.values, r.parent_asin.values)
    info = {"source": "Amazon Reviews 2023, Books", "interaction": "review (any rating), 5-core"}
    return X, items, items.astype(str), info


LOADERS = {
    "ml1m_r4": load_movielens_1m,
    "ml20m_r4": load_movielens_20m,
    "lastfm2k": load_lastfm_2k,
    "amazon_books": load_amazon_books,
}

DISPLAY_NAMES = {
    "ml1m_r4": "MovieLens-1M (≥4)", "ml20m_r4": "MovieLens-20M (≥4)",
    "lastfm2k": "Last.fm", "amazon_books": "Amazon Books",
}
