"""Poisson deviance of the co-occurrence counts over all item pairs, activity model and independence
(paper, Equation (deviance); no threshold, no pooling).

Usage: python experiments/07_pair_deviance.py --datasets ml1m_r4 ml20m_r4 lastfm2k amazon_books
Output: results/<d>/pair_deviance.json
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phantom.data import LOADERS  # noqa: E402
from phantom.deviance import deviances  # noqa: E402
from phantom.paths import RESULTS  # noqa: E402
from phantom.runinfo import save_json  # noqa: E402


def run(tag, chunk_items):
    t0 = time.time()
    X, _, _, _ = LOADERS[tag]()
    z = np.load(RESULTS / tag / "model.npz")
    X = X[:, z["fit_items"]].tocsc().astype(np.float64)
    dev = deviances(X, z["h"], z["alpha"], z["w"], chunk=chunk_items)
    res = {"items": int(X.shape[1]), "users": int(X.shape[0]), "deviance": dev,
           "share_deviance_removed_by_activity": 1 - dev["activity"] / dev["independence"]}
    save_json(RESULTS / tag / "pair_deviance.json", res, config={"dataset": tag, "chunk_items": chunk_items}, seed=None)
    print(f"{tag}: {res} ({time.time() - t0:.0f} s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ml1m_r4"])
    ap.add_argument("--chunk-items", type=int, default=1000)
    a = ap.parse_args()
    for d in a.datasets:
        run(d, a.chunk_items)


if __name__ == "__main__":
    main()
