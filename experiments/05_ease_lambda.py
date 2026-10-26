"""Regularization strength of EASE, chosen on the observed data (used by 06 and 10).

20% of each user's items are held out (seed fixed) and lambda is the value with the best recall@20 of the
held-out items among a fixed grid. The same lambda is then used for EASE on the observed data and on the
expected Gram matrices of the models.

Usage: python experiments/05_ease_lambda.py --dataset ml1m_r4
Output: results/<d>/ease_lambda.json
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phantom.data import LOADERS  # noqa: E402
from phantom.ease import choose_lambda  # noqa: E402
from phantom.paths import RESULTS  # noqa: E402
from phantom.runinfo import save_json  # noqa: E402

LAMBDAS = [10, 50, 100, 200, 500, 1000, 2000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ml1m_r4")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    X, _, _, _ = LOADERS[a.dataset]()
    z = np.load(RESULTS / a.dataset / "model.npz")
    X = X[:, z["fit_items"]].tocsr()
    lam, recall = choose_lambda(X, LAMBDAS, rng)
    print(f"{a.dataset}: lambda = {lam}", flush=True)
    save_json(RESULTS / a.dataset / "ease_lambda.json",
              {"lambda": lam, "recall_at_20_by_lambda": {str(k): v for k, v in recall.items()}},
              config={"dataset": a.dataset, "lambdas": LAMBDAS}, seed=a.seed)


if __name__ == "__main__":
    main()
