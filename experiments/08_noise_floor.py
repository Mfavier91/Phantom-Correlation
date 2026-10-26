"""Noise floor of the deviance and the share of the systematic mismatch removed by activity (paper,
Equation (share); Table 2, column "Pairs").

R datasets of the same size are simulated from the fitted activity model (phantom/simulate_fast.py, exact),
and the deviance of the same fitted model is computed on each. Their mean N_hat estimates the deviance that
counting noise alone produces. With the real-data deviances (results/<d>/pair_deviance.json):
    share = (D_ind - D_act) / (D_ind - N_hat).

Usage: python experiments/08_noise_floor.py --dataset ml1m_r4 [--reps 10]
Output: results/<d>/noise_floor.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phantom.deviance import deviances  # noqa: E402
from phantom.paths import RESULTS  # noqa: E402
from phantom.runinfo import save_json  # noqa: E402
from phantom.simulate_fast import simulate_fast  # noqa: E402


def run(tag, reps, seed0):
    z = np.load(RESULTS / tag / "model.npz")
    h, alpha, w = z["h"], z["alpha"], z["w"]
    U = int(json.loads((RESULTS / tag / "summary.json").read_text())["data"]["users"])
    real = json.loads((RESULTS / tag / "pair_deviance.json").read_text())["deviance"]
    recs = []
    for r in range(reps):
        t0 = time.time()
        rng = np.random.default_rng(seed0 + r)
        X, _ = simulate_fast(h, alpha, w, U, rng)
        C = np.asarray(X.sum(axis=0)).ravel()
        keep = (C > 0) & (C < U)                     # items never (or always) consumed in this replicate
        dev = deviances(X[:, keep], h[keep], alpha, w)
        recs.append({"replicate": r, "seed": seed0 + r, "interactions": int(X.nnz),
                     "deviance_activity": dev["activity"], "deviance_independence": dev["independence"],
                     "seconds": time.time() - t0})
        print(tag, recs[-1], flush=True)
    n_hat = float(np.mean([x["deviance_activity"] for x in recs]))
    res = {"real_deviance": real, "noise_floor": n_hat,
           "noise_floor_range": [min(x["deviance_activity"] for x in recs), max(x["deviance_activity"] for x in recs)],
           "share_structure_explained": (real["independence"] - real["activity"]) / (real["independence"] - n_hat),
           "excess_ratio": real["activity"] / n_hat, "replicates": recs}
    save_json(RESULTS / tag / "noise_floor.json", res, config={"dataset": tag, "reps": reps}, seed=seed0)
    print({k: v for k, v in res.items() if k != "replicates"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ml1m_r4")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1000)
    a = ap.parse_args()
    run(a.dataset, a.reps, a.seed)


if __name__ == "__main__":
    main()
