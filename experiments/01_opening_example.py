"""Opening example of the paper (Example 1): identical films, casual viewers and binge watchers.

Half of the users watch every film with probability 0.01, the other half with probability 0.20, all
independently. Every pair of films is then positively associated; the lift is E[q^2] / E[q]^2, where q is
a user's watching probability.

Usage: python experiments/01_opening_example.py
Output: results/synthetic/opening_example.json
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phantom import observed_cooccurrence, pair_measures  # noqa: E402
from phantom.paths import RESULTS  # noqa: E402
from phantom.runinfo import save_json  # noqa: E402

SEED = 20261009


def main():
    rng = np.random.default_rng(SEED)
    n, U, frac_light, pL, pH = 100, 100_000, 0.5, 0.01, 0.2
    light = rng.random(U) < frac_light
    p_user = np.where(light, pL, pH)
    X = rng.random((U, n)) < p_user[:, None]
    co, p = observed_cooccurrence(X.astype(float))
    obs = pair_measures(co, p)
    m = np.mean(p_user)
    res = {"config": dict(n=n, U=U, frac_light=frac_light, p_light=pL, p_heavy=pH),
           "analytic": {"lift": np.mean(p_user**2) / m**2, "corr": np.var(p_user) / (m * (1 - m))},
           "simulated_mean": {"lift": obs["lift"].mean(), "corr": obs["corr"].mean()},
           "frac_pairs_positive_corr": float(np.mean(obs["corr"] > 0))}
    path = save_json(RESULTS / "synthetic" / "opening_example.json", res, config=res["config"], seed=SEED)
    print(path.read_text())


if __name__ == "__main__":
    main()
