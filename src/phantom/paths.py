"""Where results and figures go."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


def dataset_figure(dataset, name):
    """results/figures/<dataset>/<name>"""
    d = FIGURES / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d / name
