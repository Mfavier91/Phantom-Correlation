#!/usr/bin/env bash
# Reproduce all results of the paper. Data must be in data/ (see README.md).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
DATASETS="ml1m_r4 ml20m_r4 lastfm2k amazon_books"

$PY experiments/01_opening_example.py
for d in $DATASETS; do $PY experiments/02_fit_activity_model.py --dataset "$d"; done
$PY experiments/03_movielens_pairs.py --dataset ml1m_r4
$PY experiments/04_catalog_groups.py --datasets ml20m_r4 lastfm2k amazon_books
for d in ml1m_r4 ml20m_r4 lastfm2k; do $PY experiments/05_ease_lambda.py --dataset "$d"; done   # EASE: not on Amazon Books
$PY experiments/06_main_figure.py --dataset ml1m_r4
$PY experiments/07_pair_deviance.py --datasets $DATASETS
for d in $DATASETS; do $PY experiments/08_noise_floor.py --dataset "$d"; done
$PY experiments/09_measure_agreement.py --datasets $DATASETS
for d in ml1m_r4 ml20m_r4 lastfm2k; do $PY experiments/10_ease_simulated.py --dataset "$d"; done
$PY experiments/11_paper_tables.py
