# Do Similarity Measures Measure Similarity? How User Activity Creates Phantom Correlation in Recommender Systems

Code to reproduce every table, figure and number of the paper.

`figures/` holds the figures for MovieLens-20M, Last.fm and Amazon Books that are referred to, but not shown, in the paper's appendix (block lift, cosine and Pearson correlation for the observed data, the activity model and the independence model; for Amazon Books also the popularity of a user's items vs. history length). They are produced by `experiments/04_catalog_groups.py`; `notebooks/figures.ipynb` regenerates them, together with Figure 1, and shows them inline.

The paper shows that item-item association can come purely from users differing in how many items they consume,
even when every user consumes items independently. Three assumptions on consumption lead to a unique model, the
**activity model**: logit P(X_ui = 1 | θ_u) = h_i + θ_u, with a nonparametric distribution of the activity θ
across users (a Rasch model). The scripts fit this model to four datasets and compare lift, cosine, Pearson
correlation and EASE weights under the observed data, the activity model and the independence model.

## Setup

Python 3.13 or later.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q          # brute-force checks of the recursions, the EM fit and the deviance
```

## Data

The datasets are not included. Download them and place them under `data/` as follows:

| Dataset | Source | Files used |
|---|---|---|
| MovieLens-1M | https://files.grouplens.org/datasets/movielens/ml-1m.zip | `data/ml-1m/ratings.dat`, `data/ml-1m/movies.dat` |
| MovieLens-20M | https://files.grouplens.org/datasets/movielens/ml-20m.zip | `data/ml-20m/ratings.csv`, `data/ml-20m/movies.csv` |
| Last.fm (HetRec 2011) | https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip | `data/lastfm-2k/user_artists.dat`, `data/lastfm-2k/artists.dat` |
| Amazon Books (Amazon Reviews 2023, 5-core, rating only) | https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Books.csv.gz | `data/amazon2023/Books.csv.gz` |

Preprocessing (in `src/phantom/data.py`): every interaction is binary and repeated interactions count once; on
MovieLens a rating counts as a consumption only if it is at least 4. Dataset keys: `ml1m_r4`, `ml20m_r4`,
`lastfm2k`, `amazon_books`.

## Running

`run_all.sh` runs the whole pipeline in order. The scripts can also be run one by one (each docstring gives its
usage); every result is written to `results/` together with its configuration and seed.

| Script | What it computes | Paper |
|---|---|---|
| `01_opening_example.py` | casual viewers and binge watchers | Example 1 |
| `02_fit_activity_model.py` | EM fit of the activity model on all items; Var(S), Σ p_i(1−p_i) | Section 6, Appendix B; Table 1 |
| `03_movielens_pairs.py` | MovieLens-1M: every pair of films with ≥ 50 users, and four tail buckets | Section 7.2 |
| `04_catalog_groups.py` | popularity groups for the other datasets; popularity of a user's items vs. history length | Appendix C; figures in `figures/` |
| `05_ease_lambda.py` | EASE regularization strength (recall@20 on held-out items) | Figure 1, Table 2 |
| `06_main_figure.py` | the four measures under the three models, MovieLens-1M | Figure 1 |
| `07_pair_deviance.py` | Poisson deviance over all item pairs | Equation (deviance) |
| `08_noise_floor.py` | deviance noise floor from simulations of the fitted model; share of the mismatch removed | Table 2 ("Pairs") |
| `09_measure_agreement.py` | standardized block error per measure; popularity surface | Tables 2 and 3 |
| `10_ease_simulated.py` | EASE on simulated data and on the expected Gram matrices | Section 7.2; Table 2 ("EASE") |
| `11_paper_tables.py` | LaTeX tables, number macros and figures for the paper → `results/paper/` | Tables 1–3 |

`notebooks/figures.ipynb` regenerates the figures (Figure 1 and those in `figures/`) and shows them inline; it
needs Jupyter, which is not in `requirements.txt` (`pip install notebook`).

Dependencies between the scripts: 02 first for each dataset (every later script reads `results/<d>/model.npz`);
06 and 10 need 05; 08 needs 07; 11 needs all others.

## Resources

All computations run on a CPU (NumPy/SciPy). The model is fitted from sufficient statistics only (item counts and
the distribution of history lengths), and items with the same count share their parameters, so the fit and the
whole-catalog comparisons scale to Amazon Books (776k users, 495k items) without forming any item × item matrix.
EASE needs a dense item × item matrix: MovieLens-1M runs in seconds; MovieLens-20M and Last.fm need more than
8 GB of RAM (the results were produced on a machine with 14 GB); Amazon Books is out of reach (the paper reports EASE only for the other three datasets). The full
MovieLens-1M pipeline takes a few minutes; the simulations (08, 09, 10) dominate on the large datasets.

## Layout

```
src/phantom/   model.py (activity model, EM, elementary symmetric polynomials), conditional.py (inclusion
               probabilities given the basket size), blocks.py (pooling into blocks and groups), deviance.py,
               ease.py, simulate_fast.py (exact simulation from the model), similarity.py, data.py, paths.py,
               runinfo.py (configuration and seed saved with every result)
experiments/   the scripts above
notebooks/     figures.ipynb (regenerates and shows the figures)
figures/       the figures of the other datasets, referred to in the paper's appendix
tests/         unit tests
```
