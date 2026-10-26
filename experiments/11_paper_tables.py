"""Tables, numbers and figures of the paper, generated from results/.

Writes results/paper/numbers.tex (LaTeX macros for every number quoted in the text), results/paper/
datasets_table.tex (Table 1), results_table.tex (Table 2), trend_table.tex (Table 3), and copies
Figure 1 into results/paper/figures/.

Usage: python experiments/11_paper_tables.py
"""

import inspect
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phantom.blocks import groups_for  # noqa: E402
from phantom.paths import FIGURES, RESULTS  # noqa: E402

OUT = RESULTS / "paper"
IND, ACT = "independence", "activity model"
# macro prefix (letters only) and display name, in table order
DATASETS = {"ml1m_r4": ("mlof", "MovieLens-1M ($\\geq$4)"), "ml20m_r4": ("mltwf", "MovieLens-20M ($\\geq$4)"),
            "lastfm2k": ("lfm", "Last.fm"), "amazon_books": ("amz", "Amazon Books")}
FULL = {"ml1m_r4": "fm"}    # dataset compared pair by pair: prefix of the per-pair macros
FIGS = {   # paper file name <- results/figures/<file>; the other datasets' figures are in figures/ (README)
    "ml1m_r4_paper_matrices.pdf": "ml1m_r4/paper_matrices_wide.pdf",                       # Figure 1
}


def fmt(x, d=2):
    return f"{x:.{d}f}"


def pct(x, d=0):
    return f"{100 * x:.{d}f}\\%"


def thousands(x):
    return f"{int(round(x)):,}".replace(",", "{,}")


def load(tag, name):
    return json.loads((RESULTS / tag / name).read_text())


def sfmt(x):
    """Math mode (proper minus sign), no -0.00."""
    return f"${0.0 if abs(x) < 0.005 else x:.2f}$"


def dataset_macros(tag, px, ma):
    s = load(tag, "summary.json")
    data, fit = s["data"], s["fit"]
    var_ratio = fit["degree_variance"] / fit["independence_variance_sum_p1mp"]
    # total covariance sum_{i != j} Cov(X_i, X_j) = Var(S) - sum_i p_i (1 - p_i)   (Equation (identity))
    tc = fit["degree_variance"] - fit["independence_variance_sum_p1mp"]
    tc_str = thousands(tc) if abs(tc) >= 100 else fmt(tc, 1)
    share = load(tag, "noise_floor.json")["share_structure_explained"]         # Equation (share)
    m = {
        f"{px}DevShare": fmt(0.0 if abs(share) < 0.005 else share, 2),
        f"{px}Users": thousands(data["users"]), f"{px}Items": thousands(data["items"]),
        f"{px}Interactions": thousands(data["interactions"]),
        f"{px}VarS": thousands(fit["degree_variance"]),
        f"{px}VarInd": thousands(fit["independence_variance_sum_p1mp"]),
        f"{px}VarRatio": fmt(var_ratio, 2) if var_ratio < 10 else thousands(var_ratio),
        f"{px}TotalCov": tc_str if tc >= 0 else "$-$" + tc_str.lstrip("-"),
    }
    if tag not in FULL:
        cm = load(tag, "catalog.json")
        m.update({
            f"{px}ProfAct": fmt(cm["profile_mean_abs_error_log10"][ACT], 3),
            f"{px}Groups": str(cm["groups"]["n"]),
            f"{px}LastGroupMaxUsers": thousands(cm["groups"]["least_popular_group_max_users"]),
        })
    # block-level shares per measure (09_measure_agreement.py) and EASE (10_ease_simulated.py)
    for key, mk in [("lift", "Lift"), ("cosine", "Cos"), ("Pearson", "Pearson")]:
        m[px + "Block" + mk] = sfmt(ma[tag][key]["standardized_error"]["share"])
    ef = RESULTS / tag / "ease_simulated.json"
    m[px + "BlockEase"] = sfmt(load(tag, "ease_simulated.json")["standardized_error"]["share"]) if ef.exists() else "--"
    # block pairs left out of the Pearson surface (negative correlation has no logarithm; Table 3 caption)
    m[px + "NonposPearson"] = str(ma[tag]["log_trend"]["Pearson"]["observed"]["nonpositive"])
    m[px + "BlockPairs"] = str(ma[tag]["block_pairs_used"])
    if tag == "ml1m_r4":                        # films per block in the block statistics (groups_for default)
        m["fmStatBlock"] = str(inspect.signature(groups_for).parameters["block"].default)
    return m


def main():
    A = load("synthetic", "opening_example.json")                 # Example 1
    m = {"hookItems": str(A["config"]["n"]), "hookPLight": fmt(A["config"]["p_light"]),
         "hookPHeavy": fmt(A["config"]["p_heavy"]), "hookLiftAnalytic": fmt(A["analytic"]["lift"])}

    ma = json.loads((RESULTS / "measure_agreement.json").read_text())
    rows, res_rows, trend_rows, r2_obs = [], [], [], []
    for tag, (px, name) in DATASETS.items():
        m.update(dataset_macros(tag, px, ma))
        short = name.replace(" ($\\geq$4)", "")   # rating threshold >= 4 stated in Section 7.1
        rows.append(f"{short} & {m[px + 'Users']} & {m[px + 'Items']} & {m[px + 'Interactions']} & "
                    f"{m[px + 'VarS']} & {m[px + 'VarInd']} & {m[px + 'TotalCov']} \\\\")
        res_rows.append(f"{short} & {m[px + 'DevShare']} & {m[px + 'BlockLift']} & {m[px + 'BlockCos']} & "
                        f"{m[px + 'BlockPearson']} & {m[px + 'BlockEase']} \\\\")
        # popularity surface: log(block measure) = a + b (x + y) + c x y + d (x + y)^2, observed and activity
        trend_rows.append(f"\\multicolumn{{9}}{{l}}{{\\emph{{{short}}}}} \\\\")
        for n_row, (key, mname) in enumerate([("lift", "Lift"), ("cosine", "Cosine"), ("Pearson", "Pearson")]):
            lt = ma[tag]["log_trend"][key]
            cells = []
            for coef in ["a", "b", "c", "d"]:
                for side in ["observed", "activity"]:
                    ok = lt[side]["nonpositive"] <= 0.1 * ma[tag]["block_pairs_used"]
                    cells.append(sfmt(lt[side]["quadratic"][coef]) if ok else "--")
            trend_rows.append(f"{mname} & " + " & ".join(cells) + " \\\\")
        for key in ["lift", "cosine", "Pearson"]:                    # R^2 of the observed fits shown in Table 3
            lt = ma[tag]["log_trend"][key]
            if lt["observed"]["nonpositive"] <= 0.1 * ma[tag]["block_pairs_used"]:
                r2_obs.append(lt["observed"]["quadratic"]["R2"])
        trend_rows.append("\\midrule")

    # MovieLens-1M: every pair of films with >= 50 users, and the tail buckets (03_movielens_pairs.py)
    for tag, px in FULL.items():
        fm = load(tag, "full_matrix.json")
        a, ib = fm[ACT], fm[IND]
        rare = fm["buckets"][-1]                                   # least popular bucket
        m.update({
            f"{px}Items": thousands(fm["dense_items"]), f"{px}MinUsers": str(fm["min_users"]),
            f"{px}Pairs": fmt(fm["pairs"] / 1e6, 2) + "\\,M",
            f"{px}DenseShare": pct(fm["dense_share_of_interactions"]),
            f"{px}NeverPct": pct(fm["observed"]["share_pairs_never_cooccur"], 1),
            f"{px}IndWithinTwo": pct(ib["share_pairs_ratio_within_factor_2"]),
            f"{px}ActWithinTwo": pct(a["share_pairs_ratio_within_factor_2"]),
            f"{px}ActSpearman": fmt(a["spearman_lift_all_pairs"]),
            f"{px}RareBucketUsers": rare["users"].replace("–", "--"),
            f"{px}RareBucketItems": thousands(rare["items"]),
            f"{px}RareBucketShare": pct(rare["share_of_interactions"], 1),
            f"{px}RareWithinAct": fmt(rare[f"obs_over_model_within ({ACT})"], 1),
            f"{px}TailItems": thousands(sum(r["items"] for r in fm["buckets"])),
            f"{px}RareEffUsers": fmt(rare["effective_users_within"], 1),
            f"{px}RareContribUsers": str(rare["users_contributing_within"]),
        })
    m["catalogGroups"] = str(json.loads((RESULTS / "ml20m_r4" / "catalog.json").read_text())["run"]["config"]["groups"])

    OUT.mkdir(parents=True, exist_ok=True)
    head = "% AUTO-GENERATED by experiments/11_paper_tables.py. Do not edit by hand.\n"
    m["trendRtwoMin"], m["trendRtwoMax"] = fmt(min(r2_obs)), fmt(max(r2_obs))
    (OUT / "numbers.tex").write_text(head + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()) + "\n")
    (OUT / "datasets_table.tex").write_text(
        head + "\\setlength{\\tabcolsep}{1.8pt}\n\\begin{tabular}{lrrrrrr}\n\\toprule\n"
        "Dataset & Users & Items & Interact. & $\\Var(S)$ & $\\sum\\limits_i\\Var(X_i)$ & "
        "$\\sum\\limits_{i\\neq j}\\Cov(X_i,X_j)$ \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    (OUT / "results_table.tex").write_text(
        head + "\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lrrrrr}\n\\toprule\n"
        "& Pairs & \\multicolumn{4}{c}{Blocks} \\\\\n\\cmidrule(lr){2-2}\\cmidrule(lr){3-6}\n"
        "Dataset & all & lift & cosine & Pearson & EASE \\\\\n\\midrule\n" + "\n".join(res_rows)
        + "\n\\bottomrule\n\\end{tabular}\n")
    (OUT / "trend_table.tex").write_text(
        head + "\\setlength{\\tabcolsep}{2.5pt}\n\\begin{tabular}{lrrrrrrrr}\n\\toprule\n"
        "& \\multicolumn{2}{c}{$a$ (level)} & \\multicolumn{2}{c}{$b$ (popularity)} & \\multicolumn{2}{c}{$c$ (imbalance)} "
        "& \\multicolumn{2}{c}{$d$ (curvature)} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n"
        "& obs. & act. & obs. & act. & obs. & act. & obs. & act. \\\\\n\\midrule\n"
        + "\n".join(trend_rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")
    figs = OUT / "figures"
    figs.mkdir(exist_ok=True)
    for dst, src in FIGS.items():
        if (FIGURES / src).exists():
            shutil.copy(FIGURES / src, figs / dst)
        else:
            print(f"WARNING: figure {src} missing, not copied")
    print(f"wrote {len(m)} macros and three tables to {OUT}")


if __name__ == "__main__":
    main()
