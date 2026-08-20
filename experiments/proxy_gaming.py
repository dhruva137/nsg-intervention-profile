#!/usr/bin/env python
"""Core experiment: can a proxy explanation earn NSG comparable to an honest one, and
does that advantage survive decorrelating the proxy from the true cause?

Setup (see src/synthetic.py): `label = true_feature` exactly, so the reference decision
rule is exact by construction. `proxy_feature` agrees with `true_feature` at a
controllable rate rho. Two fixed-template explanations -- "honest" names true_feature,
"proxy" names proxy_feature. A logistic-regression predictor is fit once per condition on
data drawn from the natural (rho-correlated) distribution, then evaluated for NSG three
ways: on a fresh natural-distribution draw ("ordinary"), on that draw's decorrelated
subset (rows where true and proxy disagree), and with the explanation-swap control from
src/controls.py.

Runs entirely on CPU with scikit-learn; no GPU, no API key, no download. Prints a table
and saves results/proxy_gaming.png. See src/nsg.py's module docstring for the one
non-obvious implementation detail this experiment depends on: the predictor must be fit
on the natural distribution and evaluated out-of-distribution, not fit fresh on whatever
subset is being scored.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.controls import explanation_swap
from src.nsg import fit_predictor, make_explanations, normalised_gain, simulatability
from src.plotting import COLORS, save_figure, set_style  # sets the Agg backend
from src.synthetic import decorrelated_subset, make_dataset

import matplotlib.pyplot as plt  # noqa: E402  (after src.plotting sets the Agg backend)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RHOS_FULL = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
N_TRAIN_FULL = 1000
N_EVAL_FULL = 3000
N_SEEDS_FULL = 20


def _condition_nsg(
    correlation: float,
    named_feature: str,
    seed: int,
    n_train: int,
    n_eval: int,
) -> tuple[float, float, float]:
    """Fit once on a natural-distribution training draw, then return (ordinary,
    decorrelated, swap) NSG for one explanation condition at one (rho, seed).
    """
    train_df = make_dataset(n_train, correlation, seed=seed)
    eval_df = make_dataset(n_eval, correlation, seed=seed + 1_000_003)

    baseline_model = fit_predictor(LogisticRegression(max_iter=1000), train_df, None)
    explained_model = fit_predictor(
        LogisticRegression(max_iter=1000), train_df, make_explanations(len(train_df), named_feature)
    )

    def nsg_on(items: pd.DataFrame, explanations) -> float:
        acc_without = simulatability(baseline_model, items, None)
        acc_with = simulatability(explained_model, items, explanations)
        return normalised_gain(acc_with, acc_without)

    ordinary = nsg_on(eval_df, make_explanations(len(eval_df), named_feature))

    decor = decorrelated_subset(eval_df)
    decorrelated = nsg_on(decor, make_explanations(len(decor), named_feature)) if len(decor) >= 5 else float("nan")

    rng = random.Random(seed)
    swapped = explanation_swap(make_explanations(len(eval_df), named_feature), rng)
    swap = nsg_on(eval_df, swapped)

    return ordinary, decorrelated, swap


def _bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    """Mean and a 95% percentile bootstrap CI over the per-seed NSG values."""
    values = values[~np.isnan(values)]
    if len(values) < 2:
        m = float(values[0]) if len(values) else float("nan")
        return m, m, m
    res = sp_stats.bootstrap(
        (values,), np.mean, confidence_level=0.95, n_resamples=2000, method="percentile",
        random_state=seed,
    )
    return float(values.mean()), float(res.confidence_interval.low), float(res.confidence_interval.high)


def run_sweep(rhos, n_train: int, n_eval: int, n_seeds: int, base_seed: int) -> pd.DataFrame:
    """Sweep rho, running n_seeds independent (train, eval) draws per (condition, rho).

    Returns a tidy DataFrame with one row per (rho, seed) and columns for every
    condition x regime NSG value -- summarised into the printed table by the caller.
    """
    records = []
    for rho in rhos:
        for s in range(n_seeds):
            seed = base_seed + s
            h_ord, h_dec, h_swap = _condition_nsg(rho, "true_feature", seed, n_train, n_eval)
            p_ord, p_dec, p_swap = _condition_nsg(rho, "proxy_feature", seed, n_train, n_eval)
            records.append(
                {
                    "rho": rho,
                    "seed": seed,
                    "honest_ordinary": h_ord,
                    "proxy_ordinary": p_ord,
                    "honest_decorrelated": h_dec,
                    "proxy_decorrelated": p_dec,
                    "honest_swap": h_swap,
                    "proxy_swap": p_swap,
                }
            )
    return pd.DataFrame.from_records(records)


def summarise(raw: pd.DataFrame, base_seed: int) -> pd.DataFrame:
    """Collapse per-seed draws into mean + 95% bootstrap CI per rho, for every column."""
    value_cols = [c for c in raw.columns if c not in ("rho", "seed")]
    rows = []
    for rho, group in raw.groupby("rho"):
        row = {"rho": rho}
        for col in value_cols:
            mean, lo, hi = _bootstrap_ci(group[col].to_numpy(), seed=base_seed)
            row[col] = mean
            row[f"{col}_lo"] = lo
            row[f"{col}_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows).sort_values("rho").reset_index(drop=True)


def format_main_table(summary: pd.DataFrame) -> str:
    cols = ["rho", "honest_ordinary", "proxy_ordinary", "honest_decorrelated", "proxy_decorrelated"]
    lines = [f"{'rho':>5} {'honest_ordinary':>22} {'proxy_ordinary':>22} {'honest_decorrelated':>24} {'proxy_decorrelated':>24}"]
    for _, r in summary.iterrows():
        cells = [f"{r['rho']:.2f}"]
        for c in cols[1:]:
            cells.append(f"{r[c]:+.3f} [{r[c + '_lo']:+.3f}, {r[c + '_hi']:+.3f}]")
        lines.append(f"{cells[0]:>5} {cells[1]:>22} {cells[2]:>22} {cells[3]:>24} {cells[4]:>24}")
    return "\n".join(lines)


def format_swap_table(summary: pd.DataFrame) -> str:
    lines = [f"{'rho':>5} {'honest_swap':>22} {'proxy_swap':>22}  (explanation-swap control)"]
    for _, r in summary.iterrows():
        cells = [
            f"{r['rho']:.2f}",
            f"{r['honest_swap']:+.3f} [{r['honest_swap_lo']:+.3f}, {r['honest_swap_hi']:+.3f}]",
            f"{r['proxy_swap']:+.3f} [{r['proxy_swap_lo']:+.3f}, {r['proxy_swap_hi']:+.3f}]",
        ]
        lines.append(f"{cells[0]:>5} {cells[1]:>22} {cells[2]:>22}")
    return "\n".join(lines)


def plot_results(summary: pd.DataFrame, out_path: Path) -> None:
    set_style()
    fig, ax = plt.subplots()
    for cond, color, marker in (("honest", COLORS["honest"], "o"), ("proxy", COLORS["proxy"], "s")):
        for regime, ls in (("ordinary", "-"), ("decorrelated", "--")):
            col = f"{cond}_{regime}"
            ax.plot(
                summary["rho"], summary[col],
                color=color, linestyle=ls, marker=marker,
                label=f"{cond} explanation, {regime}",
            )
            ax.fill_between(summary["rho"], summary[f"{col}_lo"], summary[f"{col}_hi"], color=color, alpha=0.12)
    ax.axhline(0.0, color=COLORS["neutral"], linewidth=0.8, linestyle=":")
    ax.set_xlabel(r"proxy--true correlation $\rho$")
    ax.set_ylabel("Normalised Simulatability Gain")
    ax.set_title("Proxy explanations gain NSG on natural counterfactuals, lose it decorrelated")
    ax.legend(loc="center left", fontsize=8.5)
    save_figure(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="tiny config, exercises the full path in <5s")
    parser.add_argument("--seed", type=int, default=0, help="base seed for the sweep")
    args = parser.parse_args()

    if args.dry_run:
        rhos, n_train, n_eval, n_seeds = (0.7, 0.95), 40, 120, 2
    else:
        rhos, n_train, n_eval, n_seeds = RHOS_FULL, N_TRAIN_FULL, N_EVAL_FULL, N_SEEDS_FULL

    t0 = time.time()
    raw = run_sweep(rhos, n_train, n_eval, n_seeds, base_seed=args.seed)
    summary = summarise(raw, base_seed=args.seed)
    elapsed = time.time() - t0

    print(f"proxy_gaming.py -- {len(rhos)} rho values x {n_seeds} seeds "
          f"(n_train={n_train}, n_eval={n_eval}), {elapsed:.1f}s\n")
    print("Mean NSG with 95% bootstrap CI over seeds:")
    print(format_main_table(summary))
    print()
    print("Explanation-swap control (third way of computing NSG, see spec):")
    print(format_swap_table(summary))
    print()
    print(
        "Note on the swap control: explanations here are fixed templates ('The decision "
        "was based on X.'), identical across every row of a condition. Swapping in another "
        "row's copy of the same string changes nothing, so honest_swap == honest_ordinary "
        "and proxy_swap == proxy_ordinary by construction -- see src/controls.py's "
        "docstring. shuffle_explanation and length_matched_null (experiments/"
        "explanation_controls.py) are the controls that actually alter the string and "
        "collapse NSG."
    )

    if not args.dry_run:
        out_path = RESULTS_DIR / "proxy_gaming.png"
        plot_results(summary, out_path)
        print(f"\nSaved figure to {out_path}")
    else:
        print("\n--dry-run: skipping figure save.")


if __name__ == "__main__":
    main()
