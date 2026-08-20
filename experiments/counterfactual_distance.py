#!/usr/bin/env python
"""Quantify Mayne et al.'s own stated limitation: as the feature space grows, a random
counterfactual is decreasingly likely to touch the decision-relevant feature at all.

Mayne et al. (https://arxiv.org/abs/2602.02639) note that their Income-dataset example --
where the explanation ignores race and marital status yet the pair still shows positive
average NSG -- is limited by counterfactual distance and the quality of the selected
counterfactual region. This script makes that concrete: if a counterfactual is built by
flipping a random subset of `hamming_distance` features out of `k` total, the probability
it includes any one specific feature (the true decision-relevant one) is exactly
`hamming_distance / k` by symmetry -- a plain combinatorial fact, not a simulation
result, but useful to see plotted and to double-check against an actual simulation.

Two things this explains: (1) why an average NSG over many counterfactuals can stay
positive even when the explanation ignores an important feature -- most counterfactuals
in a high-dimensional feature space never perturb that feature, so they cannot penalise
an explanation for ignoring it; (2) why this repo's own "decorrelated" and "targeted"
legs (src/nsg.py) matter as a supplement to plain average NSG rather than a replacement
for it -- they deliberately select or construct the counterfactuals an average sweep
would rarely sample.

CPU-only, no LLM. Prints a table and saves results/counterfactual_distance.png.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plotting import COLORS, save_figure, set_style
from src.synthetic import FEATURE_NAMES, TRUE_FEATURE, make_counterfactuals, make_dataset

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
HAMMING_DISTANCES_FULL = (1, 2, 3, 4)
FEATURE_SPACE_SIZES_FULL = (6, 10, 20, 50, 100)
N_TRIALS_FULL = 5000


def analytic_probability(hamming_distance: int, feature_space_size: int) -> float:
    """P(a uniformly random size-`hamming_distance` subset of `feature_space_size`
    features includes one specific feature) = hamming_distance / feature_space_size.

    Derivation: of the C(k, d) equally likely size-d subsets, C(k-1, d-1) contain a fixed
    feature. C(k-1, d-1) / C(k, d) simplifies to d / k. Computed here via math.comb
    rather than the closed form so the two can be checked against each other.
    """
    if hamming_distance > feature_space_size:
        return float("nan")
    return math.comb(feature_space_size - 1, hamming_distance - 1) / math.comb(feature_space_size, hamming_distance)


def simulate_probability(hamming_distance: int, feature_space_size: int, n_trials: int, seed: int) -> float:
    """Monte Carlo estimate of the same probability: repeatedly pick a random size-d
    subset of k feature indices and check whether index 0 (the "true" feature) is in it.
    """
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_trials):
        chosen = rng.choice(feature_space_size, size=hamming_distance, replace=False)
        hits += int(0 in chosen)
    return hits / n_trials


def grounded_check(hamming_distances, n_trials: int, seed: int) -> pd.DataFrame:
    """Ground the abstract calculation in this repo's actual 6-feature synthetic dataset
    and its actual `synthetic.make_counterfactuals` implementation, rather than trusting
    the abstract simulation alone: for each hamming distance, enumerate every real
    counterfactual of a real row and check the fraction that flip true_feature.
    """
    df = make_dataset(n=1, correlation=0.9, seed=seed)
    row = df.iloc[0]
    k = len(FEATURE_NAMES)
    records = []
    for d in hamming_distances:
        if d > k:
            continue
        cfs = make_counterfactuals(row, hamming_distance=d, feature_names=list(FEATURE_NAMES))
        touches_true = (cfs[TRUE_FEATURE] != row[TRUE_FEATURE]).mean()
        records.append({"hamming_distance": d, "feature_space_size": k, "p_grounded": touches_true})
    return pd.DataFrame(records)


def run_sweep(hamming_distances, feature_space_sizes, n_trials: int, seed: int) -> pd.DataFrame:
    records = []
    for k in feature_space_sizes:
        for d in hamming_distances:
            if d > k:
                continue
            p_analytic = analytic_probability(d, k)
            p_sim = simulate_probability(d, k, n_trials, seed=seed + d * 1000 + k)
            records.append(
                {
                    "hamming_distance": d,
                    "feature_space_size": k,
                    "p_analytic": p_analytic,
                    "p_simulated": p_sim,
                    "abs_diff": abs(p_analytic - p_sim),
                }
            )
    return pd.DataFrame(records)


def format_table(sweep: pd.DataFrame) -> str:
    header = f"{'d (hamming)':>12} {'k (features)':>13} {'p_analytic':>12} {'p_simulated':>12} {'abs_diff':>10}"
    lines = [header]
    for _, r in sweep.iterrows():
        lines.append(
            f"{int(r['hamming_distance']):>12} {int(r['feature_space_size']):>13} "
            f"{r['p_analytic']:>12.4f} {r['p_simulated']:>12.4f} {r['abs_diff']:>10.4f}"
        )
    return "\n".join(lines)


def plot_results(sweep: pd.DataFrame, feature_space_sizes, out_path: Path) -> None:
    set_style()
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("viridis")
    hamming_values = sorted(sweep["hamming_distance"].unique())
    for i, d in enumerate(hamming_values):
        sub = sweep[sweep["hamming_distance"] == d].sort_values("feature_space_size")
        color = cmap(i / max(1, len(hamming_values) - 1))
        ax.plot(sub["feature_space_size"], sub["p_analytic"], color=color, marker="o", label=f"d={d} (analytic)")
        ax.scatter(sub["feature_space_size"], sub["p_simulated"], color=color, marker="x", s=40, zorder=5)
    ax.set_xlabel("feature space size k")
    ax.set_ylabel("P(counterfactual touches the decision-relevant feature)")
    ax.set_title("Probability a counterfactual touches the true feature falls as k grows")
    ax.set_xscale("log")
    ax.legend(fontsize=8.5)
    save_figure(fig, out_path)


import matplotlib.pyplot as plt  # noqa: E402  (after src.plotting sets the Agg backend)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="tiny config, exercises the full path in <5s")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.dry_run:
        hamming_distances = (1, 2)
        feature_space_sizes = (6, 10)
        n_trials = 10
    else:
        hamming_distances = HAMMING_DISTANCES_FULL
        feature_space_sizes = FEATURE_SPACE_SIZES_FULL
        n_trials = N_TRIALS_FULL

    t0 = time.time()
    grounded = grounded_check(hamming_distances, n_trials=n_trials, seed=args.seed)
    sweep = run_sweep(hamming_distances, feature_space_sizes, n_trials, seed=args.seed)
    elapsed = time.time() - t0

    print(f"counterfactual_distance.py -- {elapsed:.1f}s\n")
    print("Grounded check against this repo's actual synthetic dataset (k=6 real features,")
    print("synthetic.make_counterfactuals enumerating every real combination):")
    for _, r in grounded.iterrows():
        analytic = analytic_probability(int(r["hamming_distance"]), int(r["feature_space_size"]))
        print(
            f"  d={int(r['hamming_distance'])}  k={int(r['feature_space_size'])}  "
            f"p_grounded={r['p_grounded']:.4f}  p_analytic={analytic:.4f}"
        )
    print()
    print("Abstract sweep over hamming distance x feature-space size (analytic vs Monte Carlo):")
    print(format_table(sweep))
    print()
    print(
        "Reading this against the paper: the Adult/Income dataset the paper's own example "
        "draws on has well over a dozen usable columns. If a counterfactual generator "
        "perturbs a handful of features at a time (small d) out of that many (large k), "
        "most sampled counterfactuals never touch race or marital status at all -- so an "
        "explanation that ignores those features pays almost no NSG penalty on average, "
        "which is exactly the paper's own caveat about counterfactual distance."
    )

    if not args.dry_run:
        out_path = RESULTS_DIR / "counterfactual_distance.png"
        plot_results(sweep, feature_space_sizes, out_path)
        print(f"\nSaved figure to {out_path}")
    else:
        print("\n--dry-run: skipping figure save.")


if __name__ == "__main__":
    main()
