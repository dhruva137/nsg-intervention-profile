#!/usr/bin/env python
"""Do the explanation-swap, shuffle, and null controls survive? If so, NSG in this
setup is picking up scaffolding, not information.

Two things happen here, both at a fixed, representative rho=0.9:

1. The three-number intervention profile (src/nsg.intervention_profile) for the honest
   and proxy explanations -- ordinary, decorrelated, targeted NSG -- printed as a worked
   example of what the profile looks like and why a single average would hide the gap
   between the ordinary and the other two numbers.

2. Three controls applied to the *evaluation-time* explanation string, holding the
   already-fitted predictor fixed (src/controls.py):
   - explanation_swap: another example's copy of the explanation
   - shuffle_explanation: same words, random order
   - length_matched_null: same length, no feature named
   Real NSG should clearly beat all three. Any control that matches real NSG indicates
   the predictor was never using the explanation's specific content in the first place.

CPU-only, no LLM. Prints two tables and saves results/explanation_controls.png (a grouped
bar chart of NSG per control, per condition).
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

from src.controls import explanation_swap, length_matched_null, shuffle_explanation
from src.nsg import fit_predictor, intervention_profile, make_explanations, normalised_gain, simulatability
from src.plotting import COLORS, save_figure, set_style  # sets the Agg backend
from src.synthetic import make_dataset

import matplotlib.pyplot as plt  # noqa: E402  (after src.plotting sets the Agg backend)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RHO_FULL = 0.9
N_TRAIN_FULL = 1000
N_EVAL_FULL = 3000
N_SEEDS_FULL = 10


def _fit_condition(correlation: float, named_feature: str, seed: int, n_train: int):
    train_df = make_dataset(n_train, correlation, seed=seed)
    baseline_model = fit_predictor(LogisticRegression(max_iter=1000), train_df, None)
    explained_model = fit_predictor(
        LogisticRegression(max_iter=1000), train_df, make_explanations(len(train_df), named_feature)
    )
    return baseline_model, explained_model


def _nsg_for(baseline_model, explained_model, eval_df: pd.DataFrame, eval_explanations) -> float:
    acc_without = simulatability(baseline_model, eval_df, None)
    acc_with = simulatability(explained_model, eval_df, eval_explanations)
    return normalised_gain(acc_with, acc_without)


def run_controls_sweep(correlation: float, n_train: int, n_eval: int, n_seeds: int, base_seed: int) -> pd.DataFrame:
    """For honest and proxy explanations, compute NSG under: the real explanation, the
    explanation-swap control, the shuffled explanation, and the length-matched null.
    """
    records = []
    for named_feature, label in (("true_feature", "honest"), ("proxy_feature", "proxy")):
        for s in range(n_seeds):
            seed = base_seed + s
            baseline_model, explained_model = _fit_condition(correlation, named_feature, seed, n_train)
            eval_df = make_dataset(n_eval, correlation, seed=seed + 1_000_003)
            real = make_explanations(len(eval_df), named_feature)

            rng = random.Random(seed)
            swapped = explanation_swap(real, rng)
            shuffled = [shuffle_explanation(e, rng) for e in real]
            null = [length_matched_null(e) for e in real]

            records.append(
                {
                    "condition": label,
                    "seed": seed,
                    "real": _nsg_for(baseline_model, explained_model, eval_df, real),
                    "swap": _nsg_for(baseline_model, explained_model, eval_df, swapped),
                    "shuffled": _nsg_for(baseline_model, explained_model, eval_df, shuffled),
                    "null": _nsg_for(baseline_model, explained_model, eval_df, null),
                }
            )
    return pd.DataFrame.from_records(records)


def _bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = values[~np.isnan(values)]
    if len(values) < 2:
        m = float(values[0]) if len(values) else float("nan")
        return m, m, m
    res = sp_stats.bootstrap(
        (values,), np.mean, confidence_level=0.95, n_resamples=2000, method="percentile", random_state=seed
    )
    return float(values.mean()), float(res.confidence_interval.low), float(res.confidence_interval.high)


def summarise_controls(raw: pd.DataFrame, base_seed: int) -> pd.DataFrame:
    rows = []
    for condition, group in raw.groupby("condition"):
        row = {"condition": condition}
        for col in ("real", "swap", "shuffled", "null"):
            mean, lo, hi = _bootstrap_ci(group[col].to_numpy(), seed=base_seed)
            row[col] = mean
            row[f"{col}_lo"] = lo
            row[f"{col}_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def format_controls_table(summary: pd.DataFrame) -> str:
    header = f"{'condition':>10} {'real':>20} {'swap':>20} {'shuffled':>20} {'null':>20}"
    lines = [header]
    for _, r in summary.iterrows():
        cells = [r["condition"]]
        for c in ("real", "swap", "shuffled", "null"):
            cells.append(f"{r[c]:+.3f} [{r[c + '_lo']:+.3f}, {r[c + '_hi']:+.3f}]")
        lines.append(f"{cells[0]:>10} {cells[1]:>20} {cells[2]:>20} {cells[3]:>20} {cells[4]:>20}")
    return "\n".join(lines)


def run_profile_demo(correlation: float, n_train: int, n_eval: int, n_seeds: int, base_seed: int) -> pd.DataFrame:
    records = []
    for named_feature, label in (("true_feature", "honest"), ("proxy_feature", "proxy")):
        for s in range(n_seeds):
            seed = base_seed + s
            profile = intervention_profile(
                LogisticRegression(max_iter=1000), correlation, named_feature,
                n_train=n_train, n_eval=n_eval, seed=seed,
            )
            records.append(
                {
                    "condition": label,
                    "seed": seed,
                    "ordinary": profile.ordinary,
                    "decorrelated": profile.decorrelated,
                    "targeted": profile.targeted,
                }
            )
    return pd.DataFrame.from_records(records)


def summarise_profile(raw: pd.DataFrame, base_seed: int) -> pd.DataFrame:
    rows = []
    for condition, group in raw.groupby("condition"):
        row = {"condition": condition}
        for col in ("ordinary", "decorrelated", "targeted"):
            mean, lo, hi = _bootstrap_ci(group[col].to_numpy(), seed=base_seed)
            row[col] = mean
            row[f"{col}_lo"] = lo
            row[f"{col}_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def format_profile_table(summary: pd.DataFrame) -> str:
    header = f"{'condition':>10} {'ordinary':>20} {'decorrelated':>20} {'targeted':>20}"
    lines = [header]
    for _, r in summary.iterrows():
        cells = [r["condition"]]
        for c in ("ordinary", "decorrelated", "targeted"):
            cells.append(f"{r[c]:+.3f} [{r[c + '_lo']:+.3f}, {r[c + '_hi']:+.3f}]")
        lines.append(f"{cells[0]:>10} {cells[1]:>20} {cells[2]:>20} {cells[3]:>20}")
    return "\n".join(lines)


def plot_controls(summary: pd.DataFrame, rho: float, out_path: Path) -> None:
    """Grouped bar chart: NSG under each control, per condition. The visual point is
    that real and swap bars stand tall while shuffled and null collapse toward zero.
    """
    set_style()
    fig, ax = plt.subplots()
    control_order = ["real", "swap", "shuffled", "null"]
    x = np.arange(len(control_order))
    width = 0.35
    for offset, (cond, color) in zip((-1, 1), (("honest", COLORS["honest"]), ("proxy", COLORS["proxy"]))):
        row = summary[summary["condition"] == cond].iloc[0]
        means = [row[c] for c in control_order]
        los = [row[c] - row[f"{c}_lo"] for c in control_order]
        his = [row[f"{c}_hi"] - row[c] for c in control_order]
        ax.bar(
            x + offset * width / 2, means, width=width, color=color, label=cond,
            yerr=[los, his], capsize=3,
        )
    ax.axhline(0.0, color=COLORS["neutral"], linewidth=0.8, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(control_order)
    ax.set_ylabel("Normalised Simulatability Gain")
    ax.set_title(f"NSG under each control (rho={rho})")
    ax.legend()
    save_figure(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="tiny config, exercises the full path in <5s")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.dry_run:
        rho, n_train, n_eval, n_seeds = 0.9, 40, 120, 2
    else:
        rho, n_train, n_eval, n_seeds = RHO_FULL, N_TRAIN_FULL, N_EVAL_FULL, N_SEEDS_FULL

    t0 = time.time()
    profile_raw = run_profile_demo(rho, n_train, n_eval, n_seeds, base_seed=args.seed)
    profile_summary = summarise_profile(profile_raw, base_seed=args.seed)
    controls_raw = run_controls_sweep(rho, n_train, n_eval, n_seeds, base_seed=args.seed)
    controls_summary = summarise_controls(controls_raw, base_seed=args.seed)
    elapsed = time.time() - t0

    print(f"explanation_controls.py -- rho={rho}, {n_seeds} seeds, {elapsed:.1f}s\n")
    print("Intervention profile (mean +/- 95% bootstrap CI over seeds):")
    print(format_profile_table(profile_summary))
    print()
    print("Controls comparison, ordinary-regime NSG (mean +/- 95% bootstrap CI over seeds):")
    print(format_controls_table(controls_summary))
    print()

    honest_row = controls_summary[controls_summary["condition"] == "honest"].iloc[0]
    proxy_row = controls_summary[controls_summary["condition"] == "proxy"].iloc[0]
    threshold = 0.1
    for label, row in (("honest", honest_row), ("proxy", proxy_row)):
        survives = [c for c in ("shuffled", "null") if row[c] > threshold]
        if survives:
            print(f"{label}: {', '.join(survives)} control(s) still above {threshold:+.2f} NSG -- scaffolding effect present.")
        else:
            print(f"{label}: shuffled and null controls both collapse to <= {threshold:+.2f} NSG -- "
                  f"real NSG ({row['real']:+.3f}) reflects genuine explanation content, not scaffolding.")
        if abs(row["swap"] - row["real"]) < 1e-6:
            print(f"{label}: swap == real exactly, as expected for this repo's fixed-template explanations "
                  f"(see src/controls.py docstring) -- not evidence either way about scaffolding.")

    if not args.dry_run:
        out_path = RESULTS_DIR / "explanation_controls.png"
        plot_controls(controls_summary, rho, out_path)
        print(f"\nSaved figure to {out_path}")
    else:
        print("\n--dry-run: results above are from a tiny sweep, not representative. Skipping figure save.")


if __name__ == "__main__":
    main()
