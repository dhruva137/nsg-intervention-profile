"""CLI entrypoint for the NSG intervention profile.

Runs the three-variant NSG profile and the explanation-swap control on a
synthetic dataset with configurable true/proxy feature correlation.

Use --dry-run to run the full pipeline on n=30 samples with a stub predictor.
"""
from __future__ import annotations
import argparse
import sys

import pandas as pd

from src.synthetic import make_dataset, decorrelated_counterfactuals
from src.profile import compute_profile
from src.controls import explanation_swap


def stub_predictor(row: pd.Series, explanation: str | None) -> int:
    """Dry-run stub: predict based on true_feature sign, ignore explanation."""
    return int(row["true_feature"] > 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="NSG intervention profile")
    parser.add_argument("--correlation", type=float, default=0.8,
                        help="Rate at which proxy_feature agrees with true_feature")
    parser.add_argument("--n",           type=int, default=500)
    parser.add_argument("--seed",        type=int, default=0)
    parser.add_argument("--dry-run",     action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        args.n = 30
        print("[dry-run] n=30, stub predictor")

    df = make_dataset(args.n, correlation=args.correlation, seed=args.seed)
    print(f"Dataset: {len(df)} rows, {len(decorrelated_counterfactuals(df))} decorrelated")

    # Stub explanations always name the proxy feature
    explanations = [
        f"The prediction was driven by proxy_feature (value {row['proxy_feature']:.2f})."
        for _, row in df.iterrows()
    ]

    profile = compute_profile(stub_predictor, df, explanations, claimed_feature="proxy_feature")
    print("\nNSG Profile:")
    print(profile.summary())

    swapped_expls = explanation_swap(explanations)
    profile_swap = compute_profile(stub_predictor, df, swapped_expls, claimed_feature="proxy_feature")
    print("\nExplanation-swap control:")
    print(profile_swap.summary())


if __name__ == "__main__":
    main()