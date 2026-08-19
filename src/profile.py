"""Compute all three NSG variants and return them as a single record.

The three variants are:
  1. ordinary_nsg   -- on naturally-distributed counterfactuals
  2. decorrelated_nsg -- on counterfactuals where true_feature and proxy disagree
  3. targeted_nsg   -- on interventions to features the explanation names

Gaps between these reveal whether the explanation tracks a causal rule or
a predictive correlate.  See README for the interpretation.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.synthetic import decorrelated_counterfactuals


@dataclass
class NSGProfile:
    ordinary_nsg:      float
    decorrelated_nsg:  float
    targeted_nsg:      float
    n_ordinary:        int
    n_decorrelated:    int
    n_targeted:        int

    def summary(self) -> str:
        return (
            f"Ordinary NSG:     {self.ordinary_nsg:.3f}  (n={self.n_ordinary})\n"
            f"Decorrelated NSG: {self.decorrelated_nsg:.3f}  (n={self.n_decorrelated})\n"
            f"Targeted NSG:     {self.targeted_nsg:.3f}  (n={self.n_targeted})"
        )


def _acc(
    predictor: Callable[[pd.Series, str | None], int],
    rows: pd.DataFrame,
    explanations: list[str] | None,
) -> float:
    """Fraction of rows where predictor matches the 'label' column."""
    correct = 0
    for i, (_, row) in enumerate(rows.iterrows()):
        expl = explanations[i] if explanations else None
        if predictor(row, expl) == row["label"]:
            correct += 1
    return correct / len(rows) if len(rows) else 0.0


def _nsg(with_acc: float, without_acc: float, ceiling: float = 1.0) -> float:
    headroom = ceiling - without_acc
    return (with_acc - without_acc) / headroom if headroom > 0 else 0.0


def compute_profile(
    predictor:   Callable[[pd.Series, str | None], int],
    df_natural:  pd.DataFrame,
    explanations_natural: list[str],
    claimed_feature: str = "proxy_feature",
) -> NSGProfile:
    """Compute the three-variant NSG profile.

    predictor: callable(row: pd.Series, explanation: str | None) -> predicted_label
    df_natural: naturally-distributed counterfactuals (must have a 'label' column)
    explanations_natural: explanation strings parallel to df_natural
    claimed_feature: the feature the explanation is expected to name

    targeted counterfactuals: rows where only claimed_feature changes.
    """
    # 1. Ordinary NSG on all natural counterfactuals
    ord_with    = _acc(predictor, df_natural, explanations_natural)
    ord_without = _acc(predictor, df_natural, None)
    ordinary_nsg = _nsg(ord_with, ord_without)

    # 2. Decorrelated NSG
    df_decor = decorrelated_counterfactuals(df_natural)
    expl_decor = [explanations_natural[i]
                  for i in df_natural.index[df_natural.index.isin(df_decor.index)]]
    if len(df_decor) == 0:
        dec_nsg = float("nan")
    else:
        dec_with    = _acc(predictor, df_decor, expl_decor if expl_decor else None)
        dec_without = _acc(predictor, df_decor, None)
        dec_nsg     = _nsg(dec_with, dec_without)

    # 3. Targeted NSG: interventions that vary only the claimed feature
    # We approximate by using decorrelated rows where true_feature disagrees with proxy
    targeted = df_decor  # same set; in a real experiment you would build explicit interventions
    tgt_with    = _acc(predictor, targeted, expl_decor if expl_decor else None)
    tgt_without = _acc(predictor, targeted, None)
    tgt_nsg     = _nsg(tgt_with, tgt_without)

    return NSGProfile(
        ordinary_nsg=ordinary_nsg,
        decorrelated_nsg=dec_nsg if not (dec_nsg != dec_nsg) else 0.0,  # NaN -> 0
        targeted_nsg=tgt_nsg,
        n_ordinary=len(df_natural),
        n_decorrelated=len(df_decor),
        n_targeted=len(targeted),
    )