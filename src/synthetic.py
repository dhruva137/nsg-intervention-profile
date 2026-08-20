"""Synthetic proxy-gaming dataset.

The whole point of this repo's core experiment is that ground truth has to be *exact* --
otherwise "does the explanation name the real cause" is not well-defined. So the
reference decision rule here is trivial by construction: ``label = true_feature``. Every
row also carries a ``proxy_feature`` that agrees with ``true_feature`` at a controllable
rate (``correlation``), and four ``noise_*`` features that are independent of the label
and included only so a predictor has other things to look at.

All features are binary (0/1). That is a deliberate simplification, not an oversight: it
makes "counterfactual" mean "flip one or more bits" and "Hamming distance" a literal,
countable quantity, which is exactly what `experiments/counterfactual_distance.py` needs.
"""
from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd

TRUE_FEATURE = "true_feature"
PROXY_FEATURE = "proxy_feature"
NOISE_FEATURES: tuple[str, ...] = ("noise_1", "noise_2", "noise_3", "noise_4")
FEATURE_NAMES: tuple[str, ...] = (TRUE_FEATURE, PROXY_FEATURE, *NOISE_FEATURES)
LABEL_COL = "label"


def make_dataset(n: int, correlation: float, seed: int) -> pd.DataFrame:
    """Generate `n` rows with a true decision feature, a correlated proxy, and noise.

    `label` equals `true_feature` exactly, always -- this is the reference model: a
    decision rule that looks at nothing but `true_feature`. `proxy_feature` equals
    `true_feature` with probability `correlation` and is flipped otherwise, so
    `correlation` is the natural-distribution correlation between the true cause and the
    proxy an explanation might name instead. `noise_1..noise_4` are independent coin
    flips, uninformative about the label by construction.

    Columns: true_feature, proxy_feature, noise_1, noise_2, noise_3, noise_4, label.
    """
    if not 0.0 <= correlation <= 1.0:
        raise ValueError(f"correlation must be in [0, 1], got {correlation}")
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")

    rng = np.random.default_rng(seed)
    true_feature = rng.integers(0, 2, size=n)
    agrees = rng.random(n) < correlation
    proxy_feature = np.where(agrees, true_feature, 1 - true_feature)
    noise = rng.integers(0, 2, size=(n, len(NOISE_FEATURES)))

    data = {TRUE_FEATURE: true_feature, PROXY_FEATURE: proxy_feature}
    for i, name in enumerate(NOISE_FEATURES):
        data[name] = noise[:, i]
    data[LABEL_COL] = true_feature.copy()
    return pd.DataFrame(data)


def decorrelated_subset(rows: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where true_feature and proxy_feature disagree.

    This is the region where the natural-distribution correlation the proxy explanation
    relies on has been broken -- exactly the rows an observer would need to get right to
    show it is tracking the real cause rather than a correlate of it.
    """
    disagree = rows[TRUE_FEATURE] != rows[PROXY_FEATURE]
    return rows.loc[disagree].reset_index(drop=True)


def make_counterfactuals(
    row: pd.Series,
    hamming_distance: int,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """Enumerate every counterfactual of `row` that flips exactly `hamming_distance`
    of `feature_names` (binary bit-flips), one output row per combination.

    Used two ways in this repo: (1) `counterfactual_distance.py` sweeps
    `hamming_distance` over the full feature set to quantify how the probability of
    touching the decision-relevant feature falls as the feature space grows -- Mayne et
    al.'s own stated limitation; (2) the "targeted" leg of `nsg.intervention_profile`
    calls this with `hamming_distance=1` and `feature_names=[named_feature]` to build the
    single counterfactual that intervenes on exactly the feature an explanation claims
    matters.

    If `true_feature` and `label` are both present, `label` is recomputed from the
    (possibly flipped) `true_feature` in the output rows, because the reference decision
    rule is `label = true_feature` -- reusing the stale label would silently break the
    "ground truth is exact" property this whole repo leans on.
    """
    feature_names = list(feature_names)
    if hamming_distance < 0:
        raise ValueError(f"hamming_distance must be >= 0, got {hamming_distance}")
    if hamming_distance > len(feature_names):
        raise ValueError(
            f"hamming_distance ({hamming_distance}) exceeds len(feature_names) ({len(feature_names)})"
        )
    missing = [f for f in feature_names if f not in row.index]
    if missing:
        raise KeyError(f"row is missing feature(s): {missing}")

    out_rows = []
    for combo in combinations(feature_names, hamming_distance):
        new_row = row.copy()
        for feat in combo:
            new_row[feat] = 1 - int(new_row[feat])
        if TRUE_FEATURE in new_row.index and LABEL_COL in new_row.index:
            new_row[LABEL_COL] = int(new_row[TRUE_FEATURE])
        out_rows.append(new_row)

    if not out_rows:
        return pd.DataFrame(columns=row.index)
    return pd.DataFrame(out_rows).reset_index(drop=True)
