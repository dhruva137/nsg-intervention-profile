"""Normalised Simulatability Gain (NSG) and the intervention profile.

NSG (Mayne et al., https://arxiv.org/abs/2602.02639) asks a separate predictor model to
guess what a reference model would do on a counterfactual input, with and without the
reference model's own explanation of its original answer. The gain from seeing the
explanation is normalised against the headroom above the no-explanation baseline, so
NSG=1 means the explanation lets the predictor recover every case the baseline missed,
and NSG<=0 means the explanation added nothing (or actively misled the predictor).

This module operationalises that on the synthetic dataset in `synthetic.py`. The
predictor never sees `true_feature` or `proxy_feature` directly as raw inputs -- only the
uninformative noise features -- which is what gives an explanation something real to add:
an explanation adds information exactly when it tells the predictor which feature to look
up and that feature's value turns out to move the label. See `_encode_explanations`.

Design choice that matters and is easy to get wrong: `simulatability` takes an ALREADY
FITTED predictor and only evaluates it. Fitting happens once, in `fit_predictor`, on data
drawn from the natural (correlated) distribution. Evaluating on the decorrelated subset or
on a targeted counterfactual is then a genuine out-of-distribution generalisation test of
what the predictor learned to trust. Fitting a fresh model directly on the decorrelated
subset (which an earlier version of this code did) is a bug, not a stricter test: a
flexible classifier trained and evaluated entirely within the decorrelated region simply
relearns whatever (locally inverted) pattern holds there and reports NSG=1 regardless of
whether the explanation is honest or a proxy, which defeats the entire experiment.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score

from .synthetic import LABEL_COL, NOISE_FEATURES, TRUE_FEATURE, decorrelated_subset, make_dataset

NAMEABLE_FEATURES: tuple[str, ...] = ("true_feature", "proxy_feature")

# Anchored on purpose, not a loose substring search: shuffle_explanation (controls.py)
# reorders words, and this pattern should stop matching once "based on X" is no longer a
# contiguous phrase. A substring search would not notice the difference and the shuffle
# control would test nothing.
_NAMED_FEATURE_RE = re.compile(r"\bbased on (\w+)")


def parse_named_feature(
    explanation: str | None,
    candidates: Sequence[str] = NAMEABLE_FEATURES,
) -> str | None:
    """Recover which feature, if any, an explanation claims determined the decision."""
    if not explanation:
        return None
    match = _NAMED_FEATURE_RE.search(explanation)
    if match is None:
        return None
    name = match.group(1).rstrip(".,")
    return name if name in candidates else None


def make_explanations(n: int, named_feature: str | None) -> list[str]:
    """Build the fixed-template explanation list used throughout the experiments.

    named_feature=None yields empty strings (no explanation given -- the no-explanation
    baseline condition). This is a plain template rather than per-row content by design;
    see controls.py's docstring for why that makes explanation_swap a no-op here while
    shuffle_explanation and length_matched_null still bite.
    """
    if named_feature is None:
        return [""] * n
    if named_feature not in NAMEABLE_FEATURES:
        raise ValueError(f"named_feature must be one of {NAMEABLE_FEATURES}, got {named_feature!r}")
    return [f"The decision was based on {named_feature}."] * n


def _encode_explanations(items: pd.DataFrame, explanations: Sequence[str | None] | None) -> np.ndarray:
    """Turn explanations into the extra columns a predictor sees: the value of whichever
    feature was named, plus a one-hot flag for which feature that was.

    All-zero columns when `explanations` is None: the no-explanation baseline gets
    literally zero extra information, which is what makes acc_without a genuine floor.
    """
    n = len(items)
    width = 1 + len(NAMEABLE_FEATURES)
    cols = np.zeros((n, width))
    if explanations is None:
        return cols
    # Group row indices by which feature their explanation names, then fill each group's
    # columns with one vectorised lookup. Avoids DataFrame.iterrows() (slow) since this
    # runs inside a 7 x 20-seed sweep in experiments/proxy_gaming.py and needs to stay
    # well under the script's own 60-second budget.
    idx_by_feature: dict[str, list[int]] = {}
    for i, explanation in enumerate(explanations):
        named = parse_named_feature(explanation)
        if named is not None and named in items.columns:
            idx_by_feature.setdefault(named, []).append(i)
    for feat, idxs in idx_by_feature.items():
        idx_arr = np.array(idxs, dtype=int)
        cols[idx_arr, 0] = items[feat].to_numpy(dtype=float)[idx_arr]
        cols[idx_arr, 1 + NAMEABLE_FEATURES.index(feat)] = 1.0
    return cols


def _design_matrix(
    items: pd.DataFrame,
    explanations: Sequence[str | None] | None,
    feature_names: Sequence[str],
) -> np.ndarray:
    x_surface = items[list(feature_names)].to_numpy(dtype=float)
    x_explan = _encode_explanations(items, explanations)
    return np.hstack([x_surface, x_explan])


def fit_predictor(
    predictor,
    train_items: pd.DataFrame,
    explanations: Sequence[str | None] | None,
    feature_names: Sequence[str] = NOISE_FEATURES,
    label_col: str = LABEL_COL,
):
    """Fit a fresh clone of `predictor` on `train_items`, optionally aided by
    `explanations`, and return the fitted clone.

    Always fit on data drawn from the natural distribution and reuse the resulting fitted
    model for every evaluation regime (see module docstring for why).
    """
    x = _design_matrix(train_items, explanations, feature_names)
    y = train_items[label_col].to_numpy()
    model = clone(predictor)
    if len(np.unique(y)) < 2:
        # Degenerate training label (possible only with a tiny --dry-run n): fall back to
        # a constant predictor rather than let sklearn raise on a single-class fit.
        only_class = int(y[0]) if len(y) else 0

        class _Constant:
            def predict(self, x_):
                return np.full(len(x_), only_class)

        return _Constant()
    model.fit(x, y)
    return model


def simulatability(
    predictor,
    items: pd.DataFrame,
    explanations: Sequence[str | None] | None,
    feature_names: Sequence[str] = NOISE_FEATURES,
    label_col: str = LABEL_COL,
) -> float:
    """Accuracy of the already-fitted `predictor` simulating the reference label on
    `items`, optionally aided by `explanations`. Does not fit anything -- see
    `fit_predictor` and the module docstring for why fitting and evaluating must use
    different data.
    """
    if len(items) == 0:
        raise ValueError("items is empty")
    x = _design_matrix(items, explanations, feature_names)
    y = items[label_col].to_numpy()
    preds = predictor.predict(x)
    return float(accuracy_score(y, preds))


def normalised_gain(acc_with: float, acc_without: float) -> float:
    """Normalised Simulatability Gain.

    NSG = (acc_with - acc_baseline) / (1 - acc_baseline)

    where `acc_baseline` is the no-explanation predictor accuracy -- `acc_without` here.
    It is the headroom-normalised improvement from seeing the explanation: NSG=1 means
    the explanation closes every gap the baseline left; NSG=0 means it added nothing;
    NSG<0 means it actively hurt prediction. Definition and normalisation from
    Mayne et al., https://arxiv.org/abs/2602.02639.
    """
    denom = 1.0 - acc_without
    if denom <= 1e-9:
        # Baseline is already (numerically) perfect: there is no headroom left for an
        # explanation to close, so the ratio is undefined. Report 0 rather than +-inf.
        return 0.0
    return (acc_with - acc_without) / denom


@dataclass
class InterventionProfile:
    """The three-number report this repo argues should replace a single average NSG.

    ordinary       -- NSG on naturally-distributed counterfactuals (true and proxy
                       features correlated as in training).
    decorrelated   -- NSG restricted to counterfactuals where that correlation is broken.
    targeted       -- NSG on counterfactuals that intervene on exactly the feature the
                       explanation claims matters, holding everything else fixed.

    Large gaps between these three say the explanation is describing a predictive
    correlate rather than a decision rule; all three close together says it identified
    something that actually drives the label under intervention.
    """

    ordinary: float
    decorrelated: float
    targeted: float


def intervention_profile(
    predictor,
    correlation: float,
    named_feature: str,
    n_train: int = 1000,
    n_eval: int = 3000,
    feature_names: Sequence[str] = NOISE_FEATURES,
    label_col: str = LABEL_COL,
    seed: int = 0,
) -> InterventionProfile:
    """Compute the ordinary, decorrelated, and targeted NSG for one explanation condition
    (named_feature="true_feature" for the honest explanation, "proxy_feature" for the
    proxy explanation).

    Fits a baseline (no-explanation) and an explained predictor once on a training draw
    at `correlation`, then evaluates both on a fresh, independent evaluation draw -- as a
    whole, as its decorrelated subset, and as its targeted (`named_feature`-only)
    counterfactuals. See the module docstring for why fitting and evaluating on the same
    subset would invalidate the decorrelated and targeted numbers.
    """
    train_df = make_dataset(n_train, correlation, seed=seed)
    eval_df = make_dataset(n_eval, correlation, seed=seed + 1_000_003)  # independent draw

    baseline_model = fit_predictor(predictor, train_df, None, feature_names, label_col)
    explained_model = fit_predictor(
        predictor, train_df, make_explanations(len(train_df), named_feature), feature_names, label_col
    )

    def _nsg(eval_items: pd.DataFrame) -> float:
        acc_without = simulatability(baseline_model, eval_items, None, feature_names, label_col)
        acc_with = simulatability(
            explained_model, eval_items, make_explanations(len(eval_items), named_feature), feature_names, label_col
        )
        return normalised_gain(acc_with, acc_without)

    ordinary = _nsg(eval_df)

    decor = decorrelated_subset(eval_df)
    decorrelated = _nsg(decor) if len(decor) >= 5 else float("nan")

    # Vectorised equivalent of calling synthetic.make_counterfactuals(row, hamming_distance=1,
    # feature_names=[named_feature]) on every row and concatenating the results -- flip exactly
    # named_feature, hold everything else fixed, and recompute the label from true_feature since
    # the reference rule is label = true_feature. Written directly (rather than looping the
    # per-row API, which is >100x slower here) because this repo's whole pitch is a sweep a
    # mentor can run in under a minute.
    targeted_df = eval_df.copy()
    targeted_df[named_feature] = 1 - targeted_df[named_feature]
    if TRUE_FEATURE in targeted_df.columns:
        targeted_df[label_col] = targeted_df[TRUE_FEATURE]
    targeted = _nsg(targeted_df)

    return InterventionProfile(ordinary=ordinary, decorrelated=decorrelated, targeted=targeted)
