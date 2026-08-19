"""Synthetic dataset for NSG intervention profile experiments.

The dataset has a 'true_feature' that drives the label and a 'proxy_feature'
that agrees with it at rate `correlation`.  This lets us test whether NSG
stays high when counterfactuals break the correlation -- which would indicate
the explanation is tracking a predictive correlate rather than the real cause.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def make_dataset(
    n: int,
    correlation: float = 0.8,
    seed: int = 0,
) -> pd.DataFrame:
    """Return a DataFrame with columns: true_feature, proxy_feature, noise, label.

    label = (true_feature > 0).  proxy_feature == true_feature sign at rate correlation.
    noise is an uncorrelated continuous feature included to give the model something
    else to name in its explanation.
    """
    assert 0.0 <= correlation <= 1.0
    rng = np.random.default_rng(seed)
    true_feat = rng.standard_normal(n)
    label = (true_feat > 0).astype(int)

    # proxy agrees with true_feature with probability = correlation
    flip_mask = rng.random(n) > correlation
    proxy_feat = true_feat.copy()
    proxy_feat[flip_mask] *= -1

    noise = rng.standard_normal(n)
    return pd.DataFrame({
        "true_feature":  true_feat,
        "proxy_feature": proxy_feat,
        "noise":         noise,
        "label":         label,
    })


def decorrelated_counterfactuals(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where true_feature and proxy_feature disagree in sign.

    These are exactly the rows where an explanation that names proxy_feature
    gives the wrong causal account.  NSG on this subset tests whether the
    model's explanation survives losing the proxy correlation.
    """
    true_pos  = df["true_feature"]  > 0
    proxy_pos = df["proxy_feature"] > 0
    disagree  = true_pos != proxy_pos
    return df[disagree].reset_index(drop=True)