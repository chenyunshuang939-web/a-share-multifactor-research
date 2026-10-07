from __future__ import annotations

import numpy as np
import pandas as pd


def _as_float_series(series: pd.Series) -> pd.Series:
    """Convert a cross-sectional factor slice to float while preserving its index."""
    return pd.to_numeric(series, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)

    #解决的是“异常离群值破坏统计稳定性”的问题
def mad_winsorize(series: pd.Series, n: float = 3.0) -> pd.Series:
    """MAD winsorization that safely preserves missing factor observations.

    The early part of rolling factors can be entirely NaN.  Pandas/NumPy may
    emit ``Mean of empty slice`` warnings when summary statistics are applied
    to such a group.  We therefore compute robust statistics only on the
    non-missing observations and write the clipped values back to their
    original positions.
    """
    x = _as_float_series(series)
    valid = x.dropna()

    # No factor observation exists for this date: keep the whole cross-section
    # missing.  Do not manufacture a zero factor exposure.
    if valid.empty:
        return x

    median = valid.median()
    mad = (valid - median).abs().median()

    # If every valid observation is the same, MAD is zero and there is no
    # outlier to clip. Missing observations remain missing.
    if not np.isfinite(mad) or mad == 0:
        return x

    scale = 1.4826 * mad
    lower = median - n * scale
    upper = median + n * scale

    out = x.copy()
    out.loc[valid.index] = valid.clip(lower, upper)
    return out

    #Z-Score 解决的是“不同量纲无法横向对比和多因子加权”的问题
def zscore(series: pd.Series) -> pd.Series:
    """Cross-sectional Z-score with explicit missing-data semantics.

    * all-NaN group -> all NaN;
    * constant valid values -> valid observations become 0, NaNs stay NaN;
    * otherwise standardize only valid observations.

    This distinction matters in factor research: NaN means "factor unavailable",
    whereas 0 means "neutral exposure". Treating the former as the latter would
    silently contaminate Rank IC and composite-score calculations.
    """
    x = _as_float_series(series)
    valid = x.dropna()
    out = pd.Series(np.nan, index=x.index, dtype=float)

    if valid.empty:
        return out

    std = valid.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        out.loc[valid.index] = 0.0
        return out

    mean = valid.mean()
    out.loc[valid.index] = (valid - mean) / std
    return out


def preprocess_cross_section(panel: pd.DataFrame, factors: list[str], mad_n: float) -> pd.DataFrame:
    """Winsorize and standardize each factor independently on each trading day."""
    df = panel.copy()
    for factor in factors:
        clipped = df.groupby("date", sort=False)[factor].transform(
            lambda s: mad_winsorize(s, mad_n)
        )
        df[f"z_{factor}"] = clipped.groupby(df["date"], sort=False).transform(zscore)
    return df
