"""11 个价量因子；输入必须已按完整市场日历对齐。"""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.dataset import validate_keys


def _rolling(grouped, window: int, func: str) -> pd.Series:
    return getattr(grouped.rolling(window, min_periods=window), func)().reset_index(level=0, drop=True)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """简单移动平均版 RSI（不是 Wilder 平滑版）；预热不足保留 NaN。"""
    change = close.diff()
    gain = change.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-change.clip(upper=0)).rolling(window, min_periods=window).mean()
    result = 100 - 100 / (1 + gain / loss.where(loss.ne(0)))
    result = result.mask(loss.eq(0) & gain.gt(0), 100.0)
    result = result.mask(loss.gt(0) & gain.eq(0), 0.0)
    result = result.mask(loss.eq(0) & gain.eq(0), 50.0)
    return result


def build_factors(panel: pd.DataFrame) -> pd.DataFrame:
    validate_keys(panel, ['date', 'code'], 'factors')
    df = panel.sort_values(['code', 'date']).copy()
    g = df.groupby('code', group_keys=False)
    close = 'research_close'
    for col, days in [('ret1', 1), ('mom20', 20), ('mom60', 60)]:
        df[col] = g[close].pct_change(days, fill_method=None)
    df['rev5'] = -g[close].pct_change(5, fill_method=None)
    for window in [20, 60]:
        df[f'vol{window}'] = _rolling(df.groupby('code')['ret1'], window, 'std') * np.sqrt(252)
    df['turn20'] = _rolling(df.groupby('code')['turn'], 20, 'mean')
    vol_ma = _rolling(df.groupby('code')['volume'], 20, 'mean')
    df['volratio20'] = df.volume / vol_ma.where(vol_ma.gt(0)) - 1
    df['_illiq'] = df.ret1.abs() / df.amount.where(df.amount.gt(0)) * 1e8
    df['amihud20'] = _rolling(df.groupby('code')['_illiq'], 20, 'mean')
    df['rsi14'] = g[close].transform(_rsi)
    ma20 = _rolling(g[close], 20, 'mean')
    ma60 = _rolling(g[close], 60, 'mean')
    std20 = _rolling(g[close], 20, 'std')
    df['bollpos20'] = (df[close] - ma20) / (4 * std20.where(std20.gt(0)))
    df['trend2060'] = ma20 / ma60.where(ma60.gt(0)) - 1
    return df.drop(columns='_illiq')
