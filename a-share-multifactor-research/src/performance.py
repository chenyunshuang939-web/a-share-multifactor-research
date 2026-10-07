"""统一起点及收益序列口径，首次建仓费用计入第一条收益。"""
from __future__ import annotations
import numpy as np
import pandas as pd


def max_drawdown(nav: pd.Series, initial_nav: float = 1.0) -> float:
    values = pd.concat([pd.Series([initial_nav]), nav.reset_index(drop=True)]).dropna()
    return float((values / values.cummax() - 1).min())


def performance_metrics(returns: pd.Series, annual_days: int = 252) -> dict:
    r = pd.Series(returns, dtype=float).dropna()
    if len(r) == 0:
        return {}
    if not np.isfinite(r).all() or r.le(-1).any():
        raise ValueError('Invalid return series')
    nav = (1 + r).cumprod()
    total = nav.iloc[-1] - 1
    # 所有策略相同的每日账本观察次数；首日包含执行费用，指数首日为零。
    annual = nav.iloc[-1] ** (annual_days / len(r)) - 1
    std = r.std(ddof=1)
    vol = std * np.sqrt(annual_days)
    sharpe = r.mean() / std * np.sqrt(annual_days) if std > 0 else np.nan
    mdd = max_drawdown(nav)
    return {'n_obs': len(r), 'total_return': total, 'annual_return': annual, 'annual_volatility': vol,
            'sharpe_rf0': sharpe, 'max_drawdown': mdd, 'calmar': annual / abs(mdd) if mdd < 0 else np.nan,
            'positive_day_ratio': (r > 0).mean()}


def comparison_table(daily: pd.DataFrame, annual_days: int = 252) -> pd.DataFrame:
    rows = []
    for col in [c for c in daily if c.endswith('_return')]:
        r = daily[col]
        if r.isna().any():
            raise ValueError('All compared strategies must use the same nonmissing dates')
        row = {'series': col.removesuffix('_return'), 'start_date': daily.date.min(), 'end_date': daily.date.max(), **performance_metrics(r, annual_days)}
        if col != 'benchmark_return':
            excess = r - daily.benchmark_return
            te = excess.std(ddof=1) * np.sqrt(annual_days)
            row['tracking_error'] = te
            row['information_ratio'] = excess.mean() * annual_days / te if te > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def period_comparison(daily: pd.DataFrame, train_end: str, validation_end: str) -> pd.DataFrame:
    periods = [('development', daily.date.le(train_end)),
               ('validation', daily.date.gt(train_end) & daily.date.le(validation_end)),
               ('retrospective_test', daily.date.gt(validation_end))]
    frames = []
    for label, mask in periods:
        if mask.sum() >= 2:
            part = comparison_table(daily.loc[mask])
            part.insert(0, 'period', label)
            frames.append(part)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=['period', 'series'])
