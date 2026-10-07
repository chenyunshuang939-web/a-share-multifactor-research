"""正损失口径 VaR/ES；历史尾部估计与事前滚动预测分开。"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.special import xlogy


def _confidence(confidence: float) -> None:
    if not 0 < confidence < 1:
        raise ValueError('confidence must be in (0, 1)')


def historical_var_es(returns: pd.Series, confidence: float) -> tuple[float, float]:
    _confidence(confidence)
    r = pd.Series(returns, dtype=float).dropna()
    if r.empty:
        return np.nan, np.nan
    if not np.isfinite(r).all():
        raise ValueError('Nonfinite returns')
    q = r.quantile(1 - confidence)
    return -float(q), -float(r[r <= q].mean())


def rolling_historical_var(returns: pd.Series, confidence: float, window: int = 250, min_periods: int = 100) -> pd.Series:
    _confidence(confidence)
    if not 1 <= min_periods <= window:
        raise ValueError('Invalid risk window')
    # t 行仅使用截至 t-1 的实现收益；改变 r_t 不得改变这一行阈值。
    return -returns.rolling(window, min_periods=min_periods).quantile(1 - confidence).shift(1)


def kupiec_pof_test(returns: pd.Series, var_series: pd.Series, confidence: float) -> dict:
    _confidence(confidence)
    frame = pd.concat([returns.rename('ret'), var_series.rename('var')], axis=1).dropna()
    n = len(frame)
    if not n:
        return {'n': 0, 'exceptions': 0, 'exception_rate': np.nan, 'lr_pof': np.nan, 'p_value': np.nan}
    count = int(frame.ret.lt(-frame['var']).sum())
    p, observed = 1 - confidence, count / n
    # xlogy(0, 0)=0，正确处理零突破/全部突破，不需人为截断概率。
    null_ll = xlogy(count, p) + xlogy(n - count, 1 - p)
    fitted_ll = xlogy(count, observed) + xlogy(n - count, 1 - observed)
    lr = max(0.0, float(2 * (fitted_ll - null_ll)))
    return {'n': n, 'exceptions': count, 'exception_rate': observed, 'lr_pof': lr, 'p_value': float(chi2.sf(lr, 1))}


def risk_report(returns: pd.Series, window: int = 250, min_periods: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, forecasts = [], pd.DataFrame({'portfolio_return': returns})
    for cl in [.95, .99]:
        recent = returns.dropna().tail(window)
        var, es = historical_var_es(recent, cl)
        predicted = rolling_historical_var(returns, cl, window, min_periods)
        key = str(int(cl * 100))
        forecasts[f'var_{key}'] = predicted
        forecasts[f'exception_{key}'] = returns.lt(-predicted).astype(float).where(predicted.notna())
        rows.append({'confidence': cl, 'recent_observations': len(recent), 'recent_historical_var': var,
                     'recent_historical_es': es, 'tail_observations': int(recent.le(-var).sum()) if len(recent) else 0,
                     'next_period_var_forecast': var if len(recent) >= min_periods else np.nan,
                     **kupiec_pof_test(returns, predicted, cl)})
    return pd.DataFrame(rows), forecasts


def worst_days(daily: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return daily.nsmallest(n, 'portfolio_return')[['date', 'portfolio_return', 'portfolio_nav']].reset_index(drop=True)
