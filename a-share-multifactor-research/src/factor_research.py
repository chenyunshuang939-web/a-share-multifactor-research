"""原始 IC 按信号观测日保存；可用 IC 按标签成熟日对齐，两者不能混用。"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm


def add_forward_return(panel: pd.DataFrame, horizon: int, kind: str = 'open') -> pd.DataFrame:
    if horizon < 1 or kind not in {'open', 'close'}:
        raise ValueError('horizon >= 1 and kind in {open, close} required')
    df = panel.sort_values(['code', 'date']).copy()
    calendar = pd.DatetimeIndex(sorted(df.date.unique()))
    for _, g in df.groupby('code'):
        if not pd.DatetimeIndex(g.date).equals(calendar):
            raise ValueError('Labels require a complete, identical trading calendar for every code')
    g = df.groupby('code')
    if kind == 'open':
        entry, exit_ = g.open.shift(-1), g.open.shift(-(horizon + 1))
        # 样本外未来的可成交状态只决定事后标签，只有成熟后才允许进入权重。
        valid = g.can_buy_open.shift(-1).eq(True) & g.can_sell_open.shift(-(horizon + 1)).eq(True)
        lag = horizon + 1
    else:
        entry, exit_ = df.research_close, g.research_close.shift(-horizon)
        valid = entry.gt(0) & exit_.gt(0)
        lag = horizon
    df[f'fwd_{kind}_ret_{horizon}'] = (exit_ / entry - 1).where(valid & entry.gt(0) & exit_.gt(0))
    df[f'{kind}_label_maturity'] = g.date.shift(-lag)
    return df


def rank_ic(panel: pd.DataFrame, z_factors: list[str], target: str, min_cs: int) -> pd.DataFrame:
    records = []
    for date, cross in panel.groupby('date', sort=True):
        rec = {'date': date}
        for factor in z_factors:
            sample = cross[[factor, target]].dropna()
            if len(sample) < min_cs or sample[factor].nunique() < 2 or sample[target].nunique() < 2:
                rec[factor] = np.nan
            else:
                # Spearman = Pearson(cross-sectional average ranks)，并列值不按股票顺序打破。
                rec[factor] = sample[factor].rank().corr(sample[target].rank())
        records.append(rec)  # 即使整行缺失，也保留日期，保证 shift 是市场交易日。
    return pd.DataFrame(records, columns=['date'] + z_factors)


def available_ic(raw_ic: pd.DataFrame, calendar: pd.DatetimeIndex, delay: int) -> pd.DataFrame:
    if delay < 1:
        raise ValueError('IC delay must be positive')
    data = raw_ic.set_index('date').reindex(calendar)
    data.index.name = 'date'
    return data.shift(delay).reset_index()


def dynamic_ic_weights(ic: pd.DataFrame, lookback: int, min_periods: int, clip: float = 3.0) -> pd.DataFrame:
    """输入必须是 available_ic；不再额外 shift，避免重复滞后。"""
    if not 2 <= min_periods <= lookback or clip <= 0:
        raise ValueError('Invalid IC rolling parameters')
    data = ic.set_index('date')
    mean = data.rolling(lookback, min_periods=min_periods).mean()
    std = data.rolling(lookback, min_periods=min_periods).std(ddof=1)
    raw = (mean / std.where(std.gt(1e-12))).clip(-clip, clip)
    # 所有因子完成最低样本预热后才启动组合，避免早期组合定义悄悄变化。
    valid = raw.notna().all(axis=1)
    weights = raw.div(raw.abs().sum(axis=1).replace(0, np.nan), axis=0)
    weights.loc[~valid] = np.nan
    return weights.reset_index()


def attach_composite_score(panel: pd.DataFrame, weights: pd.DataFrame, z_factors: list[str]) -> pd.DataFrame:
    df = panel.copy()
    matrix = df[z_factors].to_numpy(float)
    w = weights.set_index('date').reindex(pd.DatetimeIndex(df.date))[z_factors].to_numpy(float)
    valid = np.isfinite(matrix).all(axis=1) & np.isfinite(w).all(axis=1) & df.signal_eligible.to_numpy(bool)
    score = np.full(len(df), np.nan)
    score[valid] = (matrix[valid] * w[valid]).sum(axis=1)
    df['score'] = score
    return df


def static_scores(panel: pd.DataFrame, directions: dict[str, int], factors: list[str], eligibility_factors: list[str]) -> pd.DataFrame:
    df = panel.copy()
    cols = ['z_' + f for f in factors]
    values = df[cols].mul(pd.Series({'z_' + f: directions[f] for f in factors}), axis=1)
    valid = df[['z_' + f for f in eligibility_factors]].notna().all(axis=1) & df.signal_eligible
    df['score'] = values.mean(axis=1).where(valid)
    return df


def hac_mean_stats(values: pd.Series, lag: int) -> dict:
    """Newey-West/Bartlett 均值标准误；正态近似 p 值，未校正多重检验。"""
    x = values.dropna().to_numpy(float)
    n = len(x)
    if n < 3:
        return {'n_obs': n, 'mean': float(np.mean(x)) if n else np.nan, 'hac_t': np.nan, 'hac_p': np.nan}
    y = x - x.mean()
    lag = min(max(0, lag), n - 1)
    long_var = np.dot(y, y) / n
    for k in range(1, lag + 1):
        long_var += 2 * (1 - k / (lag + 1)) * np.dot(y[k:], y[:-k]) / n
    se = np.sqrt(max(long_var, 0) / n)
    t = x.mean() / se if se > 0 else np.nan
    return {'n_obs': n, 'mean': float(x.mean()), 'hac_t': t, 'hac_p': 2 * norm.sf(abs(t))}


def summarize_ic(ic: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    records = []
    for col in ic.columns.drop('date'):
        s = ic[col].dropna()
        stats = hac_mean_stats(s, horizon)
        std = s.std(ddof=1)
        records.append({'factor': col.removeprefix('z_'), 'mean_ic': stats.pop('mean'),
                        'std_ic': std, 'icir_annualized_descriptive': s.mean() / std * np.sqrt(252 / horizon) if std > 0 else np.nan,
                        'positive_ic_ratio': (s > 0).mean() if len(s) else np.nan, **stats})
    return pd.DataFrame(records).sort_values('mean_ic', ascending=False)


def group_test(panel: pd.DataFrame, z_factors: list[str], target: str, n_groups: int, min_cs: int) -> pd.DataFrame:
    records = []
    for date, cross in panel.groupby('date'):
        for factor in z_factors:
            sample = cross[[factor, target]].dropna().copy()
            if len(sample) < max(min_cs, n_groups) or sample[factor].nunique() < n_groups:
                continue
            groups = pd.qcut(sample[factor], q=n_groups, labels=False, duplicates='drop')
            if groups.nunique() != n_groups:
                continue  # 不足五个真实分位组时跳过，不能强行按代码拆并列值。
            sample['group'] = groups + 1
            for group, rows in sample.groupby('group'):
                records.append({'date': date, 'factor': factor.removeprefix('z_'), 'group': int(group),
                                'mean_forward_return': rows[target].mean(), 'n_stocks': len(rows)})
    return pd.DataFrame(records, columns=['date', 'factor', 'group', 'mean_forward_return', 'n_stocks'])


def summarize_groups(groups: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if groups.empty:
        return pd.DataFrame(columns=['factor', 'group', 'mean_forward_return']), pd.DataFrame(columns=['factor', 'mean_spread', 'hac_t', 'hac_p', 'n_obs'])
    summary = groups.groupby(['factor', 'group'], as_index=False).agg(mean_forward_return=('mean_forward_return', 'mean'), n_dates=('date', 'nunique'), mean_group_size=('n_stocks', 'mean'))
    records = []
    for factor, g in groups.groupby('factor'):
        pivot = g.pivot(index='date', columns='group', values='mean_forward_return')
        spread = pivot[pivot.columns.max()] - pivot[pivot.columns.min()]
        stats = hac_mean_stats(spread, horizon)
        records.append({'factor': factor, 'mean_spread': stats.pop('mean'), **stats})
    return summary, pd.DataFrame(records)


def mean_cross_sectional_corr(panel: pd.DataFrame, z_factors: list[str], min_cs: int) -> pd.DataFrame:
    matrices = [g[z_factors].dropna().corr(method='spearman').to_numpy() for _, g in panel.groupby('date') if len(g[z_factors].dropna()) >= min_cs]
    if not matrices:
        return pd.DataFrame(np.nan, index=z_factors, columns=z_factors)
    return pd.DataFrame(np.nanmean(matrices, axis=0), index=z_factors, columns=z_factors)
