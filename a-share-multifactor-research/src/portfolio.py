"""开盘到开盘、允许零碎模拟份额的现金/持仓账本。

目标金额按交易前 NAV 计算，先卖后买；资金不足时同比例缩减买单。
未成交单当日取消，冻结持仓保留，到下一次计划调仓重新评估。
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.dataset import validate_keys


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    holdings: pd.DataFrame
    trades: pd.DataFrame
    positions: pd.DataFrame
    events: pd.DataFrame


def build_execution_targets(scored: pd.DataFrame, top_n: int | None, rebalance_days: int,
                            signal_start: pd.Timestamp | None = None) -> tuple[dict, pd.DataFrame]:
    if rebalance_days < 1 or (top_n is not None and top_n < 1):
        raise ValueError('Invalid portfolio parameters')
    validate_keys(scored, ['date', 'code'], 'target panel')
    dates = pd.DatetimeIndex(sorted(scored.date.unique()))
    by_date = {d: g for d, g in scored.groupby('date')}
    need = top_n or 1
    if signal_start is None:
        valid_dates = [d for d, g in by_date.items() if len(g.loc[g.signal_eligible & g.score.notna()]) >= need]
        if not valid_dates:
            return {}, pd.DataFrame(columns=['date', 'event', 'code', 'detail'])
        signal_start = min(valid_dates)
    start_pos = dates.searchsorted(pd.Timestamp(signal_start))
    targets, events = {}, []
    # 按完整市场交易日每隔 K 日调仓，不把“有评分的日期”压缩成另一个日历。
    for pos in range(start_pos, len(dates) - 1, rebalance_days):
        signal_date, execution_date = dates[pos], dates[pos + 1]
        day = by_date[signal_date]
        candidates = day.loc[day.signal_eligible & day.score.notna()].sort_values(['score', 'code'], ascending=[False, True])
        if len(candidates) < need:
            events.append({'date': execution_date, 'event': 'no_signal', 'code': '', 'detail': f'{len(candidates)} eligible stocks on {signal_date.date()}'})
            continue
        selected = candidates if top_n is None else candidates.head(top_n)
        targets[execution_date] = {'signal_date': signal_date, 'weights': {c: 1 / len(selected) for c in selected.code}}
    return targets, pd.DataFrame(events, columns=['date', 'event', 'code', 'detail'])


def run_open_to_open_backtest(scored: pd.DataFrame, benchmark: pd.DataFrame, top_n: int | None,
                              rebalance_days: int, one_way_cost: float,
                              signal_start: pd.Timestamp | None = None) -> BacktestResult:
    if not 0 <= one_way_cost < 1:
        raise ValueError('one_way_cost must be in [0, 1)')
    targets, scheduling = build_execution_targets(scored, top_n, rebalance_days, signal_start)
    if not targets:
        raise ValueError('No executable targets; need a longer history or more eligible stocks')
    dates = pd.DatetimeIndex(sorted(scored.date.unique()))
    # 显式共同信号起点时，即使某个对照第一天股票不足，也从共同执行日起记现金。
    # 否则 Top-N 敏感性可能悄悄跳过一段行情，造成不同区间的比较。
    first = dates[dates.searchsorted(pd.Timestamp(signal_start)) + 1] if signal_start is not None else min(targets)
    bench = benchmark.set_index('date').reindex(dates)['open']
    if bench.loc[first:].isna().any() or bench.loc[first:].le(0).any():
        raise ValueError('Benchmark must cover every execution/valuation date')
    cash, previous_nav = 1.0, 1.0
    shares: dict[str, float] = {}
    records, target_records, trades, positions = [], [], [], []
    events = scheduling.to_dict('records')
    epsilon = 1e-12
    for date, group in scored.groupby('date', sort=True):
        if date < first:
            continue
        row = group.set_index('code')
        marks = row['mark_open'].to_dict()
        for code, qty in shares.items():
            if qty > epsilon and (code not in marks or not np.isfinite(marks[code]) or marks[code] <= 0):
                raise ValueError(f'Cannot value held stock {code} on {date}; never drop its asset value')
        before = cash + sum(qty * marks[code] for code, qty in shares.items())
        gross, fee_total = 0.0, 0.0
        signal = pd.NaT
        if date in targets:
            signal = targets[date]['signal_date']
            weights = targets[date]['weights']
            for code, weight in weights.items():
                target_records.append({'signal_date': signal, 'execution_date': date, 'code': code, 'target_weight': weight})
            # 卖出超额/淘汰持仓。不能卖的仓位继续占用资产和资金预算。
            for code in sorted(list(shares)):
                current_value = shares[code] * marks[code]
                desired = weights.get(code, 0.0) * before
                sell_value = max(current_value - desired, 0.0)
                if sell_value <= epsilon:
                    continue
                if not bool(row.at[code, 'can_sell_open']):
                    events.append({'date': date, 'event': 'sell_blocked', 'code': code, 'detail': 'No tradable open / conservative lower-limit guard; position retained'})
                    continue
                price = float(row.at[code, 'open'])
                qty = min(shares[code], sell_value / price)
                notional = qty * price
                fee = notional * one_way_cost
                shares[code] -= qty
                cash += notional - fee
                gross += notional
                fee_total += fee
                trades.append({'date': date, 'signal_date': signal, 'code': code, 'side': 'sell', 'quantity': qty, 'price': price, 'notional': notional, 'fee': fee})
            shares = {c: q for c, q in shares.items() if q > epsilon}
            buy_requests = {}
            for code, weight in sorted(weights.items()):
                held_value = shares.get(code, 0.0) * marks.get(code, 0.0) if code in shares else 0.0
                desired_buy = max(weight * before - held_value, 0.0)
                if desired_buy <= epsilon:
                    continue
                if code not in row.index or not bool(row.at[code, 'can_buy_open']):
                    events.append({'date': date, 'event': 'buy_blocked', 'code': code, 'detail': 'No tradable open / conservative upper-limit guard; cash retained'})
                    continue
                buy_requests[code] = desired_buy
            requested = sum(buy_requests.values())
            scale = min(1.0, max(cash, 0.0) / ((1 + one_way_cost) * requested)) if requested > 0 else 0.0
            for code, request in buy_requests.items():
                notional = request * scale
                if notional <= epsilon:
                    continue
                price = float(row.at[code, 'open'])
                qty, fee = notional / price, notional * one_way_cost
                shares[code] = shares.get(code, 0.0) + qty
                cash -= notional + fee
                gross += notional
                fee_total += fee
                trades.append({'date': date, 'signal_date': signal, 'code': code, 'side': 'buy', 'quantity': qty, 'price': price, 'notional': notional, 'fee': fee})
        if abs(cash) < epsilon:
            cash = 0.0
        stock_value = sum(qty * marks[code] for code, qty in shares.items())
        nav = cash + stock_value
        if cash < -epsilon or any(q < -epsilon for q in shares.values()) or nav <= 0:
            raise AssertionError('Negative cash/position or bankrupt ledger')
        if not np.isclose(nav, before - fee_total, rtol=1e-10, atol=1e-12):
            raise AssertionError('Asset conservation failed during rebalance')
        for code, qty in sorted(shares.items()):
            value = qty * marks[code]
            positions.append({'date': date, 'code': code, 'quantity': qty, 'mark_open': marks[code], 'market_value': value,
                              'actual_weight': value / nav, 'stale_mark': not bool(row.at[code, 'can_buy_open'] or row.at[code, 'can_sell_open'])})
        records.append({'date': date, 'portfolio_nav': nav, 'portfolio_return': nav / previous_nav - 1,
                        'cash': cash, 'stock_value': stock_value, 'cash_weight': cash / nav,
                        'turnover': gross / before, 'transaction_cost': fee_total, 'pretrade_nav': before,
                        'n_positions': len(shares)})
        previous_nav = nav
    daily = pd.DataFrame(records)
    index_open = bench.reindex(pd.DatetimeIndex(daily.date)).to_numpy()
    daily['benchmark_nav'] = index_open / index_open[0]
    daily['benchmark_return'] = daily.benchmark_nav.pct_change(fill_method=None).fillna(0.0)
    return BacktestResult(
        daily, pd.DataFrame(target_records, columns=['signal_date', 'execution_date', 'code', 'target_weight']),
        pd.DataFrame(trades, columns=['date', 'signal_date', 'code', 'side', 'quantity', 'price', 'notional', 'fee']),
        pd.DataFrame(positions, columns=['date', 'code', 'quantity', 'mark_open', 'market_value', 'actual_weight', 'stale_mark']),
        pd.DataFrame(events, columns=['date', 'event', 'code', 'detail']),
    )
