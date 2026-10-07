"""Regression tests for causal timing, missing data, and cash/position conservation."""
from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset import align_panel, sha256, load_raw_snapshot
from src.factors import _rsi, build_factors
from src.preprocess import zscore, mad_winsorize, preprocess_cross_section
from src.factor_research import (add_forward_return, rank_ic, available_ic, dynamic_ic_weights, attach_composite_score, group_test)
from src.portfolio import run_open_to_open_backtest, build_execution_targets
from src.performance import performance_metrics, max_drawdown, comparison_table
from src.risk import historical_var_es, rolling_historical_var, kupiec_pof_test
from scripts.download_data import cache_usable, choose_universe
from scripts.make_demo_data import make_demo
from config import FACTOR_COLUMNS


def raw_fixture(periods=150, n=12):
    rng = np.random.default_rng(12)
    dates = pd.bdate_range('2022-01-03', periods=periods)
    records = []
    for i in range(n):
        close = (20 + i) * np.exp(np.cumsum(rng.normal(.0002, .015, periods)))
        prev = np.r_[close[0], close[:-1]]
        open_ = prev * np.exp(rng.normal(0, .004, periods))
        records.append(pd.DataFrame({'date': dates, 'code': f'sh.{600000+i:06d}', 'open': open_, 'close': close,
            'preclose': prev, 'volume': rng.uniform(1e5, 1e6, periods), 'amount': rng.uniform(1e7, 1e8, periods),
            'turn': rng.uniform(.1, 2, periods), 'tradestatus': 1, 'isST': 0}))
    b = pd.DataFrame({'date': dates, 'open': 100., 'close': 100.})
    return pd.concat(records, ignore_index=True), b


def score_fixture(periods=5):
    dates = pd.bdate_range('2022-01-03', periods=periods)
    raw = pd.DataFrame([{'date': d, 'code': c, 'open': 10., 'close': 10., 'preclose': 10.,
                         'tradestatus': 1, 'isST': 0} for d in dates for c in ['A', 'B']])
    bench = pd.DataFrame({'date': dates, 'open': 100., 'close': 100.})
    panel, _ = align_panel(raw, bench, block_limit_open=False)
    panel['score'] = np.where(panel.code.eq('A'), 2., 1.)
    return panel, bench


class PreprocessTests(unittest.TestCase):
    def test_nan_is_not_zero(self):
        self.assertTrue(zscore(pd.Series([np.nan, np.nan])).isna().all())
    def test_constant_and_missing(self):
        out = zscore(pd.Series([1., 1., np.nan]))
        self.assertEqual(out.iloc[0], 0.)
        self.assertTrue(pd.isna(out.iloc[2]))
    def test_infinity_is_missing(self):
        self.assertTrue(pd.isna(zscore(pd.Series([1., 2., np.inf])).iloc[-1]))
    def test_outlier(self):
        self.assertLess(mad_winsorize(pd.Series([1., 1.1, .9, 1.05, 100.])).max(), 100.)
    def test_rsi_rising(self):
        self.assertEqual(_rsi(pd.Series(np.arange(30.))).iloc[-1], 100.)
    def test_rsi_falling(self):
        self.assertEqual(_rsi(pd.Series(np.arange(30., 0., -1))).iloc[-1], 0.)
    def test_rsi_flat(self):
        self.assertEqual(_rsi(pd.Series(np.ones(30))).iloc[-1], 50.)
    def test_rsi_warmup_missing(self):
        self.assertTrue(_rsi(pd.Series(np.arange(30.))).iloc[:14].isna().all())


class CalendarAndCausalityTests(unittest.TestCase):
    def test_gap_does_not_extend_horizon(self):
        raw, b = raw_fixture(15, 2)
        code = raw.code.iloc[0]
        raw = raw.loc[~(raw.code.eq(code) & raw.date.eq(b.date.iloc[3]))]
        panel, _ = align_panel(raw, b, block_limit_open=False)
        labeled = add_forward_return(panel, 5, 'open')
        row = labeled.loc[labeled.code.eq(code) & labeled.date.eq(b.date.iloc[0])].iloc[0]
        self.assertEqual(row.open_label_maturity, b.date.iloc[6])
        expected = raw.loc[raw.code.eq(code) & raw.date.eq(b.date.iloc[6]), 'open'].iloc[0] / raw.loc[raw.code.eq(code) & raw.date.eq(b.date.iloc[1]), 'open'].iloc[0] - 1
        self.assertAlmostEqual(row.fwd_open_ret_5, expected)
        self.assertEqual(len(panel), 30)
    def test_nontradable_label_entry_is_missing(self):
        p, _ = score_fixture(10)
        p.loc[p.date.eq(sorted(p.date.unique())[1]), 'can_buy_open'] = False
        out = add_forward_return(p, 5, 'open')
        self.assertTrue(out.loc[out.date.eq(p.date.min()), 'fwd_open_ret_5'].isna().all())
    def test_close_label_delay(self):
        p, _ = score_fixture(10)
        out = add_forward_return(p, 5, 'close')
        self.assertEqual(out.iloc[0].close_label_maturity, sorted(p.date.unique())[5])
    def test_empty_ic_keeps_calendar(self):
        p, _ = score_fixture(4)
        p['z_x'], p['target'] = 0., np.nan
        ic = rank_ic(p, ['z_x'], 'target', 10)
        self.assertEqual(len(ic), 4)
        self.assertTrue(ic.z_x.isna().all())
    def test_missing_ic_date_not_compressed(self):
        dates = pd.bdate_range('2022-01-03', periods=8)
        ic = pd.DataFrame({'date': dates.delete(2), 'z_x': np.arange(7.)})
        out = available_ic(ic, dates, 2).set_index('date')
        self.assertTrue(pd.isna(out.loc[dates[4], 'z_x']))
        self.assertEqual(out.loc[dates[2], 'z_x'], 0.)
    def test_complete_calendar_required(self):
        p, _ = score_fixture(10)
        with self.assertRaises(ValueError):
            add_forward_return(p.drop(index=p.index[2]), 5)
    def test_open_constraints_do_not_read_same_day_close(self):
        raw, b = raw_fixture(10, 2)
        p, _ = align_panel(raw, b)
        modified = raw.copy()
        modified.loc[modified.date.eq(b.date.iloc[4]), 'close'] *= 2
        q, _ = align_panel(modified, b)
        assert_frame_equal(p[['can_buy_open', 'can_sell_open']], q[['can_buy_open', 'can_sell_open']])
    def test_prefix_invariance_and_latest_score(self):
        raw, b = raw_fixture(145, 12)
        cutoff = b.date.iloc[120]
        raw = raw.loc[~(raw.code.eq(raw.code.iloc[0]) & raw.date.isin(b.date.iloc[115:120]))]
        def calculate(r, bench):
            p, _ = align_panel(r, bench, block_limit_open=False)
            p = build_factors(p)
            fs = FACTOR_COLUMNS
            p.loc[~p.signal_eligible, fs] = np.nan
            p = preprocess_cross_section(p, fs, 3)
            p = add_forward_return(p, 5)
            z = ['z_' + f for f in fs]
            ic = rank_ic(p, z, 'fwd_open_ret_5', 10)
            matured = available_ic(ic, pd.DatetimeIndex(bench.date), 6)
            w = dynamic_ic_weights(matured, 30, 10)
            scores = attach_composite_score(p, w, z)
            return matured, w, scores[['date', 'code', 'score']]
        full = calculate(raw, b)
        prefix = calculate(raw.loc[raw.date.le(cutoff)], b.loc[b.date.le(cutoff)])
        for left, right in zip(full, prefix):
            assert_frame_equal(left.loc[left.date.le(cutoff)].reset_index(drop=True), right.reset_index(drop=True), check_exact=False, atol=1e-12, rtol=1e-12)
        self.assertGreater(prefix[2].loc[prefix[2].date.eq(cutoff), 'score'].notna().sum(), 0)
        changed = raw.copy()
        changed.loc[changed.date.gt(cutoff), ['open', 'close']] *= 10
        mutated = calculate(changed, b)
        for a, c in zip(full, mutated):
            assert_frame_equal(a.loc[a.date.le(cutoff)].reset_index(drop=True), c.loc[c.date.le(cutoff)].reset_index(drop=True))
    def test_constant_factor_not_grouped(self):
        p, _ = raw_fixture(3, 12)
        p['z_x'], p['target'] = 0., np.arange(len(p))
        self.assertTrue(group_test(p, ['z_x'], 'target', 5, 10).empty)


class LedgerTests(unittest.TestCase):
    def test_explicit_common_start_keeps_cash_before_first_valid_target(self):
        p, b = score_fixture(7)
        p.loc[p.date.eq(b.date.iloc[0]) & p.code.eq('B'), 'score'] = np.nan
        bt = run_open_to_open_backtest(p, b, 2, 2, 0., signal_start=b.date.iloc[0])
        self.assertEqual(bt.daily.date.iloc[0], b.date.iloc[1])
        self.assertAlmostEqual(bt.daily.cash.iloc[0], 1.)
        self.assertAlmostEqual(bt.daily.portfolio_nav.iloc[0], 1.)

    def test_missing_open_uses_previous_close_not_today_close(self):
        raw, b = raw_fixture(5, 2)
        code = raw.code.iloc[0]
        previous = float(raw.loc[raw.code.eq(code) & raw.date.eq(b.date.iloc[1]), 'close'].iloc[0])
        mask = raw.code.eq(code) & raw.date.eq(b.date.iloc[2])
        raw.loc[mask, 'open'] = np.nan
        raw.loc[mask, 'close'] = 99999.
        p, _ = align_panel(raw, b)
        self.assertAlmostEqual(p.loc[p.code.eq(code) & p.date.eq(b.date.iloc[2]), 'mark_open'].iloc[0], previous)

    def test_limit_guard_blocks_only_appropriate_side(self):
        raw, b = raw_fixture(5, 2)
        code = raw.code.iloc[0]
        up = raw.code.eq(code) & raw.date.eq(b.date.iloc[1])
        down = raw.code.eq(code) & raw.date.eq(b.date.iloc[2])
        raw.loc[up, 'open'] = raw.loc[up, 'preclose'] * 1.1
        raw.loc[down, 'open'] = raw.loc[down, 'preclose'] * .9
        p, _ = align_panel(raw, b)
        up_row = p.loc[p.code.eq(code) & p.date.eq(b.date.iloc[1])].iloc[0]
        down_row = p.loc[p.code.eq(code) & p.date.eq(b.date.iloc[2])].iloc[0]
        self.assertFalse(up_row.can_buy_open)
        self.assertTrue(up_row.can_sell_open)
        self.assertTrue(down_row.can_buy_open)
        self.assertFalse(down_row.can_sell_open)

    def test_flat_prices_no_cost_nav_constant(self):
        p, b = score_fixture()
        bt = run_open_to_open_backtest(p, b, 2, 1, 0.)
        np.testing.assert_allclose(bt.daily.portfolio_nav, 1.)
    def test_first_cost_is_included(self):
        p, b = score_fixture()
        bt = run_open_to_open_backtest(p, b, 2, 10, .001)
        self.assertAlmostEqual(bt.daily.portfolio_nav.iloc[0], 1 / 1.001)
        self.assertAlmostEqual(bt.daily.portfolio_return.iloc[0], 1 / 1.001 - 1)
        self.assertAlmostEqual(bt.trades.fee.sum(), bt.daily.transaction_cost.sum())
    def test_missing_quote_does_not_remove_assets(self):
        p, b = score_fixture()
        day = b.date.iloc[2]
        p.loc[p.date.eq(day) & p.code.eq('B'), 'open'] = np.nan
        p.loc[p.date.eq(day) & p.code.eq('B'), ['can_buy_open', 'can_sell_open']] = False
        bt = run_open_to_open_backtest(p, b, 2, 10, 0.)
        np.testing.assert_allclose(bt.daily.portfolio_nav, 1.)
        self.assertEqual(bt.positions.loc[bt.positions.date.eq(day), 'code'].nunique(), 2)
    def test_blocked_sell_preserves_position_and_cash(self):
        p, b = score_fixture()
        p.loc[p.date.ge(b.date.iloc[1]) & p.code.eq('B'), 'score'] = 3.
        p.loc[p.date.eq(b.date.iloc[2]) & p.code.eq('A'), 'can_sell_open'] = False
        bt = run_open_to_open_backtest(p, b, 1, 1, 0.)
        held = bt.positions.loc[bt.positions.date.eq(b.date.iloc[2])]
        self.assertEqual(held.code.tolist(), ['A'])
        self.assertGreater(len(bt.events.loc[bt.events.event.eq('sell_blocked')]), 0)
        self.assertTrue(bt.daily.cash.ge(-1e-12).all())
    def test_blocked_buy_leaves_cash_without_reweighting(self):
        p, b = score_fixture()
        p.loc[p.date.eq(b.date.iloc[1]) & p.code.eq('B'), 'can_buy_open'] = False
        bt = run_open_to_open_backtest(p, b, 2, 10, 0.)
        self.assertAlmostEqual(bt.daily.cash.iloc[0], .5)
        self.assertAlmostEqual(bt.positions.market_value.iloc[0], .5)
    def test_fully_rotated_turnover_is_two(self):
        p, b = score_fixture()
        p.loc[p.date.ge(b.date.iloc[1]) & p.code.eq('B'), 'score'] = 3.
        bt = run_open_to_open_backtest(p, b, 1, 1, 0.)
        self.assertAlmostEqual(bt.daily.turnover.iloc[1], 2.)
    def test_quantity_drifts_without_daily_equal_weight_reset(self):
        p, b = score_fixture()
        p.loc[p.date.ge(b.date.iloc[2]) & p.code.eq('A'), ['open', 'mark_open']] = 20.
        bt = run_open_to_open_backtest(p, b, 2, 10, 0.)
        day = bt.positions.loc[bt.positions.date.eq(b.date.iloc[2])].set_index('code')
        self.assertAlmostEqual(day.loc['A', 'actual_weight'], 2 / 3)
        self.assertAlmostEqual(day.loc['A', 'quantity'], .05)
    def test_ledger_conservation_in_random_prices(self):
        raw, b = raw_fixture(40, 6)
        p, _ = align_panel(raw, b, block_limit_open=False)
        p['score'] = np.random.default_rng(22).normal(size=len(p))
        bt = run_open_to_open_backtest(p, b, 3, 3, .003)
        np.testing.assert_allclose(bt.daily.portfolio_nav, bt.daily.cash + bt.daily.stock_value, atol=1e-12)
        np.testing.assert_allclose(bt.daily.portfolio_nav, bt.daily.pretrade_nav - bt.daily.transaction_cost, atol=1e-12)
        self.assertTrue(bt.daily.cash.ge(-1e-12).all())
        self.assertAlmostEqual((1 + bt.daily.portfolio_return).prod(), bt.daily.portfolio_nav.iloc[-1])
    def test_signal_executes_next_day(self):
        p, b = score_fixture()
        bt = run_open_to_open_backtest(p, b, 1, 2, 0.)
        self.assertEqual(bt.holdings.execution_date.iloc[0], b.date.iloc[1])
        self.assertTrue((bt.holdings.signal_date < bt.holdings.execution_date).all())
    def test_schedule_does_not_compress_missing_scores(self):
        p, b = score_fixture(9)
        p.loc[p.date.eq(b.date.iloc[2]), 'score'] = np.nan
        targets, events = build_execution_targets(p, 1, 2)
        self.assertEqual(list(targets), [b.date.iloc[i] for i in [1, 5, 7]])
        self.assertEqual(events.event.iloc[0], 'no_signal')
    def test_tied_scores_deterministic(self):
        p, b = score_fixture()
        p['score'] = 1.
        a, _ = build_execution_targets(p, 1, 2)
        c, _ = build_execution_targets(p.sample(frac=1, random_state=2), 1, 2)
        self.assertEqual(a, c)
    def test_duplicate_keys_fail(self):
        p, b = score_fixture()
        with self.assertRaises(ValueError):
            run_open_to_open_backtest(pd.concat([p, p.iloc[[0]]]), b, 1, 2, 0.)
    def test_invalid_cost_fails(self):
        p, b = score_fixture()
        with self.assertRaises(ValueError):
            run_open_to_open_backtest(p, b, 1, 2, -.01)
    def test_no_valuation_for_held_stock_fails_loudly(self):
        p, b = score_fixture()
        p.loc[p.date.eq(b.date.iloc[2]) & p.code.eq('A'), 'mark_open'] = np.nan
        with self.assertRaises(ValueError):
            run_open_to_open_backtest(p, b, 1, 10, 0.)


class RiskAndPerformanceTests(unittest.TestCase):
    def test_var_does_not_read_current_return(self):
        r = pd.Series([.01, -.01, .02, -.02, .01, -.4])
        before = rolling_historical_var(r, .95, 4, 3)
        r.iloc[-1] = .8
        assert_series_equal(before, rolling_historical_var(r, .95, 4, 3))
    def test_var_es_order(self):
        v, e = historical_var_es(pd.Series(np.linspace(-.1, .1, 1001)), .95)
        self.assertGreater(v, 0)
        self.assertGreaterEqual(e, v)
    def test_kupiec_zero_and_all_exceptions(self):
        for r in [pd.Series(np.zeros(100)), pd.Series(np.full(100, -.1))]:
            out = kupiec_pof_test(r, pd.Series(np.full(100, .02)), .95)
            self.assertTrue(np.isfinite(out['lr_pof']))
            self.assertTrue(0 <= out['p_value'] <= 1)
    def test_standard_sharpe(self):
        r = pd.Series([-.1, .11, .01, -.02])
        self.assertAlmostEqual(performance_metrics(r)['sharpe_rf0'], r.mean() / r.std(ddof=1) * np.sqrt(252))
    def test_initial_cost_drawdown(self):
        self.assertAlmostEqual(max_drawdown(pd.Series([.99, .99])), -.01)
    def test_performance_keeps_first_cost(self):
        self.assertAlmostEqual(performance_metrics(pd.Series([-.001, 0., 0.]))['total_return'], -.001)
    def test_comparison_rejects_misaligned_dates(self):
        d = pd.DataFrame({'date': pd.bdate_range('2022-01-03', periods=3), 'portfolio_return': [0., .01, -.01], 'benchmark_return': [np.nan, .01, 0.]})
        with self.assertRaises(ValueError):
            comparison_table(d)


class SnapshotTests(unittest.TestCase):
    def test_nontrading_requested_start_does_not_break_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'stock.csv'
            path.write_text('date,close\n2021-01-04,10\n', encoding='utf-8')
            metadata = {'adjustflag': '2', 'coverage_start': '2021-01-04', 'coverage_end': '2021-01-08', 'sha256': sha256(path)}
            self.assertTrue(cache_usable(path, metadata, '2021-01-04', '2021-01-08'))
            path.write_text('modified', encoding='utf-8')
            self.assertFalse(cache_usable(path, metadata, '2021-01-04', '2021-01-08'))
    def test_crop_and_checksum_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            make_demo(path, periods=20, n_stocks=10)
            raw, b, _ = load_raw_snapshot(path, '2023-01-09', '2023-01-20', 10)
            self.assertTrue(raw.date.between('2023-01-09', '2023-01-20').all())
            self.assertTrue(b.date.between('2023-01-09', '2023-01-20').all())
            file = path / 'raw' / 'sh_600000.csv'
            file.write_text(file.read_text() + '\n')
            with self.assertRaises(ValueError):
                load_raw_snapshot(path)
    def test_invalid_limit(self):
        with self.assertRaises(ValueError):
            choose_universe(None, 9)


if __name__ == '__main__':
    unittest.main()
