"""离线研究入口。只有 scripts/download_data.py 会访问 BaoStock。"""
from __future__ import annotations
import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import config as cfg
from src.dataset import sha256, write_json, validate_keys
from src.factors import build_factors
from src.preprocess import preprocess_cross_section
from src.factor_research import (add_forward_return, rank_ic, available_ic, dynamic_ic_weights, attach_composite_score,
                                 static_scores, summarize_ic, group_test, summarize_groups, mean_cross_sectional_corr)
from src.portfolio import run_open_to_open_backtest, build_execution_targets
from src.performance import comparison_table, period_comparison, performance_metrics
from src.risk import risk_report, worst_days


def run(data_dir: Path, output_dir: Path, experiments: bool = False) -> None:
    processed = data_dir / 'processed'
    manifest = json.loads((processed / 'build_manifest.json').read_text(encoding='utf-8'))
    for filename, digest in manifest['processed_files'].items():
        if sha256(processed / filename) != digest:
            raise ValueError(f'Processed checksum mismatch: {filename}; rerun build_panel.py')
    if manifest['block_limit_open'] != cfg.BLOCK_LIMIT_OPEN:
        raise ValueError('Trading guard changed; rebuild the panel before running research')
    output_dir.mkdir(parents=True, exist_ok=True)
    def save(frame, name, index=False):
        frame.to_csv(output_dir / name, index=index, encoding='utf-8-sig')
    print('[1/7] Load validated, calendar-aligned snapshot', flush=True)
    panel = pd.read_csv(processed / 'panel.csv', parse_dates=['date'])
    benchmark = pd.read_csv(processed / 'benchmark.csv', parse_dates=['date'])
    validate_keys(panel, ['date', 'code'], 'processed panel')
    calendar = pd.DatetimeIndex(sorted(panel.date.unique()))
    print('[2/7] Factors, cross-sectional preprocessing and two forward labels', flush=True)
    panel = build_factors(panel)
    panel.loc[~panel.signal_eligible, cfg.FACTOR_COLUMNS] = np.nan
    panel = preprocess_cross_section(panel, cfg.FACTOR_COLUMNS, cfg.MAD_N)
    for kind in ['open', 'close']:
        panel = add_forward_return(panel, cfg.FORWARD_HORIZON, kind)
    factors = ['z_' + f for f in cfg.FACTOR_COLUMNS]
    target = f'fwd_open_ret_{cfg.FORWARD_HORIZON}'
    print('[3/7] Raw IC, matured IC, dynamic weights and group diagnostics', flush=True)
    raw_ic = rank_ic(panel, factors, target, cfg.MIN_CROSS_SECTION)
    if not raw_ic[factors].notna().any().any():
        raise ValueError('No usable IC: increase history or eligible cross section')
    close_ic = rank_ic(panel, factors, f'fwd_close_ret_{cfg.FORWARD_HORIZON}', cfg.MIN_CROSS_SECTION)
    matured = available_ic(raw_ic, calendar, cfg.FORWARD_HORIZON + 1)
    weights = dynamic_ic_weights(matured, cfg.IC_LOOKBACK, cfg.IC_MIN_PERIODS, cfg.ICIR_CLIP)
    scored = attach_composite_score(panel, weights, factors)
    groups = group_test(panel, factors, target, cfg.N_GROUPS, cfg.MIN_CROSS_SECTION)
    group_summary, spreads = summarize_groups(groups, cfg.FORWARD_HORIZON)
    corr = mean_cross_sectional_corr(panel, factors, cfg.MIN_CROSS_SECTION)
    targets, _ = build_execution_targets(scored, cfg.TOP_N, cfg.REBALANCE_DAYS)
    if not targets:
        raise ValueError('No usable portfolio after factor/IC warm-up; use longer history')
    first_signal = targets[min(targets)]['signal_date']
    print(f'[4/7] Cash/position ledger; first signal={first_signal.date()}', flush=True)
    dynamic = run_open_to_open_backtest(scored, benchmark, cfg.TOP_N, cfg.REBALANCE_DAYS, cfg.ONE_WAY_COST, first_signal)
    static = static_scores(panel, cfg.STATIC_DIRECTIONS, cfg.FACTOR_COLUMNS, cfg.FACTOR_COLUMNS)
    reduced = static_scores(panel, cfg.STATIC_DIRECTIONS, cfg.REDUCED_FACTORS, cfg.FACTOR_COLUMNS)
    equal = static.copy()
    equal.loc[equal.score.notna(), 'score'] = 0.0
    controls = {}
    for name, source, top_n in [('static', static, cfg.TOP_N), ('static_reduced', reduced, cfg.TOP_N), ('equal_weight', equal, None)]:
        controls[name] = run_open_to_open_backtest(source, benchmark, top_n, cfg.REBALANCE_DAYS, cfg.ONE_WAY_COST, first_signal)
    comparison = dynamic.daily[['date', 'portfolio_nav', 'portfolio_return', 'benchmark_nav', 'benchmark_return']].copy()
    for name, bt in controls.items():
        extra = bt.daily[['date', 'portfolio_nav', 'portfolio_return']].rename(columns={'portfolio_nav': name + '_nav', 'portfolio_return': name + '_return'})
        comparison = comparison.merge(extra, on='date', how='left', validate='one_to_one')
    performance = comparison_table(comparison)
    periods = period_comparison(comparison, cfg.TRAIN_END, cfg.VALIDATION_END)
    print('[5/7] Ex-ante VaR, Kupiec and historical worst days', flush=True)
    indexed_returns = dynamic.daily.set_index('date').portfolio_return
    risk, rolling_risk = risk_report(indexed_returns)
    sensitivity = []
    if experiments:
        cases = [(f'cost_{c:g}', cfg.TOP_N, cfg.REBALANCE_DAYS, c) for c in [0, .0005, cfg.ONE_WAY_COST, .002]]
        cases += [(f'top_{n}', n, cfg.REBALANCE_DAYS, cfg.ONE_WAY_COST) for n in [3, 10]]
        cases += [('rebalance_10', cfg.TOP_N, 10, cfg.ONE_WAY_COST)]
        for name, top, rebalance, cost in cases:
            print(f'      sensitivity: {name}', flush=True)
            bt = dynamic if (top, rebalance, cost) == (cfg.TOP_N, cfg.REBALANCE_DAYS, cfg.ONE_WAY_COST) else run_open_to_open_backtest(scored, benchmark, top, rebalance, cost, first_signal)
            sensitivity.append({'case': name, 'top_n': top, 'rebalance_days': rebalance, 'one_way_cost': cost,
                                'total_two_way_turnover': bt.daily.turnover.sum(), **performance_metrics(bt.daily.portfolio_return)})
    print('[6/7] Save audit tables, latest signals, figures and source fingerprints', flush=True)
    save(raw_ic, 'factor_ic_daily.csv')
    save(close_ic, 'factor_ic_close_daily.csv')
    save(matured, 'available_ic_daily.csv')
    save(summarize_ic(raw_ic, cfg.FORWARD_HORIZON), 'factor_ic_summary.csv')
    save(weights, 'dynamic_factor_weights.csv')
    save(groups, 'factor_group_returns.csv')
    save(group_summary, 'factor_group_summary.csv')
    save(spreads, 'factor_group_spread.csv')
    save(corr, 'factor_rank_corr.csv', True)
    save(scored[['date', 'code', 'signal_eligible', 'score'] + factors], 'scored_panel.csv')
    save(panel[['date', 'code', target, 'open_label_maturity', f'fwd_close_ret_{cfg.FORWARD_HORIZON}', 'close_label_maturity']], 'label_audit.csv')
    save(dynamic.daily, 'portfolio_daily.csv')
    save(dynamic.holdings, 'holdings.csv')
    save(dynamic.trades, 'trades.csv')
    save(dynamic.positions, 'positions_daily.csv')
    save(dynamic.events, 'execution_events.csv')
    for name, bt in controls.items():
        save(bt.daily, name + '_daily.csv')
    save(comparison, 'comparison_daily.csv')
    save(performance, 'performance_summary.csv')
    save(periods, 'period_performance.csv')
    save(risk, 'risk_report.csv')
    save(rolling_risk.reset_index(), 'rolling_var.csv')
    save(worst_days(dynamic.daily), 'worst_5_days.csv')
    save(pd.read_csv(processed / 'data_quality.csv'), 'data_quality.csv')
    # 每次运行都覆盖敏感性文件，避免留下旧参数生成的表格。
    save(pd.DataFrame(sensitivity, columns=None if sensitivity else ['case', 'top_n', 'rebalance_days', 'one_way_cost']), 'sensitivity.csv')
    latest = scored.loc[scored.date.eq(calendar[-1]), ['date', 'code', 'score', 'signal_eligible']].sort_values(['score', 'code'], ascending=[False, True])
    save(latest, 'latest_signals.csv')
    fig, ax = plt.subplots(figsize=(11, 5))
    for col in comparison.columns:
        if col.endswith('_nav'):
            ax.plot(comparison.date, comparison[col], label=col.removesuffix('_nav'), lw=1.2)
    ax.set(title='Multi-factor research: cost-inclusive comparison', ylabel='NAV (initial capital = 1)')
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(output_dir / 'equity_curve.png', dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5))
    raw_ic.set_index('date').fillna(0).cumsum().plot(ax=ax, lw=.8)
    ax.set(title='Cumulative raw Rank IC (diagnostic, not tradable P&L)', ylabel='Cumulative IC')
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(output_dir / 'cumulative_rank_ic.png', dpi=150); plt.close(fig)
    source_paths = sorted(set([*cfg.ROOT.glob('*.py'), *cfg.ROOT.glob('src/*.py'), *cfg.ROOT.glob('scripts/*.py'), *cfg.ROOT.glob('tests/*.py')]))
    run_manifest = {
        'project_version': cfg.PROJECT_VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(), 'packages': {'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__, 'matplotlib': matplotlib.__version__},
        'data_source': manifest['source'], 'input_files': manifest['files'], 'processed_files': manifest['processed_files'],
        'selected_codes': manifest['selected_codes'], 'actual_start': str(calendar[0].date()), 'actual_end': str(calendar[-1].date()),
        'parameters': {k: getattr(cfg, k) for k in ['FORWARD_HORIZON', 'MIN_CROSS_SECTION', 'N_GROUPS', 'MAD_N', 'IC_LOOKBACK', 'IC_MIN_PERIODS', 'ICIR_CLIP', 'TOP_N', 'REBALANCE_DAYS', 'ONE_WAY_COST', 'BLOCK_LIMIT_OPEN', 'TRAIN_END', 'VALIDATION_END', 'STATIC_DIRECTIONS', 'REDUCED_FACTORS']},
        'label': 'open_T+1_to_open_T+H+1', 'ic_delay_trading_days': cfg.FORWARD_HORIZON + 1,
        'first_signal': str(first_signal.date()), 'first_execution': str(dynamic.daily.date.min().date()),
        'latest_scored_stocks': int(latest.score.notna().sum()), 'experiments_run': experiments,
        'source_sha256': {str(p.relative_to(cfg.ROOT)): sha256(p) for p in source_paths},
        'limitations': ['current constituents / selected sample, not historical point-in-time membership',
                        'retrospective time split: historical data already inspected; not an untouched holdout',
                        'fractional adjusted-price units; no real-share corporate-action ledger or lot rounding',
                        'approximate opening-limit guard; no intraday order-book/fill model',
                        'price-index benchmark versus adjusted-price stock returns; dividend convention differs',
                        'unknown missing source rows marked separately; not claimed to be confirmed suspensions'],
    }
    write_json(output_dir / 'run_manifest.json', run_manifest)
    print('[7/7] Done', flush=True)
    print(performance[['series', 'total_return', 'annual_return', 'sharpe_rf0', 'max_drawdown']].to_string(index=False))
    print(f'Latest date {calendar[-1].date()}: {latest.score.notna().sum()} scored stocks. Outputs: {output_dir.resolve()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run multi-factor research using local processed data only')
    parser.add_argument('--data-dir', type=Path, default=cfg.DATA_DIR)
    parser.add_argument('--output-dir', type=Path, default=cfg.OUTPUT_DIR)
    parser.add_argument('--experiments', action='store_true', help='Also run declared cost / top-N / rebalance sensitivity cases')
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.experiments)
