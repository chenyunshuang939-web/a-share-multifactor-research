"""确定性合成数据，用于 GitHub clone 后无网络运行；绝不作为真实策略业绩。"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset import sha256, write_json


def make_demo(data_dir: Path, periods: int = 360, n_stocks: int = 20) -> None:
    rng = np.random.default_rng(20261006)
    dates = pd.bdate_range('2023-01-02', periods=periods)  # 合成工作日日历，不冒充交易所日历。
    market = rng.normal(.00015, .009, periods)
    raw = data_dir / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    codes, files = [], {}
    for i in range(n_stocks):
        code = f'sh.{600000 + i:06d}'
        codes.append(code)
        returns = market * (.6 + i / n_stocks) + rng.normal(0, .012 + .001 * (i % 3), periods)
        close = (15 + 3 * i) * np.exp(np.cumsum(returns))
        prev = np.r_[close[0], close[:-1]]
        open_ = prev * np.exp(rng.normal(0, .005, periods))
        volume = rng.lognormal(15, .3, periods).astype(int)
        frame = pd.DataFrame({'date': dates, 'code': code, 'open': open_, 'close': close,
                              'high': np.maximum(open_, close) * 1.002, 'low': np.minimum(open_, close) * .998,
                              'preclose': prev, 'volume': volume, 'amount': close * volume,
                              'turn': rng.uniform(.1, 3, periods), 'pctChg': (close / prev - 1) * 100,
                              'tradestatus': 1, 'isST': 0})
        if i % 5 == 0 and periods > 140:
            frame.loc[120:124, 'tradestatus'] = 0
            frame.loc[120:124, ['open', 'high', 'low', 'close', 'preclose']] = float(frame.loc[119, 'close'])
            frame.loc[120:124, ['volume', 'amount', 'turn']] = 0
        if i == 1 and periods > 180:
            frame = frame.drop(index=[175, 176])  # 故意保留未知缺行，测试诊断。
        name = code.replace('.', '_') + '.csv'
        frame.to_csv(raw / name, index=False)
        files[name] = {'coverage_start': str(dates[0].date()), 'coverage_end': str(dates[-1].date()), 'adjustflag': '2', 'sha256': sha256(raw / name)}
    close = 4000 * np.exp(np.cumsum(market))
    benchmark = pd.DataFrame({'date': dates, 'code': 'sh.000300', 'open': np.r_[close[0], close[:-1]], 'close': close})
    benchmark.to_csv(raw / 'benchmark_hs300.csv', index=False)
    files['benchmark_hs300.csv'] = {'coverage_start': str(dates[0].date()), 'coverage_end': str(dates[-1].date()), 'adjustflag': '3', 'sha256': sha256(raw / 'benchmark_hs300.csv')}
    pd.DataFrame({'code': codes, 'name': ['SYNTHETIC_' + str(i) for i in range(n_stocks)]}).to_csv(raw / 'active_universe.csv', index=False)
    write_json(raw / 'download_manifest.json', {'schema_version': 1, 'source': 'synthetic_demo', 'seed': 20261006,
        'requested_start': str(dates[0].date()), 'requested_end': str(dates[-1].date()), 'codes': codes, 'files': files})
    print(f'[demo] synthetic data only: {n_stocks} stocks x {periods} dates', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=Path, default=ROOT / 'data' / 'demo')
    a = p.parse_args()
    make_demo(a.data_dir)
