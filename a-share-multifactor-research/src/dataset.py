"""交易日历、数据质量、开盘可交易性。禁止用当天收盘/最高最低价决定开盘交易。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

NUMERIC = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pctChg', 'tradestatus', 'isST']


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def validate_keys(frame: pd.DataFrame, keys: list[str], name: str) -> None:
    if frame[keys].isna().any().any():
        raise ValueError(f'{name}: missing key values')
    if frame.duplicated(keys).any():
        raise ValueError(f'{name}: duplicate keys {keys}')


def validate_benchmark(benchmark: pd.DataFrame) -> pd.DataFrame:
    b = benchmark.copy()
    b['date'] = pd.to_datetime(b['date'], errors='raise')
    validate_keys(b, ['date'], 'benchmark')
    for col in ['open', 'close']:
        b[col] = pd.to_numeric(b[col], errors='coerce')
        if (~np.isfinite(b[col]) | b[col].le(0)).any():
            raise ValueError(f'benchmark: invalid {col}')
    if len(b) < 2:
        raise ValueError('Need at least two benchmark trading dates')
    return b.sort_values('date').reset_index(drop=True)


def open_limit_rate(code: str, dates: pd.Series, st: pd.Series) -> np.ndarray:
    """研究用近似阈值，未覆盖 IPO 无涨跌幅限制、历史特殊处理等全部规则。"""
    if code.startswith('sh.688'):
        return np.full(len(dates), .20)
    if code.startswith(('sz.300', 'sz.301')):
        return np.where(dates.ge(pd.Timestamp('2020-08-24')), .20, np.where(st.eq(1), .05, .10))
    return np.where(st.eq(1), .05, .10)


def align_panel(raw: pd.DataFrame, benchmark: pd.DataFrame, *, block_limit_open: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """补齐市场日历但不伪造可成交价格。原始数据中的缺行标记为 unknown_missing。

    research_close: 最近已知收盘价，用于停牌期间的估值收益及固定日历滚动。
    mark_open: 当天有效开盘报价，否则最近的历史收盘价，仅供估值。
    can_*_open: 开盘信息集内可判断的成交约束；不查看当天最终成交量和收盘价。
    """
    b = validate_benchmark(benchmark)
    calendar = pd.DatetimeIndex(b['date'])
    raw = raw.copy()
    raw['date'] = pd.to_datetime(raw['date'], errors='raise')
    validate_keys(raw, ['date', 'code'], 'stock panel')
    if not raw['date'].isin(calendar).all():
        raise ValueError('Stock dates outside benchmark calendar; crop both inputs before alignment')
    blocks, quality = [], []
    for code, data in raw.groupby('code', sort=True):
        data = data.set_index('date').reindex(calendar).rename_axis('date').reset_index()
        source = data['code'].notna()
        data['code'] = code
        for col in NUMERIC:
            if col not in data:
                data[col] = np.nan
            data[col] = pd.to_numeric(data[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
        data['source_row'] = source
        data['unknown_missing'] = ~source
        data['confirmed_suspension'] = source & data.tradestatus.eq(0)
        open_valid = data.open.gt(0) & data.tradestatus.eq(1)
        close_valid = data.close.gt(0)
        # 无有效收盘时只沿用过去值，绝不 bfill。
        known_close = data.close.where(close_valid)
        data['research_close'] = known_close.ffill()
        data['mark_open'] = data.open.where(open_valid).fillna(known_close.ffill().shift(1))
        data['signal_eligible'] = open_valid & close_valid & data.isST.eq(0)
        rate = open_limit_rate(code, data.date, data.isST)
        opening_change = data.open.div(data.preclose.where(data.preclose.gt(0))) - 1
        data['limit_up_open'] = opening_change.ge(rate - .0005) if block_limit_open else False
        data['limit_down_open'] = opening_change.le(-rate + .0005) if block_limit_open else False
        data['can_buy_open'] = open_valid & ~data.limit_up_open
        data['can_sell_open'] = open_valid & ~data.limit_down_open
        # 有确认停牌状态的量和换手率为零；unknown 缺行保持 NaN，避免掩盖数据问题。
        for col in ['volume', 'amount', 'turn']:
            data.loc[data.confirmed_suspension, col] = 0.0
        quality.append({
            'code': code, 'calendar_rows': len(data), 'source_rows': int(source.sum()),
            'unknown_missing_rows': int((~source).sum()),
            'confirmed_suspensions': int(data.confirmed_suspension.sum()),
            'invalid_open_source_rows': int((source & ~data.open.gt(0)).sum()),
            'invalid_close_source_rows': int((source & ~close_valid).sum()),
            'blocked_buy_open_rows': int((open_valid & data.limit_up_open).sum()),
            'blocked_sell_open_rows': int((open_valid & data.limit_down_open).sum()),
        })
        blocks.append(data)
    if not blocks:
        raise ValueError('Empty stock panel')
    return pd.concat(blocks, ignore_index=True).sort_values(['code', 'date']).reset_index(drop=True), pd.DataFrame(quality)


def load_raw_snapshot(data_dir: Path, start: str | None = None, end: str | None = None,
                      limit: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw_dir = data_dir / 'raw'
    manifest_path = raw_dir / 'download_manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError('Missing download_manifest.json. Download a data snapshot or run --demo.')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    codes = manifest['codes']
    if limit is not None:
        if limit < 10:
            raise ValueError('--limit must be at least 10')
        codes = codes[:limit]
    if len(codes) < 10:
        raise ValueError('At least 10 stocks are required')
    paths = [raw_dir / (c.replace('.', '_') + '.csv') for c in codes]
    paths.append(raw_dir / 'benchmark_hs300.csv')
    for path in paths:
        expected = manifest['files'].get(path.name, {}).get('sha256')
        if not expected or sha256(path) != expected:
            raise ValueError(f'Snapshot checksum mismatch: {path.name}; rerun download instead of mixing files')
    start = start or manifest['requested_start']
    end = end or manifest['requested_end']
    if pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError('start must be <= end')
    if pd.Timestamp(start) < pd.Timestamp(manifest['requested_start']) or pd.Timestamp(end) > pd.Timestamp(manifest['requested_end']):
        raise ValueError('Requested range exceeds saved snapshot request range; download a new snapshot')
    frames = []
    for code, path in zip(codes, paths[:-1]):
        frame = pd.read_csv(path, parse_dates=['date'])
        if not frame.code.eq(code).all():
            raise ValueError(f'Wrong code inside {path.name}')
        frame = frame.loc[frame.date.between(start, end)].copy()
        if frame.empty:
            raise ValueError(f'No rows for {code} in requested range')
        frames.append(frame)
    benchmark = pd.read_csv(paths[-1], parse_dates=['date'])
    benchmark = benchmark.loc[benchmark.date.between(start, end)].copy()
    selection = dict(manifest, selected_codes=codes, selected_start=start, selected_end=end)
    return pd.concat(frames, ignore_index=True), benchmark, selection
