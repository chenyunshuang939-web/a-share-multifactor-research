"""BaoStock 下载；缓存用查询覆盖日历与校验和判断，不用首条股票记录猜测覆盖范围。"""
from __future__ import annotations
import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import json
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import DEFAULT_LIMIT, START_DATE, END_DATE, PREFERRED_UNIVERSE, DOWNLOAD_RETRIES, DOWNLOAD_RETRY_WAIT, DOWNLOAD_SLEEP
from src.data_provider import BaoStockProvider
from src.dataset import sha256, write_json, validate_benchmark


def choose_universe(provider, limit: int) -> pd.DataFrame:
    if limit < 10:
        raise ValueError('--limit must be at least 10')
    live = provider.current_hs300().drop_duplicates('code')
    codes = sorted(live.code.tolist())
    if limit > len(codes):
        raise ValueError('Requested stock count exceeds current constituents')
    selected = [c for c in PREFERRED_UNIVERSE if c in codes]
    selected += [c for c in codes if c not in selected]
    selected = selected[:limit]
    names = dict(zip(live.code, live['code_name'])) if 'code_name' in live else {}
    return pd.DataFrame({'code': selected, 'name': [names.get(c, '') for c in selected]})


def cache_usable(path: Path, metadata: dict, calendar_start: str, calendar_end: str) -> bool:
    if not path.exists() or not metadata or metadata.get('adjustflag') != '2':
        return False
    if metadata.get('sha256') != sha256(path):
        return False
    return (pd.Timestamp(metadata['coverage_start']) <= pd.Timestamp(calendar_start)
            and pd.Timestamp(metadata['coverage_end']) >= pd.Timestamp(calendar_end))


def retry(call, label: str):
    error = None
    for attempt in range(DOWNLOAD_RETRIES):
        try:
            data = call()
            if data.empty:
                raise ValueError(f'Empty response: {label}')
            return data
        except Exception as exc:
            error = exc
            print(f'[retry] {label}: {attempt + 1}/{DOWNLOAD_RETRIES}: {exc}', flush=True)
            if attempt + 1 < DOWNLOAD_RETRIES:
                time.sleep(DOWNLOAD_RETRY_WAIT * (attempt + 1))
    raise RuntimeError(f'Download failed: {label}; snapshot was not published') from error


def download(data_dir: Path, start: str, end: str, limit: int, force: bool = False) -> None:
    if pd.Timestamp(start) > pd.Timestamp(end) or limit < 10:
        raise ValueError('Invalid date range or stock limit (<10)')
    raw = data_dir / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    previous_path = raw / 'download_manifest.json'
    previous = json.loads(previous_path.read_text(encoding='utf-8')) if previous_path.exists() else {}
    prepared, metadata = {}, {}
    with BaoStockProvider() as provider:
        universe = choose_universe(provider, limit)
        benchmark = validate_benchmark(retry(lambda: provider.history_index('sh.000300', start, end), 'benchmark'))
        first, last = str(benchmark.date.min().date()), str(benchmark.date.max().date())
        for i, code in enumerate(universe.code, 1):
            filename = code.replace('.', '_') + '.csv'
            old = previous.get('files', {}).get(filename, {})
            if not force and previous.get('source') != 'synthetic_demo' and cache_usable(raw / filename, old, first, last):
                metadata[filename] = dict(old)
                print(f'[{i}/{limit}] cached {code}', flush=True)
                continue
            print(f'[{i}/{limit}] downloading {code}', flush=True)
            frame = retry(lambda: provider.history_stock(code, start, end), code)
            prepared[filename] = frame
            metadata[filename] = {'coverage_start': first, 'coverage_end': last, 'adjustflag': '2'}
            time.sleep(DOWNLOAD_SLEEP)
    # 全部网络查询成功后才发布本地文件；每个 CSV 原子替换，manifest 最后写入。
    prepared['benchmark_hs300.csv'] = benchmark
    for filename, frame in prepared.items():
        temporary = raw / (filename + '.tmp')
        frame.to_csv(temporary, index=False)
        temporary.replace(raw / filename)
    universe.to_csv(raw / 'active_universe.csv', index=False)
    metadata['benchmark_hs300.csv'] = {'coverage_start': first, 'coverage_end': last, 'adjustflag': '3'}
    for filename in metadata:
        metadata[filename]['sha256'] = sha256(raw / filename)
    write_json(previous_path, {'schema_version': 1, 'source': 'baostock_current_hs300',
        'downloaded_utc': datetime.now(timezone.utc).isoformat(), 'requested_start': start, 'requested_end': end,
        'universe_selection': 'preferred list then sorted current constituents; not historical membership',
        'codes': universe.code.tolist(), 'files': metadata})
    print(f'[download] published {limit} stocks; calendar {first} -> {last}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    p.add_argument('--start', default=START_DATE)
    p.add_argument('--end', default=END_DATE)
    p.add_argument('--limit', type=int, default=DEFAULT_LIMIT)
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    download(a.data_dir, a.start, a.end, a.limit, a.force)
