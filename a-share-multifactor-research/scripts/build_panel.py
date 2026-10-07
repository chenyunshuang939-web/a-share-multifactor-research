from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import BLOCK_LIMIT_OPEN
from src.dataset import align_panel, load_raw_snapshot, write_json, sha256


def build(data_dir: Path, start: str | None = None, end: str | None = None, limit: int | None = None) -> None:
    raw, benchmark, manifest = load_raw_snapshot(data_dir, start, end, limit)
    panel, quality = align_panel(raw, benchmark, block_limit_open=BLOCK_LIMIT_OPEN)
    out = data_dir / 'processed'
    out.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out / 'panel.csv', index=False)
    benchmark.sort_values('date').to_csv(out / 'benchmark.csv', index=False)
    quality.to_csv(out / 'data_quality.csv', index=False)
    manifest['processed_files'] = {p.name: sha256(p) for p in [out / 'panel.csv', out / 'benchmark.csv', out / 'data_quality.csv']}
    manifest['block_limit_open'] = BLOCK_LIMIT_OPEN
    manifest['actual_start'] = str(panel.date.min().date())
    manifest['actual_end'] = str(panel.date.max().date())
    write_json(out / 'build_manifest.json', manifest)
    print(f'[panel] stocks={panel.code.nunique()}, rows={len(panel)}, trading_dates={panel.date.nunique()}, missing_rows={int(panel.unknown_missing.sum())}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Build a calendar-aligned, range-cropped panel from a checksummed snapshot')
    p.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    p.add_argument('--start')
    p.add_argument('--end')
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    build(args.data_dir, args.start, args.end, args.limit)
