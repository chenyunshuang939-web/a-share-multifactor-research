"""默认在线下载；--offline 使用本地快照；--demo 使用独立目录生成合成数据。"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import DEFAULT_LIMIT


def run(command: list[str]) -> None:
    print('\n$ ' + ' '.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description='Multi-factor research: download/demo -> calendar panel -> research -> ledger -> diagnostics')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--offline', action='store_true')
    mode.add_argument('--demo', action='store_true')
    p.add_argument('--limit', type=int)
    p.add_argument('--start')
    p.add_argument('--end')
    p.add_argument('--force', action='store_true')
    p.add_argument('--experiments', action='store_true')
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--output-dir', type=Path)
    a = p.parse_args()
    if a.limit is not None and a.limit < 10:
        p.error('--limit must be at least 10')
    if (a.offline or a.demo) and a.force:
        p.error('--force applies to online downloads only')
    data_dir = (a.data_dir or (ROOT / 'data' / 'demo' if a.demo else ROOT / 'data')).resolve()
    output_dir = (a.output_dir or (ROOT / 'outputs' / 'demo' if a.demo else ROOT / 'outputs')).resolve()
    if a.demo:
        run([sys.executable, str(ROOT / 'scripts/make_demo_data.py'), '--data-dir', str(data_dir)])
    elif not a.offline:
        cmd = [sys.executable, str(ROOT / 'scripts/download_data.py'), '--data-dir', str(data_dir), '--limit', str(a.limit or DEFAULT_LIMIT)]
        for flag, value in [('--start', a.start), ('--end', a.end)]:
            if value:
                cmd += [flag, value]
        if a.force:
            cmd.append('--force')
        run(cmd)
    build = [sys.executable, str(ROOT / 'scripts/build_panel.py'), '--data-dir', str(data_dir)]
    for flag, value in [('--start', a.start), ('--end', a.end), ('--limit', a.limit)]:
        if value is not None:
            build += [flag, str(value)]
    run(build)
    main_cmd = [sys.executable, str(ROOT / 'main.py'), '--data-dir', str(data_dir), '--output-dir', str(output_dir)]
    if a.experiments:
        main_cmd.append('--experiments')
    run(main_cmd)


if __name__ == '__main__':
    main()
