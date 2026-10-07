"""生成适合网页上传或代码审阅的纯源码包，使用白名单而不是猜测需要排除哪些文件。"""
from __future__ import annotations
from pathlib import Path
import zipfile
ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {'README.md', 'LICENSE', 'config.py', 'main.py', 'requirements.txt',
              'requirements-tested.txt', 'environment.yml', '.gitignore', '.gitattributes'}
ALLOWED_DIRS = {'src', 'scripts', 'tests', 'docs', 'reports', '.github'}


def package(destination: Path | None = None) -> Path:
    destination = destination or ROOT / 'dist' / 'a-share-multifactor-research.zip'
    destination.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for path in ROOT.rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix in {'.pyc', '.tmp'}:
            continue
        rel = path.relative_to(ROOT)
        if (len(rel.parts) == 1 and rel.name in ROOT_FILES) or rel.parts[0] in ALLOWED_DIRS or str(rel).replace('\\', '/') in {'data/README.md', 'data/.gitkeep', 'outputs/.gitkeep'}:
            paths.append(path)
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path, Path(ROOT.name) / path.relative_to(ROOT))
    print(f'Created {destination} ({len(paths)} files; no market-data cache or full outputs)')
    return destination


if __name__ == '__main__':
    package()
