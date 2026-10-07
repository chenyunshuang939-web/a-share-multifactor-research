from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_provider import BaoStockProvider  # noqa: E402


def main() -> None:
    end = date.today()
    start = end - timedelta(days=60)
    with BaoStockProvider() as provider:
        df = provider.history_stock("sz.000001", start.isoformat(), end.isoformat())
    print(df.tail().to_string(index=False))
    print(f"\nBaoStock connection OK. rows={len(df)}")


if __name__ == "__main__":
    main()
