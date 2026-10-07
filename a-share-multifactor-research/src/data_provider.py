from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


class BaoStockError(RuntimeError):
    pass


@dataclass
class BaoStockProvider:
    """Small wrapper around BaoStock with explicit login/logout semantics."""

    _bs: object | None = None
    _logged_in: bool = False

    def login(self) -> None:
        if self._logged_in:
            return
        #如果登陆的话直接返回
        try:
            import baostock as bs
            #将baostock导入
        except ImportError as exc:
            raise BaoStockError(
                "baostock is not installed. Run: python -m pip install -r requirements.txt"
            ) from exc

        result = bs.login()
        if result.error_code != "0":
            raise BaoStockError(
                f"BaoStock login failed: {result.error_code} {result.error_msg}"
            )
        self._bs = bs
        self._logged_in = True

    def logout(self) -> None:
        if self._logged_in and self._bs is not None:
            try:
                self._bs.logout()
            finally:
                self._logged_in = False
                self._bs = None

    def __enter__(self) -> "BaoStockProvider":
        self.login()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.logout()

    @staticmethod
    def _rows_to_frame(result) -> pd.DataFrame:
        if result.error_code != "0":
            raise BaoStockError(f"BaoStock query failed: {result.error_code} {result.error_msg}")
        rows: list[list[str]] = []
        while result.next():
            rows.append(result.get_row_data())
        if result.error_code != "0":
            raise BaoStockError(f"BaoStock stream failed: {result.error_code} {result.error_msg}")
        return pd.DataFrame(rows, columns=result.fields)

    def current_hs300(self) -> pd.DataFrame:
        self.login()
        result = self._bs.query_hs300_stocks()
        frame = self._rows_to_frame(result)
        if frame.empty:
            raise BaoStockError("BaoStock returned an empty CSI 300 constituent list.")
        return frame

    def history_stock(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        self.login()
        fields = ",".join(
            [
                "date",
                "code",
                "open",
                "high",
                "low",
                "close",
                "preclose",
                "volume",
                "amount",
                "turn",
                "pctChg",
                "tradestatus",
                "isST",
            ]
        )
        result = self._bs.query_history_k_data_plus(
            code,
            fields,
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2",  # forward-adjusted prices
        )
        return self._clean_history(self._rows_to_frame(result), is_index=False)

    def history_index(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        self.login()
        fields = ",".join(
            [
                "date",
                "code",
                "open",
                "high",
                "low",
                "close",
                "preclose",
                "volume",
                "amount",
                "pctChg",
            ]
        )
        result = self._bs.query_history_k_data_plus(
            code,
            fields,
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3",
        )
        return self._clean_history(self._rows_to_frame(result), is_index=True)

    @staticmethod
    def _clean_history(frame: pd.DataFrame, *, is_index: bool) -> pd.DataFrame:
        if frame.empty:
            return frame
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        numeric_columns: Iterable[str] = [
            c
            for c in [
                "open",
                "high",
                "low",
                "close",
                "preclose",
                "volume",
                "amount",
                "turn",
                "pctChg",
                "tradestatus",
                "isST",
            ]
            if c in frame.columns
        ]
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        # 保留停牌行与缺失价格，交由统一交易日历区分“无报价”和“可成交”。
        frame = frame.dropna(subset=["date", "code"]).sort_values("date")
        if frame.duplicated(["date", "code"]).any():
            raise BaoStockError("Duplicate date/code returned by provider")
        return frame.drop_duplicates(["date", "code"], keep="last").reset_index(drop=True)
