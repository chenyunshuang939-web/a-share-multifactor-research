from __future__ import annotations

from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT / "outputs"

START_DATE = "2021-01-01"
END_DATE = date.today().strftime("%Y-%m-%d")
DEFAULT_LIMIT = 30
BENCHMARK_CODE = "sh.000300"

# Factor research
FORWARD_HORIZON = 5
MIN_CROSS_SECTION = 10
N_GROUPS = 5
MAD_N = 3.0
IC_LOOKBACK = 60
IC_MIN_PERIODS = 20
ICIR_CLIP = 3.0
# 信号 T 收盘；T+1 开盘买入，T+H+1 开盘退出。因此标签延迟 H+1 个市场交易日。

# Portfolio / backtest
TOP_N = 5
REBALANCE_DAYS = 5
ONE_WAY_COST = 0.0010
ANNUAL_TRADING_DAYS = 252
BLOCK_LIMIT_OPEN = True
TRAIN_END = "2023-12-31"
VALIDATION_END = "2024-12-31"
PROJECT_VERSION = "1.0.0"

# Data download
DOWNLOAD_RETRIES = 3
DOWNLOAD_RETRY_WAIT = 2.0
DOWNLOAD_SLEEP = 0.25

# A diversified fallback list. The downloader first checks the current CSI 300
# universe from BaoStock, then uses names from this list that are still members.
PREFERRED_UNIVERSE = [
    "sh.600519",  # Kweichow Moutai
    "sz.000858",  # Wuliangye
    "sz.000333",  # Midea Group
    "sz.000651",  # Gree Electric
    "sz.000001",  # Ping An Bank
    "sh.600036",  # China Merchants Bank
    "sh.601318",  # Ping An Insurance
    "sh.600276",  # Hengrui Medicine
    "sz.300750",  # CATL
    "sz.002594",  # BYD
    "sh.601012",  # LONGi
    "sh.600900",  # China Yangtze Power
    "sh.601166",  # Industrial Bank
    "sh.601398",  # ICBC
    "sh.601288",  # ABC
    "sh.600030",  # CITIC Securities
    "sh.600887",  # Yili
    "sz.002415",  # Hikvision
    "sz.300059",  # East Money Information (stock, not data source)
    "sh.601899",  # Zijin Mining
]

FACTOR_COLUMNS = [
    "mom20",
    "mom60",
    "rev5",
    "vol20",
    "vol60",
    "turn20",
    "volratio20",
    "amihud20",
    "rsi14",
    "bollpos20",
    "trend2060",
]

# 静态对照的方向是预先声明的经济假设，不按全样本 IC 的正负选方向。
STATIC_DIRECTIONS = {
    "mom20": 1, "mom60": 1, "rev5": 1, "vol20": -1, "vol60": -1,
    "turn20": -1, "volratio20": 1, "amihud20": -1, "rsi14": -1,
    "bollpos20": -1, "trend2060": 1,
}
REDUCED_FACTORS = ["mom60", "rev5", "vol20", "turn20", "volratio20", "amihud20"]
