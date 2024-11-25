from pathlib import Path
from gym_trading_env.downloader import download
import datetime
import warnings
import calendar

warnings.filterwarnings("ignore", category=DeprecationWarning)


for y in range(2023, 2024 + 1):
    for m in range(1, 12 + 1):
        path = f"data/train/month_15m/{y}/{m}"
        Path(path).mkdir(parents=True, exist_ok=True)
        download(
            exchange_names=["binance"],
            symbols=[
                "DOGE/USDT",
                "XRP/USDT",
                "BTC/USDT",
                "XLM/USDT",
                "BNB/USDT",
                "ETH/USDT",
                "SOL/USDT",
                "ADA/USDT",
            ],
            timeframe="15m",
            dir=path,
            since=datetime.datetime(year=y, month=m, day=1),
            until=datetime.datetime(year=y, month=m, day=calendar.monthrange(y, m)[1]),
        )
