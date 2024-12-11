from pathlib import Path
from gym_trading_env.downloader import download
import datetime
import warnings
import calendar

warnings.filterwarnings("ignore", category=DeprecationWarning)


for y in range(2021, 2024 + 1):
    for m in range(1, 12 + 1):
        path = f"data/futures/15m/{y}/{m}"
        Path(path).mkdir(parents=True, exist_ok=True)
        download(
            exchange_names=["binanceusdm"],
            symbols=[
                "BTCDOM/USDT",
                "BTC/USDT",
                "ETH/USDT",
                "BNB/USDT",
                "DOGE/USDT",
                "XRP/USDT",
                "XLM/USDT",
                "SOL/USDT",
                "ADA/USDT",
                "SAND/USDT",
                "DOT/USDT",
                "ALGO/USDT",
                "TRX/USDT",
                "EOS/USDT",
                "FTM/USDT",
                "LTC/USDT",
                "LINK/USDT",
                "BCH/USDT",
                "AVAX/USDT",
                "ATOM/USDT",
            ],
            timeframe="15m",
            dir=path,
            since=datetime.datetime(year=y, month=m, day=1),
            until=datetime.datetime(year=y, month=m, day=calendar.monthrange(y, m)[1]),
        )
