from gym_trading_env.downloader import download
import datetime
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 

download(exchange_names = ["binance"],
    symbols= ["DOGE/USDT"],
    timeframe= "15m",
    dir = "15m",
    since= datetime.datetime(year= 2024, month= 1, day=1),
)