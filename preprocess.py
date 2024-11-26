# Preprocess
import numpy as np
import pandas as pd
from sklearn.preprocessing import robust_scale


def SMA(df, ndays):
    SMA = pd.Series(df.close.rolling(ndays).mean(), name="SMA_" + str(ndays))
    return SMA.astype(float).round(2)


def BBANDS(df, n):
    MA = df.close.rolling(window=n).mean()
    SD = df.close.rolling(window=n).std()
    upperBand = MA + (2 * SD)
    lowerBand = MA - (2 * SD)
    return upperBand.astype(float).round(2), lowerBand.astype(float).round(2)


def RSI(df, periods=14):
    close_delta = df.close.diff()
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()

    _rsi = ma_up / ma_down
    return (100 - (100 / (1 + _rsi))).astype(float).round(2)


def MACD(df):
    k = df["close"].ewm(span=12, adjust=False, min_periods=12).mean()
    d = df["close"].ewm(span=26, adjust=False, min_periods=26).mean()
    macd = k - d
    macd_s = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    macd_h = macd - macd_s
    # return df.index.map(macd), df.index.map(macd_s), df.index.map(macd_h)
    return (
        macd.astype(float).round(2),
        macd_s.astype(float).round(2),
        macd_h.astype(float).round(2),
    )


def add_robust_features(df):
    df["feature_close"] = robust_scale(df.close.pct_change())
    df["feature_open"] = robust_scale(df.open / df.close)
    df["feature_high"] = robust_scale(df.high / df.close)
    df["feature_low"] = robust_scale(df.low / df.close)
    df["feature_volume"] = robust_scale(df.volume / df.volume.rolling(7 * 24).max())
    df.dropna(inplace=True)
    return df


def normalize(df):
    result = df.copy()
    columns = [x for x in df.columns if "feature" in x]
    for feature_name in columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


def robust(df):
    result = df.copy()
    columns = [x for x in df.columns if "feature" in x]
    for feature_name in columns:
        result[feature_name] = robust_scale(df[feature_name])
    return result


def stochastic_fast_k(df, n=5):
    fast_k = (
        (df.close - df.low.rolling(n).min())
        / (df.high.rolling(n).max() - df.low.rolling(n).min())
    ) * 100
    return fast_k


def stochastic_slow_k(fast_k, n=3):
    slow_k = fast_k.rolling(n).mean()
    return slow_k


def stochastic_slow_d(slow_k, n=3):
    slow_d = slow_k.rolling(n).mean()
    return slow_d


def preprocess(df):
    dataframe = pd.DataFrame(
        dict(
            feature_log_returns=np.log(df.close).diff().dropna(),
            feature_volume_lr=np.log(df.volume + 1e-8).diff().dropna(),
            open=df.open,
            high=df.high,
            low=df.low,
            close=df.close,
            volume=df.volume,
        )
    )

    dataframe["feature_volatility"] = dataframe["feature_log_returns"].rolling(
        window=30
    ).std() * np.sqrt(30)

    # df["feature_SMA_7"] = SMA(df, 7)
    # df["feature_SMA_25"] = SMA(df, 25)
    # df["feature_SMA_99"] = SMA(df, 99)
    # df["feature_MiddleBand"], df["feature_LowerBand"] = BBANDS(df, 21)
    # df["feature_MACD"], df["feature_MACD_S"], df["feature_MACD_H"] = MACD(df)

    # df_robust = robust(df)

    dataframe["feature_RSI_6"] = RSI(dataframe, periods=6) * 0.01
    # df_robust["feature_RSI_12"] = RSI(df, periods=12)
    # df_robust["feature_RSI_24"] = RSI(df, periods=24)

    dataframe["feature_RVOL"] = RVOL(dataframe)
    # dataframe["feature_VO"] = VO(dataframe)
    dataframe["feature_CMF"] = CMF(dataframe)
    # dataframe["feature_VWAP"] = VWAP(dataframe)
    # dataframe["feature_OBV"] = OBV(dataframe)

    return dataframe.fillna(0)


def only_sub_indicators(df):
    df["fast_k"] = stochastic_fast_k(df, 5)
    df["feature_slow_stochastic_k"] = stochastic_slow_k(df.fast_k, 3)
    df["feature_slow_stochastic_d"] = stochastic_slow_d(df.feature_slow_stochastic_k, 3)
    df["feature_OBV"] = OBV(df)
    df["feature_RSI_6"] = RSI(df, periods=6)
    df["feature_RSI_12"] = RSI(df, periods=12)
    df["feature_RSI_24"] = RSI(df, periods=24)
    df["feature_MACD"], df["feature_MACD_S"], df["feature_MACD_H"] = MACD(df)
    df = df.dropna()

    return df


def refined_pnl(history):
    total_roe = (
        history["portfolio_valuation", -1] / history["portfolio_valuation", 0] - 1
    )

    roe = (
        history["portfolio_valuation", -1] - history["entry_valuation", -1]
    ) / history["portfolio_valuation", 0]
    if abs(roe + total_roe) > 1:
        return 1 if roe + total_roe > 0 else -1

    return roe + total_roe


def RVOL(df, window=14):
    sma_volume = df.volume.rolling(window=window).mean()
    rvol = df.volume / sma_volume
    return rvol


def VO(df, short_window=12, long_window=26):
    short_sma = df.volume.rolling(window=short_window).mean()
    long_sma = df.volume.rolling(window=long_window).mean()
    volume_oscillator = (short_sma - long_sma) / long_sma * 100
    return np.tanh(volume_oscillator)


def CMF(df, window=20):
    mf_multiplier = ((df.close - df.low) - (df.high - df.close)) / (df.high - df.low)
    mf_volume = mf_multiplier * df.volume
    cmf = (
        mf_volume.rolling(window=window).sum() / df.volume.rolling(window=window).sum()
    )
    return cmf


def VWAP(df):
    vwap = (df.close * df.volume).cumsum() / df.volume.cumsum()
    return vwap


def OBV(df):
    direction = np.where(df.close.diff() > 0, 1, -1)
    direction[df.close.diff() == 0] = 0
    obv = (df.volume * direction).cumsum()
    return pd.Series(obv, index=df.index)
