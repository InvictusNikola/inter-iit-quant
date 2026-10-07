import numpy as np
import pandas as pd


def moving_average(df: pd.DataFrame, window: int):
    return df.close.rolling(window).mean()


def exponential_ma(df: pd.DataFrame, window: int):
    return df.close.ewm(span=window, adjust=False).mean()


def wilder(s: pd.Series, n: int):
    return s.ewm(alpha=1/n, adjust=False).mean()


def true_range(df: pd.DataFrame):
    pc = df.close.shift(1)
    return pd.concat(
        [df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1
    ).max(axis=1)


def atr(df: pd.DataFrame, smoothing_bars: int = 14):
    return wilder(true_range(df), smoothing_bars)


def adx(df: pd.DataFrame, smoothing_bars: int = 14):
    n = smoothing_bars
    up = df.high.diff()
    down = -df.low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)

    a = atr(df, n)
    plus_di = 100 * wilder(plus_dm, n) / a
    minus_di = 100 * wilder(minus_dm, n) / a
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)

    return pd.DataFrame({'plus_di': plus_di, 'minus_di': minus_di, 'adx': wilder(dx, n)})

def moving_average(df : pd.DataFrame , window : int):
    
   return df.close.rolling(window).mean()

def exponential_ma(df: pd.date_range , window : int):
    
    return df.close.ewm(span = window , adjust = False).mean()

def hurst_exponent(df, max_lag=300):
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(df[lag:], df[:-lag]))) for lag in lags]
    # Linear fit on log-log scale to get slope
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0


def variance_ratio(df: pd.DataFrame, tau: int):
    log_price = np.log(df['close'])

    r1 = log_price.diff()

    r_tau = log_price.diff(tau)


    var_ratio = r_tau.var() / (tau * r1.var())

    return var_ratio


def rolling_vr(df : pd.DataFrame, window : int):

    roll_vr = pd.Series(np.nan, df.index)

    for dt in df.index:
        roll_vr.loc[dt] = variance_ratio(df[:dt], window)

    return roll_vr
