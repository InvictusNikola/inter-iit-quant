import numpy as np
import pandas as pd

import src.indicators as indicators


def mean_reversion(df: pd.DataFrame, window: int = 14, entry: float = 2.0,
                   exit: float = 0.0, stoploss: float = 4.0) -> pd.Series:
    """Fade extreme daily returns. Returns the position (-1/0/+1) held DURING each bar."""
    returns = df.close.pct_change()                         # local, don't touch caller's df

    roll_mean = returns.rolling(window).mean().shift(1)
    roll_sig = returns.rolling(window).std().shift(1)

    standardized = (returns - roll_mean) / roll_sig

    signals = []
    prev = 0
    for z_scr in standardized:
        if pd.isna(z_scr):
            signals.append(prev)                            # warm-up: stay as is (flat)
            continue

        if prev == 0:
            if z_scr > entry:
                prev = -1
            elif z_scr < -entry:
                prev = 1

        elif prev == 1:
            if z_scr > entry:
                prev = -1
            elif z_scr >= exit:
                prev = 0
            elif z_scr < -stoploss:
                prev = 0

        elif prev == -1:
            if z_scr < -entry:
                prev = 1
            elif z_scr <= -exit:
                prev = 0
            elif z_scr > stoploss:
                prev = 0

        signals.append(prev)

    signals = pd.Series(signals, index=df.index)
    return signals.shift(1).fillna(0)                       # act on the NEXT bar


def ma_crossover(df : pd.DataFrame, fast:int = 30, slow :int = 90):

    signals = pd.Series(0, df.index)

    fast_ma = indicators.moving_average(df, fast)
    slow_ma = indicators.moving_average(df, slow)

    adx = indicators.adx(df).adx

    prev = 0

    for dt in df.index:

        if (slow_ma.loc[dt] > fast_ma.loc[dt]) & ( adx.loc[dt]>25):
            signals.loc[dt]= prev = -1

        elif (slow_ma.loc[dt] < fast_ma.loc[dt]) & ( adx.loc[dt]>25):
            signals.loc[dt]= prev = 1


    return signals.shift(1).fillna(0)
