import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from statsmodels.tsa.stattools import coint
import src.indicators as indicators


def mean_reversion(df: pd.DataFrame, window: int = 14, entry: float = 2.0,
                   exit: float = 0.0, stoploss: float = 4.0) -> pd.Series:
    returns = df.close.pct_change()                         

    roll_mean = returns.rolling(window).mean().shift(1)
    roll_sig = returns.rolling(window).std().shift(1)

    standardized = (returns - roll_mean) / roll_sig

    signals = []
    prev = 0
    for z_scr in standardized:
        if pd.isna(z_scr):
            signals.append(prev)                            
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



def ma_crossover_better(df: pd.DataFrame, fast: int = 70, slow: int = 200, atr_period: int = 20, atr_k: float = 3):

    signals = pd.Series(0, index=df.index)

    fast_ma = indicators.moving_average(df, fast)
    slow_ma = indicators.moving_average(df, slow)
    adx = indicators.adx(df).adx
    atr_complete = indicators.atr(df, atr_period)
    
    prev = 0
    stoploss = 0.0
    highest = 0.0
    lowest = 0.0
    
    is_bull = fast_ma > slow_ma
    bull_cross = is_bull & (~is_bull.shift(1).fillna(False))
    
    is_bear = fast_ma < slow_ma
    bear_cross = is_bear & (~is_bear.shift(1).fillna(False))
    
    
    for dt in df.index:
        
        closing_price = df.close.loc[dt]
        high_price = df.high.loc[dt]
        low_price = df.low.loc[dt]
        atr = atr_complete.loc[dt]
        
        if prev == 1:
            closing_price = max(closing_price, high_price)
            stoploss = max(stoploss, closing_price - (atr_k * atr))
            
            if low_price <= stoploss:
                prev = 0  # Just update prev!
                
        elif prev == -1:
            closing_price = min(closing_price, low_price)
            stoploss = min(stoploss, closing_price + (atr_k * atr))
            
            if high_price >= stoploss:
                prev = 0  # Just update prev!

        elif prev == 0:
            if bear_cross.loc[dt] & (adx.loc[dt] > 25):
                prev = -1  # Just update prev!
                lowest = low_price
                stoploss = closing_price + (atr_k * atr)
                
            elif bull_cross.loc[dt] & (adx.loc[dt] > 25):  #stop loss was previosuly calc using high and low but I decided to use closing
                prev = 1   # Just update prev!
                highest = high_price
                stoploss = closing_price - (atr_k * atr)

        signals.loc[dt] = prev

    return signals.shift(1).fillna(0) 


def ma_cross_long(df: pd.DataFrame, fast: int = 50, slow: int = 200, atr_period: int = 20,
                  atr_k: float = 7.0, adx_min: float = 25, cooldown: int = 6):

    signals = pd.Series(0, index=df.index)

    fast_ma = indicators.moving_average(df, fast)
    slow_ma = indicators.moving_average(df, slow)
    adx = indicators.adx(df).adx
    atr_complete = indicators.atr(df, atr_period)

    is_bull = fast_ma > slow_ma

    prev = 0
    stoploss = 0.0
    wait = 0

    for dt in df.index:

        closing_price = df.close.loc[dt]
        high_price = df.high.loc[dt]
        low_price = df.low.loc[dt]
        atr = atr_complete.loc[dt]

        if prev == 1:
            if low_price <= stoploss or not is_bull.loc[dt]:
                prev = 0
                wait = cooldown

            else:
                stoploss = max(stoploss, high_price - (atr_k * atr))

        elif prev == 0:
            wait = max(wait - 1, 0)

            if wait == 0 and is_bull.loc[dt] and adx.loc[dt] > adx_min:
                prev = 1
                stoploss = closing_price - (atr_k * atr)

        signals.loc[dt] = prev

    return signals.shift(1).fillna(0)


def _garch(y):
    return arch_model(y, mean='Constant', vol='Garch', p=1, q=1, dist='t')


def garch_sigma(df: pd.DataFrame, test_start: str = '2023-01-01', step: int = 540) -> pd.Series:
    r = np.log(df.close).diff().dropna() * 100
    t_start = int((r.index >= pd.Timestamp(test_start, tz=r.index.tz)).argmax())
    sigma = pd.Series(np.nan, index=r.index)
    for t0 in range(t_start, len(r), step):
        t1 = min(t0 + step, len(r))
        p = _garch(r.iloc[:t0]).fit(disp='off').params
        cv = _garch(r.iloc[:t1]).fix(p).conditional_volatility
        sigma.iloc[t0:t1] = cv.iloc[t0:t1].values
    return sigma.reindex(df.index) / 100


def garch_size(df: pd.DataFrame, test_start: str = '2023-01-01', step: int = 540,
               cap: float = 1.0, bucket: float = 0.25, sigma: pd.Series = None) -> pd.Series:
    if sigma is None:
        sigma = garch_sigma(df, test_start, step)
    r = np.log(df.close).diff().dropna() * 100
    n_train = int((r.index < pd.Timestamp(test_start, tz=r.index.tz)).sum())
    target = _garch(r.iloc[:n_train]).fit(disp='off').conditional_volatility.median() / 100
    size = (target / sigma).clip(upper=cap)
    size = (size / bucket).round() * bucket
    return size.fillna(0)


def ma_cross_long_garch(df: pd.DataFrame, test_start: str = '2023-01-01', step: int = 540,
                        cap: float = 1.0, bucket: float = 0.25, **kw) -> pd.Series:
    return ma_cross_long(df, **kw) * garch_size(df, test_start, step, cap, bucket)

def garch_sigma_v2(df: pd.DataFrame, test_start: str = '2023-01-01', step: int = 90 * 6,
                   window: int = 365 * 6 * 2) -> pd.Series:
    r = np.log(df.close).diff().dropna() * 100
    t_start = int((r.index >= pd.Timestamp(test_start, tz=r.index.tz)).argmax())
    sigma = pd.Series(np.nan, index=r.index)
    for t0 in range(t_start, len(r), step):                 # refit every 90 days
        t1 = min(t0 + step, len(r))
        lo = max(t0 - window, 0)                            # rolling window, no look-ahead
        p = _garch(r.iloc[lo:t0]).fit(disp='off').params
        cv = _garch(r.iloc[lo:t1]).fix(p).conditional_volatility
        sigma.iloc[t0:t1] = cv.iloc[t0 - lo:].values
    return sigma.reindex(df.index) / 100


def garch_size_v2(df: pd.DataFrame, test_start: str = '2023-01-01', step: int = 90 * 6,
                  cap: float = 1.0, bucket: float = 0.25, sigma: pd.Series = None,
                  window: int = 365 * 6 * 2) -> pd.Series:
    if sigma is None:
        sigma = garch_sigma_v2(df, test_start, step, window)
    r = np.log(df.close).diff().dropna() * 100
    n_train = int((r.index < pd.Timestamp(test_start, tz=r.index.tz)).sum())
    target = _garch(r.iloc[max(n_train - window, 0):n_train]).fit(disp='off').conditional_volatility.median() / 100
    size = (target / sigma).clip(upper=cap)
    size = (size / bucket).round() * bucket
    return size.fillna(0)


def ma_cross_long_garch_v2(df: pd.DataFrame, test_start: str = '2023-01-01', step: int = 90 * 6,
                           cap: float = 1.0, bucket: float = 0.25, spike_q: float = 0.9,
                           spike_window: int = 90 * 6, window: int = 365 * 6 * 2, **kw) -> pd.Series:
    sigma = garch_sigma_v2(df, test_start, step, window)
    size = garch_size_v2(df, test_start, step, cap, bucket, sigma, window)

    # kill switch: flat when forecast vol is in the top (1 - spike_q) of the last 90 days
    spike_lvl = sigma.rolling(spike_window, min_periods=spike_window // 2).quantile(spike_q)
    calm = ~(sigma > spike_lvl)                             # NaN warm-up -> filter off

    return ma_cross_long(df, **kw) * size * calm


def pairs_fit(df_y: pd.DataFrame, df_x: pd.DataFrame, test_start: str = '2022-01-01',
              window: int = 365 * 6, step: int = 90 * 6, p_max: float = 0.10):
    y = np.log(df_y.close)
    x = np.log(df_x.close)
    y, x = y.align(x, join='inner')
    t_start = max(int((y.index >= pd.Timestamp(test_start, tz=y.index.tz)).argmax()), window)
    z = pd.Series(np.nan, index=y.index)
    beta = pd.Series(np.nan, index=y.index)
    gate = pd.Series(False, index=y.index)
    for t0 in range(t_start, len(y), step):                 # refit every 90 days
        t1 = min(t0 + step, len(y))
        yy, xx = y.iloc[t0 - window:t0], x.iloc[t0 - window:t0]      # past data only
        ols = sm.OLS(yy, sm.add_constant(xx)).fit()
        a, b = ols.params.iloc[0], ols.params.iloc[1]
        sd = ols.resid.std()
        p = coint(yy, xx, trend='c')[1]                     # Engle-Granger on the fit window
        z.iloc[t0:t1] = ((y.iloc[t0:t1] - a - b * x.iloc[t0:t1]) / sd).values   # params frozen
        beta.iloc[t0:t1] = b
        gate.iloc[t0:t1] = p < p_max
    return z, beta, gate


def pairs_reversion(df_y: pd.DataFrame, df_x: pd.DataFrame, test_start: str = '2022-01-01',
                    window: int = 365 * 6, step: int = 90 * 6, p_max: float = 0.10,
                    entry: float = 2.0, exit: float = 0.0, stoploss: float = 4.0,
                    fit: tuple = None) -> pd.Series:
    """+1 = long spread (long y, short beta * x), -1 = short spread. Position held DURING each bar."""
    if fit is None:
        fit = pairs_fit(df_y, df_x, test_start, window, step, p_max)
    z, beta, gate = fit

    signals = []
    prev = 0
    armed = False                                           # re-enter only after z is back inside the band
    for z_scr, ok in zip(z, gate):
        if pd.isna(z_scr) or not ok:
            prev = 0
            armed = False

        elif prev == 0:
            if abs(z_scr) < entry:
                armed = True
            elif armed:
                prev = 1 if z_scr < -entry else -1

        elif prev == 1:
            if z_scr >= exit:
                prev = 0
            elif z_scr < -stoploss:
                prev = 0
                armed = False

        elif prev == -1:
            if z_scr <= -exit:
                prev = 0
            elif z_scr > stoploss:
                prev = 0
                armed = False

        signals.append(prev)

    signals = pd.Series(signals, index=z.index)
    return signals.shift(1).fillna(0)                       



def pairs_spread(df_y: pd.DataFrame, df_x: pd.DataFrame, beta: pd.Series) -> pd.DataFrame:
    """Synthetic price of the spread (long y, short beta * x) per unit of gross capital, for bactester."""
    ret_y = df_y.close.pct_change()
    ret_x = df_x.close.pct_change()
    ret = (ret_y - beta * ret_x) / (1 + beta.abs())         
    ret = ret.reindex(beta.index).fillna(0)
    return pd.DataFrame({'close': (1 + ret).cumprod()})