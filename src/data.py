import ccxt
import pandas as pd
import time
import os

def fetch_ohlcv_full(symbol, timeframe="4h", since="2021-01-01T00:00:00Z", until="2026-01-01T00:00:00Z"):
    ex = ccxt.binance({"enableRateLimit": True})
    since_ms = ex.parse8601(since)
    until_ms = ex.parse8601(until)
    rows = []
    while since_ms < until_ms:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        since_ms = batch[-1][0] + 1
        time.sleep(ex.rateLimit / 1000)
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").set_index("timestamp")
    return df[df.index < pd.to_datetime(until, utc=True)]

def load_ohlcv(symbol, timeframe="4h", since="2021-01-01T00:00:00Z", until="2026-01-01T00:00:00Z", refresh=False):
    os.makedirs("data", exist_ok=True)
    path = f"data/{symbol.replace('/', '_')}_{timeframe}_{since[:10]}_{until[:10]}.parquet"
    if os.path.exists(path) and not refresh:
        return pd.read_parquet(path)
    df = fetch_ohlcv_full(symbol, timeframe, since, until)
    df.to_parquet(path)
    return df


def clean_and_validate_crypto(df, ticker_label="BTC", freq="4h"):

    df = df.copy()

    # Flatten MultiIndex columns if yfinance returns (Price, Ticker)

    df.columns = df.columns.get_level_values(0)

    # 2. Ensure consistent UTC Timestamps & Sort
    if df.index.tz is None:
        df.index = pd.to_datetime(df.index).tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df = df.sort_index()


    zero_vol_mask = df["volume"] == 0
    zero_vol_count = zero_vol_mask.sum()


    returns = df["close"].pct_change()

    invalid_ohlc = df[(df["low"] > df["high"]) | (df["close"] > df["high"]) | (df["close"] < df["low"])]

    # Display Summary Report
    print(f"=== Data Cleaning Report: {ticker_label} ===")
    print(f"Zero-Volume Bars Found:     {zero_vol_count}")
    print(f"Invalid OHLC Bars Found:    {len(invalid_ohlc)}")
    print("-" * 42)

    return df, {"invalid_ohlc": invalid_ohlc}
