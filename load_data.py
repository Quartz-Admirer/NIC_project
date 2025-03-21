import requests
import pandas as pd

def load_binance_data(symbol="BTCUSDT", interval="1d", limit=500):
    base_url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    response = requests.get(base_url, params=params)
    data = response.json()

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "trades",
        "taker_base_vol", "taker_quote_vol", "ignore"
    ])

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    return df

def prepare_data(df, ma_window=5):

    df = df.copy()
    df["ma_close"] = df["close"].rolling(window=ma_window).mean()
    df["future_close"] = df["close"].shift(-1)
    df.dropna(inplace=True)
    return df

def load_and_preprocess(symbol="BTCUSDT", interval="1d", limit=500, ma_window=5):

    df = load_binance_data(symbol, interval, limit)
    df = prepare_data(df, ma_window)
    return df

if __name__ == "__main__":
    data_df = load_and_preprocess("BTCUSDT", "1d", 100, 5)
    print(data_df.head())
