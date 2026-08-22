"""
Adaptive Market Intelligence
Data Layer V1

Purpose:
Collect and prepare real market data
for hypothesis testing.

PROVE BEFORE TRADE
"""

import yfinance as yf
import pandas as pd


def download_market_data(
    symbol="BTC-USD",
    start="2018-01-01",
    end=None
):
    print("=" * 60)
    print("ADAPTIVE MARKET INTELLIGENCE")
    print("DATA LAYER V1")
    print("PROVE BEFORE TRADE")
    print("=" * 60)

    print(f"Downloading: {symbol}")
    print(f"Start: {start}")
    print(f"End: {end}")

    data = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        raise ValueError("No market data received.")

    data = data.reset_index()

    # Flatten columns if necessary
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [
            col[0] if isinstance(col, tuple) else col
            for col in data.columns
        ]

    print()
    print("DATA SUMMARY")
    print("-" * 60)
    print(f"Rows : {len(data)}")
    print(f"From : {data['Date'].min()}")
    print(f"To   : {data['Date'].max()}")

    return data


if __name__ == "__main__":
    df = download_market_data()

    print()
    print(df.head())
