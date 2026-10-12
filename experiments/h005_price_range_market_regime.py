
"""
Adaptive Market Intelligence
H005 — PRICE RANGE × MARKET REGIME

Purpose:
Test whether price range is associated with future 7-day volatility,
and whether the relationship changes across trend and volatility regimes.

PROVE BEFORE TRADE
TRUTH BEFORE BELIEF
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

from app.data.market_data import download_market_data


def create_h005_dataset(raw=None):

    print("=" * 70)
    print("H005 — PRICE RANGE × MARKET REGIME")
    print("=" * 70)

    # ------------------------------------------------------------
    # 1. Download the SAME raw BTC-USD data used by Data Layer V1
    # ------------------------------------------------------------

    if raw is None:
        raw = download_market_data(
            symbol="BTC-USD", start="2018-01-01", end=None
        )

    df = raw.copy()

    # ------------------------------------------------------------
    # 2. Normalize price column
    # ------------------------------------------------------------

    df["price"] = pd.to_numeric(
        df["Close"],
        errors="coerce"
    )

    df["High"] = pd.to_numeric(
        df["High"],
        errors="coerce"
    )

    df["Low"] = pd.to_numeric(
        df["Low"],
        errors="coerce"
    )

    # ------------------------------------------------------------
    # 3. Daily return
    # ------------------------------------------------------------

    df["return_1"] = (
        df["price"].pct_change(fill_method=None)
    )

    # ------------------------------------------------------------
    # 4. PRICE RANGE
    #
    # Intraday high-low range normalized by closing price
    # ------------------------------------------------------------

    df["price_range"] = (
        (df["High"] - df["Low"])
        / df["price"]
    )

    # ------------------------------------------------------------
    # 5. Future 7-day volatility
    #
    # Volatility of the NEXT 7 daily returns.
    # shift(-1) prevents today's return from entering target.
    # ------------------------------------------------------------

    future_returns = df["return_1"].shift(-1)

    df["future_volatility_7d"] = (
        future_returns
        .rolling(7)
        .std()
        .shift(-6)
    )

    # ------------------------------------------------------------
    # 6. Trend regime
    #
    # Based on SMA relationship.
    #
    # price > SMA20  -> TREND_UP
    # price < SMA20  -> TREND_DOWN
    # otherwise      -> SIDEWAYS
    # ------------------------------------------------------------

    df["sma_20"] = (
        df["price"]
        .rolling(20)
        .mean()
    )

    df["trend_regime"] = np.select(
        [
            df["price"] > df["sma_20"],
            df["price"] < df["sma_20"],
        ],
        [
            "TREND_UP",
            "TREND_DOWN",
        ],
        default="SIDEWAYS"
    )

    # ------------------------------------------------------------
    # 7. Current volatility regime
    #
    # 10-day realized volatility.
    # Historical expanding median split = HIGH / LOW
    # ------------------------------------------------------------

    df["volatility_10"] = (
        df["return_1"]
        .rolling(10)
        .std()
    )

    # Only earlier observations may set today's regime threshold.
    volatility_median = df["volatility_10"].expanding(min_periods=20).median().shift(1)
    df["volatility_threshold"] = volatility_median

    df["volatility_regime"] = np.where(
        df["volatility_10"] >= volatility_median,
        "HIGH_VOLATILITY",
        "LOW_VOLATILITY"
    )

    # ------------------------------------------------------------
    # 8. Remove incomplete rows
    # ------------------------------------------------------------

    df = df[
        [
            "Date",
            "price",
            "High",
            "Low",
            "return_1",
            "price_range",
            "future_volatility_7d",
            "sma_20",
            "trend_regime",
            "volatility_10",
            "volatility_threshold",
            "volatility_regime",
        ]
    ]

    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    return df


def run_h005():

    df = create_h005_dataset()
    split = int(len(df) * 0.70)
    if split <= 7 or split >= len(df):
        raise ValueError("Not enough observations for purged 70/30 validation.")
    # Seven-day targets must end before the holdout starts.
    train = df.iloc[:split - 7]
    holdout = df.iloc[split:]

    print()
    print("=" * 70)
    print("H005 DATA SUMMARY")
    print("=" * 70)

    print(f"Rows: {len(df)}")
    print(f"From: {df['Date'].min()}")
    print(f"To:   {df['Date'].max()}")

    print()
    print("Trend regime:")
    print(df["trend_regime"].value_counts())

    print()
    print("Volatility regime:")
    print(df["volatility_regime"].value_counts())

    # ------------------------------------------------------------
    # H005.1
    # PRICE RANGE × TREND REGIME
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("H005.1 — PRICE RANGE × TREND REGIME")
    print("=" * 70)

    model_trend = smf.ols(
        """
        future_volatility_7d
        ~ price_range
        * C(trend_regime)
        """,
        data=train
    ).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 7}
    )

    print(model_trend.summary())

    # ------------------------------------------------------------
    # H005.2
    # PRICE RANGE × VOLATILITY REGIME
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("H005.2 — PRICE RANGE × VOLATILITY REGIME")
    print("=" * 70)

    model_volatility = smf.ols(
        """
        future_volatility_7d
        ~ price_range
        * C(volatility_regime)
        """,
        data=train
    ).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 7}
    )

    print(model_volatility.summary())

    from sklearn.metrics import mean_squared_error, r2_score

    validation_rows = []
    for name, model in [("H005.1", model_trend), ("H005.2", model_volatility)]:
        unseen = set(holdout["trend_regime" if name == "H005.1" else "volatility_regime"]) - set(train["trend_regime" if name == "H005.1" else "volatility_regime"])
        if unseen:
            raise ValueError(f"Holdout contains unseen regimes: {sorted(unseen)}")
        predicted = model.predict(holdout)
        actual = holdout["future_volatility_7d"]
        validation_rows.append({
            "experiment": name, "train_samples": len(train),
            "purged_samples": 7, "holdout_samples": len(holdout),
            "oos_r2": r2_score(actual, predicted),
            "oos_rmse": np.sqrt(mean_squared_error(actual, predicted)),
            "baseline_rmse": np.sqrt(mean_squared_error(actual, np.full(len(actual), train["future_volatility_7d"].mean()))),
        })
    validation = pd.DataFrame(validation_rows)
    print("Frozen-model holdout results:")
    print(validation.to_string(index=False))
    validation.to_csv(ROOT / "experiments" / "h005_validation.csv", index=False)

    # ------------------------------------------------------------
    # Save dataset used by H005
    # ------------------------------------------------------------

    output = (
        ROOT
        / "experiments"
        / "h005_dataset.csv"
    )

    df.to_csv(
        output,
        index=False
    )

    print()
    print("=" * 70)
    print("H005 DATASET SAVED")
    print("=" * 70)

    print(output)

    # ------------------------------------------------------------
    # Save model coefficients
    # ------------------------------------------------------------

    coef_output = (
        ROOT
        / "experiments"
        / "h005_results.csv"
    )

    rows = []

    for name, value in model_trend.params.items():
        rows.append({
            "experiment": "H005.1",
            "term": name,
            "coef": value,
            "p_value": model_trend.pvalues[name]
        })

    for name, value in model_volatility.params.items():
        rows.append({
            "experiment": "H005.2",
            "term": name,
            "coef": value,
            "p_value": model_volatility.pvalues[name]
        })

    pd.DataFrame(rows).to_csv(
        coef_output,
        index=False
    )

    print()
    print("Results saved:")
    print(coef_output)


if __name__ == "__main__":
    run_h005()
