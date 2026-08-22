import numpy as np
import pandas as pd
import yfinance as yf

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


SEED = 42


def load_market_data():

    print("Downloading BTC historical data...")

    df = yf.download(
        "BTC-USD",
        period="5y",
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        raise RuntimeError("No market data received.")

    # Handle possible MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    return df


def create_features(df):

    df = df.copy()

    df["return_1"] = df["Close"].pct_change()

    df["return_3"] = df["Close"].pct_change(3)

    df["return_7"] = df["Close"].pct_change(7)

    df["sma_7"] = (
        df["Close"]
        .rolling(7)
        .mean()
    )

    df["sma_20"] = (
        df["Close"]
        .rolling(20)
        .mean()
    )

    df["sma_50"] = (
        df["Close"]
        .rolling(50)
        .mean()
    )

    df["volatility_7"] = (
        df["return_1"]
        .rolling(7)
        .std()
    )

    df["volume_change"] = (
        df["Volume"].pct_change()
    )

    # Future return
    df["future_return"] = (
        df["Close"].shift(-1)
        / df["Close"]
        - 1
    )

    # Target
    # 1 = price increases next day
    # 0 = price decreases or stays flat
    df["target"] = (
        df["future_return"] > 0
    ).astype(int)

    df = df.dropna().reset_index(drop=True)

    return df


def run_experiment():

    print("=" * 60)
    print("ADAPTIVE MARKET INTELLIGENCE")
    print("V1.1 - REAL MARKET DATA")
    print("XGBOOST")
    print("PROVE BEFORE TRADE")
    print("=" * 60)

    df = load_market_data()

    df = create_features(df)

    features = [
        "return_1",
        "return_3",
        "return_7",
        "sma_7",
        "sma_20",
        "sma_50",
        "volatility_7",
        "volume_change",
    ]

    X = df[features]
    y = df["target"]

    # Time-based split
    # NEVER shuffle financial time series
    train_size = int(len(df) * 0.70)

    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]

    X_test = X.iloc[train_size:]
    y_test = y.iloc[train_size:]

    print()
    print(f"Total samples : {len(df)}")
    print(f"Train samples : {len(X_train)}")
    print(f"Test samples  : {len(X_test)}")

    print()
    print(
        f"Train period : "
        f"{df['Date'].iloc[0]} "
        f"→ "
        f"{df['Date'].iloc[train_size - 1]}"
    )

    print(
        f"Test period  : "
        f"{df['Date'].iloc[train_size]} "
        f"→ "
        f"{df['Date'].iloc[-1]}"
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        eval_metric="logloss",
    )

    print()
    print("Training XGBoost...")

    model.fit(
        X_train,
        y_train
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print()
    print("=" * 60)
    print("OUT-OF-SAMPLE RESULT")
    print("=" * 60)

    print(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    # Prediction Ledger
    results = df.iloc[
        train_size:
    ][[
        "Date",
        "Close"
    ]].copy()

    results["prediction"] = predictions

    results["confidence"] = np.where(
        probabilities >= 0.5,
        probabilities,
        1 - probabilities
    )

    results["actual"] = y_test.values

    results["correct"] = (
        results["prediction"]
        == results["actual"]
    )

    results["future_return"] = (
        df.iloc[
            train_size:
        ]["future_return"].values
    )

    results.to_csv(
        "prediction_ledger_v1_1.csv",
        index=False
    )

    print()
    print(
        "Prediction ledger saved:"
    )

    print(
        "prediction_ledger_v1_1.csv"
    )

    print()
    print("PROVE BEFORE TRADE")


if __name__ == "__main__":
    run_experiment()
