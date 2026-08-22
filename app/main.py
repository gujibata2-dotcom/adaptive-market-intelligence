import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


SEED = 42


def create_dataset(n_samples=5000):
    rng = np.random.default_rng(SEED)

    returns = rng.normal(0, 0.01, n_samples)

    price = 100 * np.exp(np.cumsum(returns))

    volume = rng.lognormal(mean=10, sigma=0.5, size=n_samples)

    df = pd.DataFrame({
        "price": price,
        "volume": volume,
    })

    df["return_1"] = df["price"].pct_change()
    df["return_5"] = df["price"].pct_change(5)

    df["sma_10"] = df["price"].rolling(10).mean()
    df["sma_20"] = df["price"].rolling(20).mean()

    df["volatility_10"] = (
        df["return_1"].rolling(10).std()
    )

    # Target:
    # 1 = price higher after next period
    # 0 = price lower/equal
    df["future_return"] = df["price"].shift(-1) / df["price"] - 1

    df["target"] = (
        df["future_return"] > 0
    ).astype(int)

    df = df.dropna().reset_index(drop=True)

    return df


def run_experiment():

    print("=" * 60)
    print("ADAPTIVE MARKET INTELLIGENCE")
    print("V1 - XGBOOST BASELINE")
    print("PROVE BEFORE TRADE")
    print("=" * 60)

    df = create_dataset()

    features = [
        "price",
        "volume",
        "return_1",
        "return_5",
        "sma_10",
        "sma_20",
        "volatility_10",
    ]

    X = df[features]
    y = df["target"]

    # Time-based split
    train_size = int(len(df) * 0.70)

    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]

    X_test = X.iloc[train_size:]
    y_test = y.iloc[train_size:]

    print()
    print(f"Total samples : {len(df)}")
    print(f"Train samples : {len(X_train)}")
    print(f"Test samples  : {len(X_test)}")

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

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print()
    print("=" * 60)
    print("OUT-OF-SAMPLE RESULT")
    print("=" * 60)

    print(f"Accuracy: {accuracy * 100:.2f}%")

    results = pd.DataFrame({
        "actual": y_test.values,
        "prediction": predictions,
    })

    results.to_csv(
        "prediction_log_v1.csv",
        index=False
    )

    print()
    print("Prediction log saved:")
    print("prediction_log_v1.csv")

    print()
    print("PROVE BEFORE TRADE")


if __name__ == "__main__":
    run_experiment()
