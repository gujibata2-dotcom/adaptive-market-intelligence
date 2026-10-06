"""
Feature Lab V1
PROVE BEFORE TRADE

Purpose:
- Build candidate features from BTC and cross-market data.
- Evaluate each feature on a chronological train/test split.
- Reject unstable features instead of assuming popular indicators have edge.
- Save an evidence table for later model/router use.

This module does NOT place trades.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "experiments"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _download(symbol: str, start: str = "2018-01-01") -> pd.DataFrame:
    df = yf.download(
        symbol,
        start=start,
        auto_adjust=False,
        progress=False,
    )
    if df.empty:
        raise RuntimeError(f"No data received for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def _merge_market(base: pd.DataFrame, symbol: str, prefix: str) -> pd.DataFrame:
    other = _download(symbol)[["Date", "Close"]].copy()
    other[f"{prefix}_return_1"] = other["Close"].pct_change()
    other = other[["Date", f"{prefix}_return_1"]]
    # Backward as-of merge: never use future external-market values.
    return pd.merge_asof(
        base.sort_values("Date"),
        other.sort_values("Date"),
        on="Date",
        direction="backward",
        tolerance=pd.Timedelta("3D"),
    )


def build_feature_dataset(start: str = "2018-01-01") -> pd.DataFrame:
    btc = _download("BTC-USD", start=start)
    df = btc.copy()

    close = pd.to_numeric(df["Close"], errors="coerce")
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    volume = pd.to_numeric(df["Volume"], errors="coerce")

    df["return_1"] = close.pct_change()
    df["return_3"] = close.pct_change(3)
    df["return_7"] = close.pct_change(7)

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    df["sma20_gap"] = close / sma20 - 1
    df["sma50_gap"] = close / sma50 - 1

    df["rsi_14"] = _rsi(close, 14) / 100.0
    df["atr_14_norm"] = _atr(df, 14) / close
    df["price_range"] = (high - low) / close

    df["volatility_7"] = df["return_1"].rolling(7).std()
    df["volatility_30"] = df["return_1"].rolling(30).std()

    df["volume_change"] = volume.pct_change()
    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std()
    df["volume_z20"] = (volume - vol_mean) / vol_std.replace(0, np.nan)

    # Cross-market context. These are context inputs, not assumed predictors.
    for symbol, prefix in [
        ("ETH-USD", "eth"),
        ("^IXIC", "nasdaq"),
        ("DX-Y.NYB", "dxy"),
        ("^VIX", "vix"),
    ]:
        try:
            df = _merge_market(df, symbol, prefix)
        except Exception:
            df[f"{prefix}_return_1"] = np.nan

    df["future_return_1d"] = close.shift(-1) / close - 1
    df["future_return_7d"] = close.shift(-7) / close - 1

    df["trend_regime"] = np.where(
        df["sma20_gap"] > 0.02,
        "TREND_UP",
        np.where(df["sma20_gap"] < -0.02, "TREND_DOWN", "SIDEWAYS"),
    )

    vol_median = df["volatility_30"].expanding(min_periods=90).median()
    df["vol_regime"] = np.where(
        df["volatility_30"] >= vol_median,
        "HIGH_VOL",
        "LOW_VOL",
    )

    return df.replace([np.inf, -np.inf], np.nan)


FEATURES = [
    "return_1",
    "return_3",
    "return_7",
    "sma20_gap",
    "sma50_gap",
    "rsi_14",
    "atr_14_norm",
    "price_range",
    "volatility_7",
    "volatility_30",
    "volume_change",
    "volume_z20",
    "eth_return_1",
    "nasdaq_return_1",
    "dxy_return_1",
    "vix_return_1",
]


def _rank_corr(a: pd.Series, b: pd.Series) -> float:
    x = pd.concat([a, b], axis=1).dropna()
    if len(x) < 30:
        return np.nan
    return float(x.iloc[:, 0].rank().corr(x.iloc[:, 1].rank()))


def _quintile_spread(df: pd.DataFrame, feature: str, target: str) -> float:
    x = df[[feature, target]].dropna().copy()
    if len(x) < 100 or x[feature].nunique() < 5:
        return np.nan
    try:
        q = pd.qcut(x[feature], 5, labels=False, duplicates="drop")
        if q.nunique() < 5:
            return np.nan
        means = x.groupby(q, observed=True)[target].mean()
        return float(means.iloc[-1] - means.iloc[0])
    except ValueError:
        return np.nan


def evaluate_features(
    df: pd.DataFrame,
    target: str = "future_return_1d",
    train_fraction: float = 0.70,
) -> pd.DataFrame:
    clean = df.dropna(subset=[target]).reset_index(drop=True)
    split = int(len(clean) * train_fraction)
    train = clean.iloc[:split]
    test = clean.iloc[split:]

    rows = []
    for feature in FEATURES:
        if feature not in clean.columns:
            continue

        train_corr = _rank_corr(train[feature], train[target])
        test_corr = _rank_corr(test[feature], test[target])
        train_spread = _quintile_spread(train, feature, target)
        test_spread = _quintile_spread(test, feature, target)

        same_corr_sign = (
            pd.notna(train_corr)
            and pd.notna(test_corr)
            and np.sign(train_corr) == np.sign(test_corr)
        )
        same_spread_sign = (
            pd.notna(train_spread)
            and pd.notna(test_spread)
            and np.sign(train_spread) == np.sign(test_spread)
        )

        # Deliberately conservative: "CANDIDATE" means worth more testing,
        # not proven trading edge.
        candidate = bool(
            same_corr_sign
            and same_spread_sign
            and abs(test_corr) >= 0.03
            and abs(test_spread) >= 0.001
        )

        rows.append(
            {
                "feature": feature,
                "target": target,
                "train_rank_corr": train_corr,
                "test_rank_corr": test_corr,
                "train_quintile_spread": train_spread,
                "test_quintile_spread": test_spread,
                "same_corr_sign": same_corr_sign,
                "same_spread_sign": same_spread_sign,
                "status": "CANDIDATE" if candidate else "REJECT_OR_RETEST",
            }
        )

    scores = pd.DataFrame(rows)
    if not scores.empty:
        scores["test_strength"] = scores["test_rank_corr"].abs()
        scores = scores.sort_values(
            ["status", "test_strength"],
            ascending=[True, False],
        ).reset_index(drop=True)
    return scores


def run_feature_lab() -> pd.DataFrame:
    print("=" * 70)
    print("FEATURE LAB V1 — PROVE BEFORE TRADE")
    print("=" * 70)

    df = build_feature_dataset()
    scores_1d = evaluate_features(df, target="future_return_1d")
    scores_7d = evaluate_features(df, target="future_return_7d")

    scores = pd.concat([scores_1d, scores_7d], ignore_index=True)
    scores.to_csv(OUT_DIR / "feature_lab_scores.csv", index=False)

    keep = ["Date", "Close", "trend_regime", "vol_regime"] + [
        f for f in FEATURES if f in df.columns
    ]
    df[keep].tail(500).to_csv(
        OUT_DIR / "feature_lab_latest_features.csv",
        index=False,
    )

    print()
    print("Feature evidence saved:")
    print(OUT_DIR / "feature_lab_scores.csv")
    print()
    print(scores.to_string(index=False))
    print()
    print("CANDIDATE != PROVEN EDGE. Candidates must still pass walk-forward")
    print("and paper-trading proof gates before any live trading.")
    return scores


if __name__ == "__main__":
    run_feature_lab()
