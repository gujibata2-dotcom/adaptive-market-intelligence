"""
Feature Lab V2 — strict statistical pruning
PROVE BEFORE TRADE

Rules:
1) Feature must be significant in TRAIN using HAC-robust OLS.
2) TRAIN p-values are adjusted with Benjamini-Hochberg FDR.
3) Feature must also be significant OUT-OF-SAMPLE.
4) Coefficient direction must remain the same in train and test.
5) Otherwise the feature is rejected and excluded from selected_features.json.

This module does NOT place trades.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "experiments"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA_TRAIN_FDR = 0.05
ALPHA_OOS = 0.05
MIN_OBS = 120


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
    other[f"{prefix}_return_1"] = other["Close"].pct_change().shift(1)
    other = other[["Date", f"{prefix}_return_1"]]
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


def _hac_ols(df: pd.DataFrame, feature: str, target: str, maxlags: int):
    x = df[[feature, target]].dropna().copy()
    if len(x) < MIN_OBS or x[feature].nunique() < 10:
        return {
            "n": len(x),
            "coef": np.nan,
            "p_value": np.nan,
            "r2": np.nan,
        }

    X = sm.add_constant(x[[feature]], has_constant="add")
    model = sm.OLS(x[target], X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": maxlags},
    )
    return {
        "n": int(model.nobs),
        "coef": float(model.params[feature]),
        "p_value": float(model.pvalues[feature]),
        "r2": float(model.rsquared),
    }


def evaluate_features(
    df: pd.DataFrame,
    target: str = "future_return_1d",
    train_fraction: float = 0.70,
) -> pd.DataFrame:
    clean = df.dropna(subset=[target]).sort_values("Date").reset_index(drop=True)
    split = int(len(clean) * train_fraction)
    train = clean.iloc[:split].copy()
    test = clean.iloc[split:].copy()
    horizon = 7 if target.endswith("7d") else 1

    rows = []
    for feature in FEATURES:
        if feature not in clean.columns:
            continue

        tr = _hac_ols(train, feature, target, maxlags=horizon)
        te = _hac_ols(test, feature, target, maxlags=horizon)

        rows.append(
            {
                "feature": feature,
                "target": target,
                "train_n": tr["n"],
                "train_coef": tr["coef"],
                "train_p_raw": tr["p_value"],
                "train_r2": tr["r2"],
                "test_n": te["n"],
                "test_coef": te["coef"],
                "test_p": te["p_value"],
                "test_r2": te["r2"],
            }
        )

    scores = pd.DataFrame(rows)
    if scores.empty:
        return scores

    valid = scores["train_p_raw"].notna()
    scores["train_p_fdr"] = np.nan
    if valid.any():
        adjusted = multipletests(
            scores.loc[valid, "train_p_raw"].values,
            alpha=ALPHA_TRAIN_FDR,
            method="fdr_bh",
        )[1]
        scores.loc[valid, "train_p_fdr"] = adjusted

    scores["same_direction"] = (
        np.sign(scores["train_coef"]) == np.sign(scores["test_coef"])
    )
    scores["train_significant"] = scores["train_p_fdr"] < ALPHA_TRAIN_FDR
    scores["oos_significant"] = scores["test_p"] < ALPHA_OOS

    scores["keep"] = (
        scores["train_significant"]
        & scores["oos_significant"]
        & scores["same_direction"]
    )

    scores["status"] = np.where(
        scores["keep"],
        "KEEP",
        "REJECT",
    )

    def reason(row):
        failures = []
        if not bool(row["train_significant"]):
            failures.append("TRAIN_NOT_SIGNIFICANT_AFTER_FDR")
        if not bool(row["oos_significant"]):
            failures.append("OOS_NOT_SIGNIFICANT")
        if not bool(row["same_direction"]):
            failures.append("DIRECTION_UNSTABLE")
        return "PASS_ALL" if not failures else "|".join(failures)

    scores["reason"] = scores.apply(reason, axis=1)
    return scores.sort_values(
        ["keep", "test_p"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)


def select_model_features(scores: pd.DataFrame) -> dict:
    selected = {}
    for target in scores["target"].dropna().unique():
        selected[target] = (
            scores.loc[
                (scores["target"] == target) & (scores["keep"]),
                "feature",
            ]
            .drop_duplicates()
            .tolist()
        )
    return selected


def run_feature_lab() -> pd.DataFrame:
    print("=" * 78)
    print("FEATURE LAB V2 — STRICT SIGNIFICANCE PRUNING")
    print("PROVE BEFORE TRADE")
    print("=" * 78)

    df = build_feature_dataset()
    scores = pd.concat(
        [
            evaluate_features(df, target="future_return_1d"),
            evaluate_features(df, target="future_return_7d"),
        ],
        ignore_index=True,
    )

    selected = select_model_features(scores)
    rejected = scores.loc[~scores["keep"]].copy()

    scores.to_csv(OUT_DIR / "feature_lab_scores.csv", index=False)
    rejected.to_csv(OUT_DIR / "rejected_features.csv", index=False)
    with (OUT_DIR / "selected_features.json").open("w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)

    keep_cols = ["Date", "Close", "trend_regime", "vol_regime"] + [
        f for f in FEATURES if f in df.columns
    ]
    df[keep_cols].tail(500).to_csv(
        OUT_DIR / "feature_lab_latest_features.csv",
        index=False,
    )

    print()
    print(scores.to_string(index=False))
    print()
    print("SELECTED FEATURES")
    print(json.dumps(selected, ensure_ascii=False, indent=2))
    print()
    print("Rejected features are excluded from selected_features.json")
    print("No variable is kept merely because it is popular or intuitive.")
    return scores


if __name__ == "__main__":
    run_feature_lab()
