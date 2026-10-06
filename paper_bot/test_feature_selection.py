import numpy as np
import pandas as pd
from bot.feature_lab import evaluate_features, select_model_features

rng = np.random.default_rng(42)
n = 1200
dates = pd.date_range("2020-01-01", periods=n, freq="D")
signal = rng.normal(size=n)
noise = rng.normal(size=n)
eps = rng.normal(scale=0.6, size=n)

df = pd.DataFrame({
    "Date": dates,
    "return_1": signal,
    "return_3": noise,
    "future_return_1d": 0.35 * signal + eps,
})

# Supply unused candidate columns as NaN so the evaluator can skip/fail them safely.
for c in [
    "return_7","sma20_gap","sma50_gap","rsi_14","atr_14_norm",
    "price_range","volatility_7","volatility_30","volume_change",
    "volume_z20","eth_return_1","nasdaq_return_1","dxy_return_1","vix_return_1"
]:
    df[c] = np.nan

scores = evaluate_features(df, target="future_return_1d")
selected = select_model_features(scores)["future_return_1d"]

assert "return_1" in selected, scores.to_string()
assert "return_3" not in selected, scores.to_string()
print("STRICT FEATURE SELECTION TEST PASSED")
