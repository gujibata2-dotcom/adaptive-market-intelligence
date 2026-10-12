"""Validated, offline-testable dashboard analysis."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from xgboost import XGBClassifier
from app.main import create_features, SEED

FEATURES = ["return_1", "return_3", "return_7", "sma_7", "sma_20", "sma_50", "volatility_7", "volume_change"]


def validate_market_data(raw):
    required = ["Date", "Close", "High", "Low", "Volume"]
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError("CSV ขาดคอลัมน์: " + ", ".join(sorted(missing)))
    df = raw[required].copy()
    df["Date"] = pd.to_datetime(df.Date, errors="coerce", utc=True).dt.tz_convert(None)
    for col in required[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df.isna().any().any() or not np.isfinite(df[required[1:]]).all().all():
        raise ValueError("พบวันที่หรือค่าตัวเลขไม่ถูกต้อง กรุณาตรวจ CSV")
    if df.Date.duplicated().any():
        raise ValueError("พบวันที่ซ้ำ กรุณาใช้ข้อมูลหนึ่งแถวต่อวัน")
    if (df[["Close", "High", "Low"]] <= 0).any().any() or (df.Volume < 0).any():
        raise ValueError("ราคาต้องมากกว่า 0 และ Volume ต้องไม่ติดลบ")
    if ((df.High < df.Low) | (df.Close > df.High) | (df.Close < df.Low)).any():
        raise ValueError("High/Low ไม่สอดคล้องกับ Close")
    df = df.sort_values("Date").reset_index(drop=True)
    if not df.Date.dt.normalize().diff().iloc[1:].eq(pd.Timedelta(days=1)).all():
        raise ValueError("ใช้ข้อมูลรายวันต่อเนื่องเท่านั้น: พบวันที่ขาดหายหรือมากกว่าหนึ่งแถวต่อวัน")
    if len(df) < 180:
        raise ValueError("ต้องมีข้อมูลอย่างน้อย 180 วัน")
    return df


def demo_data():
    rng = np.random.default_rng(SEED)
    close = 100 * np.exp(np.cumsum(rng.normal(0, .02, 400)))
    return pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=400),
        "Close": close, "High": close * 1.025, "Low": close * .975,
        "Volume": rng.integers(1000, 10000, 400)})


def analyze(raw):
    raw = validate_market_data(raw)
    df = create_features(raw)
    if len(df) < 100:
        raise ValueError("ข้อมูลสมบูรณ์ไม่พอหลังสร้างตัวแปร ตรวจสอบ Volume และข้อมูลราคา")
    split = int(len(df) * .7)
    train, test = df.iloc[:split - 1], df.iloc[split:]
    if train.target.nunique() < 2:
        raise ValueError("ชุดฝึกต้องมีทั้งวันที่ราคาขึ้นและไม่ขึ้น")
    model = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=.05,
        subsample=.8, colsample_bytree=.8, random_state=SEED, eval_metric="logloss", n_jobs=2)
    model.fit(train[FEATURES], train.target)
    probabilities = model.predict_proba(test[FEATURES])[:, 1]
    predictions = (probabilities >= .5).astype(int)
    baseline = int(train.target.mean() >= .5)
    ledger = test[["Date", "Close", "future_return"]].copy()
    ledger["probability_up"] = probabilities
    ledger["prediction"] = predictions
    ledger["actual"] = test.target
    ledger["correct"] = ledger.prediction == ledger.actual
    return {"data": raw, "ledger": ledger, "train_samples": len(train),
        "train_end": train.Date.iloc[-1], "test_start": test.Date.iloc[0],
        "accuracy": accuracy_score(test.target, predictions),
        "balanced_accuracy": balanced_accuracy_score(test.target, predictions),
        "baseline_accuracy": accuracy_score(test.target, np.full(len(test), baseline)),
        "log_loss": log_loss(test.target, probabilities, labels=[0, 1])}
