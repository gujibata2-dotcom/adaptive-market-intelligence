"""Chronological expanding-window evaluation; past-only refits and fixed settings."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from app.dashboard_service import validate_market_data, FEATURES
from app.main import create_features, SEED
from app.paper_trading import simulate


def walk_forward(raw, folds=3, **paper_settings):
    raw = validate_market_data(raw)
    df = create_features(raw)
    first_test = max(100, len(df)//2)
    if len(df) - first_test < folds * 20:
        raise ValueError('ต้องมีข้อมูลมากขึ้นเพื่อทดสอบหลายช่วง (อย่างน้อย 20 วันต่อช่วง)')
    rows, ledgers = [], []
    boundaries = np.linspace(first_test, len(df), folds+1, dtype=int)
    for fold, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]), 1):
        train, test = df.iloc[:start-1], df.iloc[start:end]
        if train.target.nunique() < 2:
            raise ValueError('ชุดฝึกต้องมีทั้งขึ้นและไม่ขึ้น')
        model = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=.05,
            subsample=.8, colsample_bytree=.8, random_state=SEED, eval_metric='logloss', n_jobs=2)
        model.fit(train[FEATURES], train.target)
        ledger = test[['Date','Close']].copy()
        ledger['probability_up'] = model.predict_proba(test[FEATURES])[:,1]
        ledger['fold'] = fold
        ledgers.append(ledger)
        next_date = raw.loc[raw.Date > test.Date.iloc[-1], 'Date'].iloc[0]
        prices = raw.loc[(raw.Date >= test.Date.iloc[0]) & (raw.Date <= next_date)]
        paper = simulate(prices, ledger, **paper_settings)
        market_return = float(prices.Close.iloc[-1]/prices.Close.iloc[0]-1)
        rows.append(dict(fold=fold, train_end=train.Date.iloc[-1], start=test.Date.iloc[0],
            end=test.Date.iloc[-1], test_samples=len(test),
            market_regime='UP' if market_return > .05 else 'DOWN' if market_return < -.05 else 'SIDEWAYS',
            market_return=market_return, accuracy=accuracy_score(test.target, ledger.probability_up >= .5),
            strategy_return=paper['total_return'], matched_buy_hold_return=paper['matched_benchmark_return'],
            max_drawdown=paper['max_drawdown'], closed_trades=paper['closed_trades']))
    return pd.DataFrame(rows), pd.concat(ledgers, ignore_index=True)
