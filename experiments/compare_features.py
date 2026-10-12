"""Fixed feature-set comparison plus training-only permutation diagnostics.
Existing history has been inspected before; this is research, not a fresh holdout.
"""
import sys
from pathlib import Path
import pandas as pd
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.dashboard_service import analyze, validate_market_data
from app.features import create_features, FEATURE_SETS
from app.paper_trading import simulate
from app.robustness import walk_forward
from app.main import SEED


def main():
    raw=pd.read_csv(ROOT/'experiments/paper_validation/btc_daily.csv')
    output=ROOT/'experiments/feature_validation'
    output.mkdir(exist_ok=True)
    summaries, windows=[],[]
    for name in ['legacy','normalized']:
        result=analyze(raw,feature_set=name)
        paper=simulate(result['data'],result['ledger'])
        summaries.append(dict(feature_set=name, start=result['ledger'].Date.min(),end=result['ledger'].Date.max(),
            test_samples=len(result['ledger']), accuracy=result['accuracy'],log_loss=result['log_loss'],
            baseline_accuracy=result['baseline_accuracy'],strategy_return=paper['total_return'],
            matched_buy_hold_return=paper['matched_benchmark_return'],max_drawdown=paper['max_drawdown'],
            closed_trades=paper['closed_trades']))
        folds,_=walk_forward(raw,feature_set=name)
        folds.insert(0,'feature_set',name)
        windows.append(folds)
    pd.DataFrame(summaries).to_csv(output/'comparison.csv',index=False)
    pd.concat(windows).to_csv(output/'walk_forward.csv',index=False)
    # Diagnose only inside the outer training portion. No automatic feature removal.
    df=create_features(validate_market_data(raw),'normalized')
    outer_train=df.iloc[:int(len(df)*.7)-1]
    split=int(len(outer_train)*.8)
    train,validation=outer_train.iloc[:split-1],outer_train.iloc[split:]
    features=FEATURE_SETS['normalized']
    model=XGBClassifier(n_estimators=150,max_depth=3,learning_rate=.05,subsample=.8,
        colsample_bytree=.8,random_state=SEED,eval_metric='logloss',n_jobs=2)
    model.fit(train[features],train.target)
    diagnostic=permutation_importance(model,validation[features],validation.target,
        scoring='neg_log_loss',n_repeats=10,random_state=SEED,n_jobs=1)
    pd.DataFrame({'feature':features,'log_loss_increase_mean':diagnostic.importances_mean,
        'permutation_std':diagnostic.importances_std,
        'validation_start':str(validation.Date.iloc[0]),'validation_end':str(validation.Date.iloc[-1])}).sort_values(
        'log_loss_increase_mean',ascending=False).to_csv(output/'training_permutation.csv',index=False)
    print(pd.DataFrame(summaries).to_string(index=False))
    print(pd.concat(windows).to_string(index=False))


if __name__=='__main__': main()
