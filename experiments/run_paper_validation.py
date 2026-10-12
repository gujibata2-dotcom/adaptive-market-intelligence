"""Reproduce real-data paper results with the committed snapshot or fresh data."""
import argparse
import json
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.data.btc_sources import load_btc
from app.dashboard_service import analyze
from app.paper_trading import simulate
from app.robustness import walk_forward


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--csv', type=Path)
    parser.add_argument('--feature-set', choices=['legacy','normalized'], default='legacy')
    parser.add_argument('--output', type=Path, default=ROOT/'experiments'/'paper_validation')
    args=parser.parse_args()
    if args.csv:
        raw=pd.read_csv(args.csv)
        provider=str(args.csv)
    else:
        raw,provider=load_btc()
    result=analyze(raw, feature_set=args.feature_set)
    paper=simulate(result['data'],result['ledger'])
    folds,signals=walk_forward(result['data'], feature_set=args.feature_set)
    args.output.mkdir(parents=True,exist_ok=True)
    result['data'].to_csv(args.output/'btc_daily.csv',index=False)
    result['ledger'].to_csv(args.output/'predictions.csv',index=False)
    paper['trades'].to_csv(args.output/'trades.csv',index=False)
    paper['curve'].to_csv(args.output/'equity.csv',index=False)
    folds.to_csv(args.output/'walk_forward.csv',index=False)
    summary={'feature_set':args.feature_set,'source':provider,'start':str(result['data'].Date.min()),'end':str(result['data'].Date.max()),
        'initial_cash_usd':10000,'allocation':.25,'fee_rate':.001,'slippage':.001,
        'buy_threshold':.6,'sell_threshold':.4,'stop_loss':.1,
        'accuracy':result['accuracy'],'baseline_accuracy':result['baseline_accuracy'],
        **{k:v for k,v in paper.items() if k not in ['curve','trades']}}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    print(folds.to_string(index=False))


if __name__=='__main__': main()
