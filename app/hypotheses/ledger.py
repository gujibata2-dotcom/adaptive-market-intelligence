"""
Adaptive Market Intelligence
Hypothesis Ledger

PROVE BEFORE TRADE
TRUTH BEFORE BELIEF
"""

from datetime import datetime
import pandas as pd
from pathlib import Path


LEDGER_PATH = Path("experiments/hypothesis_ledger.csv")


def record_hypothesis(
    hypothesis_id,
    hypothesis,
    sample_size,
    method,
    result,
    p_value=None,
    effect_size=None,
    status="INCONCLUSIVE"
):
    LEDGER_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    record = pd.DataFrame([{
        "timestamp": datetime.utcnow().isoformat(),
        "hypothesis_id": hypothesis_id,
        "hypothesis": hypothesis,
        "sample_size": sample_size,
        "method": method,
        "result": result,
        "p_value": p_value,
        "effect_size": effect_size,
        "status": status
    }])

    if LEDGER_PATH.exists():
        record.to_csv(
            LEDGER_PATH,
            mode="a",
            header=False,
            index=False
        )
    else:
        record.to_csv(
            LEDGER_PATH,
            index=False
        )

    print(f"Hypothesis {hypothesis_id} recorded.")
