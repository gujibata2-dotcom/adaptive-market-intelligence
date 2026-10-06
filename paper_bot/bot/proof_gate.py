"""
Live Gate V1
PROVE BEFORE TRADE

This is a hard safety gate. The current system contains no live-order executor.
Even if all checks pass, live trading remains disabled until a future explicit
code change adds a broker executor and separately enables it.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List


LIVE_TRADING_ENABLED = False


@dataclass(frozen=True)
class GateThresholds:
    min_paper_days: int = 60
    min_closed_trades: int = 50
    min_net_return_pct: float = 0.0
    max_drawdown_pct: float = 10.0
    min_profit_factor: float = 1.20
    min_regimes_tested: int = 3


DEFAULT_THRESHOLDS = GateThresholds()


def evaluate_live_gate(
    metrics: Dict[str, Any],
    thresholds: GateThresholds = DEFAULT_THRESHOLDS,
) -> Dict[str, Any]:
    checks = {
        "paper_days": metrics.get("paper_days", 0) >= thresholds.min_paper_days,
        "closed_trades": metrics.get("closed_trades", 0) >= thresholds.min_closed_trades,
        "net_return_positive": metrics.get("net_return_pct", -999) > thresholds.min_net_return_pct,
        "drawdown_controlled": abs(metrics.get("max_drawdown_pct", 999)) <= thresholds.max_drawdown_pct,
        "profit_factor": metrics.get("profit_factor", 0) >= thresholds.min_profit_factor,
        "out_of_sample": metrics.get("out_of_sample_pass", False) is True,
        "walk_forward": metrics.get("walk_forward_pass", False) is True,
        "fees_and_slippage_stress": metrics.get("fee_slippage_stress_pass", False) is True,
        "regime_coverage": metrics.get("regimes_tested", 0) >= thresholds.min_regimes_tested,
        "regime_robustness": metrics.get("regime_robustness_pass", False) is True,
        "prediction_ledger_locked": metrics.get("prediction_ledger_locked", False) is True,
    }

    failed: List[str] = [name for name, ok in checks.items() if not ok]
    evidence_pass = len(failed) == 0

    # Hard lock: evidence alone can never turn on real trading in V1.
    live_allowed = evidence_pass and LIVE_TRADING_ENABLED

    return {
        "evidence_pass": evidence_pass,
        "live_trading_enabled_in_code": LIVE_TRADING_ENABLED,
        "live_allowed": live_allowed,
        "decision": "LIVE_LOCKED" if not live_allowed else "LIVE_ALLOWED",
        "failed_checks": failed,
        "checks": checks,
        "thresholds": asdict(thresholds),
    }


if __name__ == "__main__":
    demo = {
        "paper_days": 0,
        "closed_trades": 0,
        "net_return_pct": 0,
        "max_drawdown_pct": 0,
        "profit_factor": 0,
        "out_of_sample_pass": False,
        "walk_forward_pass": False,
        "fee_slippage_stress_pass": False,
        "regimes_tested": 0,
        "regime_robustness_pass": False,
        "prediction_ledger_locked": False,
    }
    print(evaluate_live_gate(demo))
