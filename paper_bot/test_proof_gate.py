from bot.proof_gate import evaluate_live_gate, LIVE_TRADING_ENABLED


def test_gate_fails_with_no_evidence():
    result = evaluate_live_gate({})
    assert result["evidence_pass"] is False
    assert result["live_allowed"] is False
    assert result["decision"] == "LIVE_LOCKED"


def test_even_strong_evidence_cannot_enable_live_in_v1():
    strong = {
        "paper_days": 120,
        "closed_trades": 200,
        "net_return_pct": 12.0,
        "max_drawdown_pct": 5.0,
        "profit_factor": 1.8,
        "out_of_sample_pass": True,
        "walk_forward_pass": True,
        "fee_slippage_stress_pass": True,
        "regimes_tested": 4,
        "regime_robustness_pass": True,
        "prediction_ledger_locked": True,
    }
    result = evaluate_live_gate(strong)
    assert result["evidence_pass"] is True
    assert LIVE_TRADING_ENABLED is False
    assert result["live_allowed"] is False
    assert result["decision"] == "LIVE_LOCKED"


if __name__ == "__main__":
    test_gate_fails_with_no_evidence()
    test_even_strong_evidence_cannot_enable_live_in_v1()
    print("ALL PROOF GATE TESTS PASSED")
