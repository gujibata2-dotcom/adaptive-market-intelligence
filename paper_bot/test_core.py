from bot.strategy import SMACrossover
from bot.paper import PaperBroker
from bot.risk import RiskManager

s = SMACrossover(2, 3)
for p in [100, 99, 98, 101, 103]:
    out = s.update(p)
assert out["signal"] in {"BUY","SELL","HOLD"}

b = PaperBroker(1000, fee_rate=0)
r = b.buy_thb(200, 100)
assert r["ok"] and abs(b.asset - 2) < 1e-9
r = b.sell_all(110)
assert r["ok"] and abs(b.cash - 1020) < 1e-9

risk = RiskManager(max_position_pct=0.2)
assert risk.buy_amount(1000, 1000) == 200
print("ALL CORE TESTS PASSED")
