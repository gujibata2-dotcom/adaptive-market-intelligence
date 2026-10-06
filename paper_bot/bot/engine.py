import csv
from datetime import datetime, timezone
from pathlib import Path
from .market import get_last_price
from .strategy import SMACrossover
from .paper import PaperBroker
from .risk import RiskManager

class TradingEngine:
    def __init__(self, symbol="BTC_THB", start_thb=1000,
                 fast=5, slow=20, log_path="trading_log.csv"):
        self.symbol = symbol
        self.strategy = SMACrossover(fast, slow)
        self.broker = PaperBroker(start_thb=start_thb)
        self.risk = RiskManager()
        self.log_path = Path(log_path)
        if not self.log_path.exists():
            with self.log_path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "timestamp","symbol","price","signal","reason",
                    "cash","asset","equity","entry_price","action"
                ])

    def step(self):
        price = get_last_price(self.symbol)
        info = self.strategy.update(price)
        equity = self.broker.equity(price)
        action = "NO_TRADE"

        forced = self.risk.forced_exit(self.broker.entry_price, price)
        if forced and self.broker.asset > 0:
            self.broker.sell_all(price)
            action = forced
        elif info["signal"] == "BUY" and self.broker.asset == 0:
            amount = self.risk.buy_amount(equity, self.broker.cash)
            if amount > 0:
                self.broker.buy_thb(amount, price)
                action = "PAPER_BUY"
        elif info["signal"] == "SELL" and self.broker.asset > 0:
            self.broker.sell_all(price)
            action = "PAPER_SELL"

        equity = self.broker.equity(price)
        with self.log_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                datetime.now(timezone.utc).isoformat(),
                self.symbol, price, info["signal"], info["reason"],
                self.broker.cash, self.broker.asset, equity,
                self.broker.entry_price, action
            ])

        return {
            "price": price,
            "signal": info["signal"],
            "reason": info["reason"],
            "action": action,
            "cash": self.broker.cash,
            "asset": self.broker.asset,
            "equity": equity,
            "killed": self.risk.killed
        }
