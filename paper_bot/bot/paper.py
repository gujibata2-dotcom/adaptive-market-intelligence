class PaperBroker:
    def __init__(self, start_thb=1000.0, fee_rate=0.0025):
        self.cash = float(start_thb)
        self.asset = 0.0
        self.entry_price = None
        self.fee_rate = float(fee_rate)

    def equity(self, price):
        return self.cash + self.asset * price

    def buy_thb(self, amount, price):
        amount = min(float(amount), self.cash)
        if amount <= 0:
            return {"ok": False, "reason": "no_cash"}
        fee = amount * self.fee_rate
        qty = (amount - fee) / price
        old_cost = self.asset * (self.entry_price or price)
        new_cost = old_cost + (amount - fee)
        self.cash -= amount
        self.asset += qty
        self.entry_price = new_cost / self.asset if self.asset else None
        return {"ok": True, "side": "BUY", "price": price,
                "qty": qty, "amount_thb": amount, "fee": fee}

    def sell_all(self, price):
        if self.asset <= 0:
            return {"ok": False, "reason": "no_asset"}
        qty = self.asset
        gross = qty * price
        fee = gross * self.fee_rate
        net = gross - fee
        self.cash += net
        self.asset = 0.0
        self.entry_price = None
        return {"ok": True, "side": "SELL", "price": price,
                "qty": qty, "amount_thb": net, "fee": fee}
