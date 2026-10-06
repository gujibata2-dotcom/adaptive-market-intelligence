from datetime import date

class RiskManager:
    def __init__(self, max_position_pct=0.20,
                 max_daily_loss_pct=0.03,
                 stop_loss_pct=0.02,
                 take_profit_pct=0.04):
        self.max_position_pct = max_position_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.day = None
        self.day_start_equity = None
        self.killed = False

    def refresh_day(self, equity):
        today = date.today()
        if self.day != today:
            self.day = today
            self.day_start_equity = equity
            self.killed = False

    def allow_trading(self, equity):
        self.refresh_day(equity)
        loss = (self.day_start_equity - equity) / max(self.day_start_equity, 1e-9)
        if loss >= self.max_daily_loss_pct:
            self.killed = True
        return not self.killed

    def buy_amount(self, equity, cash):
        if not self.allow_trading(equity):
            return 0.0
        return max(0.0, min(cash, equity * self.max_position_pct))

    def forced_exit(self, entry_price, price):
        if not entry_price:
            return None
        pct = (price - entry_price) / entry_price
        if pct <= -self.stop_loss_pct:
            return "STOP_LOSS"
        if pct >= self.take_profit_pct:
            return "TAKE_PROFIT"
        return None
