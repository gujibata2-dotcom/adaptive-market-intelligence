from collections import deque

class SMACrossover:
    def __init__(self, fast=5, slow=20):
        if fast <= 0 or slow <= 0 or fast >= slow:
            raise ValueError("Require 0 < fast < slow")
        self.fast = fast
        self.slow = slow
        self.prices = deque(maxlen=slow + 2)
        self.prev_fast = None
        self.prev_slow = None

    def update(self, price):
        self.prices.append(float(price))
        if len(self.prices) < self.slow:
            return {"signal": "HOLD", "reason": "warming_up",
                    "fast": None, "slow": None}

        p = list(self.prices)
        fast_now = sum(p[-self.fast:]) / self.fast
        slow_now = sum(p[-self.slow:]) / self.slow
        signal, reason = "HOLD", "no_cross"

        if self.prev_fast is not None:
            if self.prev_fast <= self.prev_slow and fast_now > slow_now:
                signal, reason = "BUY", "bullish_cross"
            elif self.prev_fast >= self.prev_slow and fast_now < slow_now:
                signal, reason = "SELL", "bearish_cross"

        self.prev_fast, self.prev_slow = fast_now, slow_now
        return {"signal": signal, "reason": reason,
                "fast": fast_now, "slow": slow_now}
