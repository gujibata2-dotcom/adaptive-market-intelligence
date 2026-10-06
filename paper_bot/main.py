import time
from bot.engine import TradingEngine

engine = TradingEngine(symbol="BTC_THB", start_thb=1000, fast=5, slow=20)

print("PROVE BEFORE TRADE - PAPER MODE")
print("No real orders will be sent.")

while True:
    try:
        result = engine.step()
        print(result)
    except KeyboardInterrupt:
        print("Stopped.")
        break
    except Exception as e:
        print("ERROR:", e)
    time.sleep(60)
