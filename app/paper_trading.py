"""Historical long-only paper execution; signals fill at the NEXT daily close."""
import numpy as np
import pandas as pd


def simulate(prices, signals, initial_cash=10000., allocation=.25, fee_rate=.001,
             slippage=.001, buy_threshold=.6, sell_threshold=.4, stop_loss=.1):
    params = [initial_cash, allocation, fee_rate, slippage, buy_threshold, sell_threshold, stop_loss]
    if not all(np.isfinite(v) for v in params) or initial_cash <= 0:
        raise ValueError("เงินเริ่มต้นและค่าตั้งต้องเป็นตัวเลขที่ถูกต้อง")
    if not (0 < allocation <= 1 and 0 <= fee_rate < 1 and 0 <= slippage < 1
            and 0 <= sell_threshold < buy_threshold <= 1 and 0 < stop_loss < 1):
        raise ValueError("ค่าตั้งอยู่นอกช่วงที่รองรับ")
    prices = prices[['Date', 'Close']].copy()
    signals = signals[['Date', 'probability_up']].copy()
    for frame in (prices, signals):
        frame['Date'] = pd.to_datetime(frame.Date, errors='coerce', utc=True).dt.tz_convert(None)
        if frame.Date.isna().any() or frame.Date.duplicated().any():
            raise ValueError("วันที่ไม่ถูกต้องหรือซ้ำ")
    prices = prices.sort_values('Date').reset_index(drop=True)
    signals = signals.sort_values('Date')
    if signals.empty or prices.empty:
        raise ValueError("ต้องมีราคาและสัญญาณ")
    if not np.isfinite(prices.Close).all() or (prices.Close <= 0).any():
        raise ValueError("ราคาต้องเป็นบวกและมีค่าจำกัด")
    if not np.isfinite(signals.probability_up).all() or not signals.probability_up.between(0, 1).all():
        raise ValueError("ความน่าจะเป็นต้องอยู่ระหว่าง 0 และ 1")
    if not signals.Date.isin(prices.Date).all():
        raise ValueError("วันที่สัญญาณต้องมีในข้อมูลราคา")
    if not prices.Date.dt.normalize().diff().iloc[1:].eq(pd.Timedelta(days=1)).all():
        raise ValueError("ราคาต้องต่อเนื่องรายวัน")
    prices = prices.loc[prices.Date >= signals.Date.min()].reset_index(drop=True)
    if len(prices) < 2 or signals.Date.max() >= prices.Date.iloc[-1]:
        raise ValueError("ต้องมีราคาหลังสัญญาณสุดท้ายเพื่อจำลองการส่งคำสั่ง")
    lookup = signals.set_index('Date').probability_up.to_dict()
    cash, units, entry, entry_cost = float(initial_cash), 0., 0., 0.
    trades, curve, realized = [], [], []
    pending = None
    for i, row in prices.iterrows():
        date, price = row.Date, float(row.Close)
        action = None
        reason = None
        signal_date = None
        if units and price <= entry * (1 - stop_loss):
            action, reason = 'SELL', 'stop_at_close'
        elif pending is not None:
            signal_date, probability = pending
            if units and probability <= sell_threshold:
                action, reason = 'SELL', 'signal'
            elif not units and probability >= buy_threshold and i < len(prices) - 1:
                action, reason = 'BUY', 'signal'
        if i == len(prices) - 1 and units:
            action, reason = 'SELL', 'end_of_test'
        if action == 'BUY':
            fill = price * (1 + slippage)
            budget = cash * allocation
            units = budget / (fill * (1 + fee_rate))
            fee = units * fill * fee_rate
            cash -= budget
            entry, entry_cost = fill, budget
            trades.append(dict(Date=date, signal_date=signal_date, side=action, reason=reason,
                               price=fill, units=units, fee=fee, realized_pnl=0.))
        elif action == 'SELL':
            fill = price * (1 - slippage)
            fee = units * fill * fee_rate
            proceeds = units * fill - fee
            pnl = proceeds - entry_cost
            cash += proceeds
            realized.append(pnl)
            trades.append(dict(Date=date, signal_date=signal_date, side=action, reason=reason,
                               price=fill, units=units, fee=fee, realized_pnl=pnl))
            units = 0.
        equity = cash + units * price
        curve.append(dict(Date=date, cash=cash, units=units, equity=equity))
        # A signal based on this close cannot execute at this same close.
        pending = (date, lookup[date]) if date in lookup else None
    curve = pd.DataFrame(curve)
    curve['drawdown'] = curve.equity / curve.equity.cummax() - 1
    # Buy-and-hold enters at the same first eligible execution close and exits at end.
    benchmark_units = initial_cash / (float(prices.Close.iloc[1]) * (1 + slippage) * (1 + fee_rate))
    benchmark = benchmark_units * float(prices.Close.iloc[-1]) * (1 - slippage) * (1 - fee_rate)
    trade_frame = pd.DataFrame(trades, columns=['Date', 'signal_date', 'side', 'reason', 'price', 'units', 'fee', 'realized_pnl'])
    return dict(curve=curve, trades=trade_frame, final_equity=cash,
        net_pnl=cash-initial_cash, total_return=cash/initial_cash-1,
        max_drawdown=-float(curve.drawdown.min()), closed_trades=len(realized),
        win_rate=float(np.mean(np.array(realized) > 0)) if realized else None,
        fees=float(trade_frame.fee.sum()), benchmark_return=benchmark/initial_cash-1)
