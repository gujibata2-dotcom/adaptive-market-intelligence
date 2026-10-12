import unittest
import numpy as np
import pandas as pd
from app.paper_trading import simulate


def inputs(close, probabilities):
    dates = pd.date_range('2024-01-01', periods=len(close))
    return (pd.DataFrame({'Date': dates, 'Close': close}),
            pd.DataFrame({'Date': dates[:len(probabilities)], 'probability_up': probabilities}))


class PaperTests(unittest.TestCase):
    def test_next_close_execution_and_cash_conservation(self):
        p, s = inputs([100, 120, 150, 150], [.9, .1, .1])
        r = simulate(p, s, initial_cash=1000, allocation=1, fee_rate=0, slippage=0)
        self.assertEqual(r['trades'].Date.tolist(), p.Date.iloc[[1, 2]].tolist())
        self.assertAlmostEqual(r['final_equity'], 1250)
        self.assertTrue((r['curve'].cash >= -1e-8).all())
        self.assertEqual(r['closed_trades'], 1)
        self.assertEqual(r['win_rate'], 1)
        self.assertTrue((r['trades'].signal_date < r['trades'].Date).all())

    def test_fees_slippage_and_terminal_liquidation(self):
        p, s = inputs([100, 100, 100], [.9, .9])
        r = simulate(p, s, initial_cash=1000, allocation=1, fee_rate=.01, slippage=.01)
        expected = 1000 / (101 * 1.01) * 99 * .99
        self.assertAlmostEqual(r['final_equity'], expected)
        self.assertAlmostEqual(r['net_pnl'], r['trades'].realized_pnl.sum())
        self.assertEqual(r['trades'].reason.iloc[-1], 'end_of_test')
        self.assertEqual(r['curve'].units.iloc[-1], 0)
        self.assertGreater(r['fees'], 0)
        self.assertGreater(r['max_drawdown'], 0)

    def test_stop_and_no_trade(self):
        p, s = inputs([100, 100, 70, 70], [.9, .9, .9])
        r = simulate(p, s, fee_rate=0, slippage=0)
        self.assertEqual(r['trades'].reason.iloc[-1], 'stop_at_close')
        self.assertLess(r['net_pnl'], 0)
        r = simulate(p, s.assign(probability_up=.5))
        self.assertEqual(r['final_equity'], 10000)
        self.assertEqual(r['closed_trades'], 0)
        self.assertIsNone(r['win_rate'])

    def test_bad_settings_and_missing_execution_price(self):
        p, s = inputs([100, 100, 100], [.9, .9])
        for settings in [{'allocation': 2}, {'fee_rate': -1}, {'initial_cash': np.nan}, {'buy_threshold': .3}]:
            with self.assertRaises(ValueError):
                simulate(p, s, **settings)
        with self.assertRaises(ValueError):
            simulate(p.iloc[:2], s)
