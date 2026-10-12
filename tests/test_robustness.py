import unittest
from unittest.mock import patch, Mock
import pandas as pd
import requests
from app.dashboard_service import demo_data
from app.data.btc_sources import load_btc, coinbase_daily
from app.robustness import walk_forward
from app.paper_trading import simulate


class RobustnessTests(unittest.TestCase):
    def test_source_fallback_and_failure(self):
        with patch('app.data.btc_sources.coinbase_daily', side_effect=requests.Timeout), patch('app.data.btc_sources.yahoo_daily', return_value=demo_data()):
            raw, name = load_btc()
            self.assertIn('Yahoo', name)
            self.assertEqual(len(raw),400)
        with patch('app.data.btc_sources.coinbase_daily', side_effect=ValueError), patch('app.data.btc_sources.yahoo_daily', side_effect=ValueError):
            with self.assertRaises(ValueError): load_btc()

    def test_coinbase_sort_filter_and_pagination(self):
        dates = pd.date_range('2024-01-01',periods=400,tz='UTC')
        payload = [[int(d.timestamp()),90,110,100,100,10] for d in dates]
        response=Mock()
        response.json.return_value=list(reversed(payload))
        session=Mock()
        session.get.return_value=response
        with patch('app.data.btc_sources.requests.Session', return_value=session):
            raw=coinbase_daily(days=399,end='2025-02-03T00:00:00Z')
        self.assertEqual(len(raw),399)
        self.assertEqual(session.get.call_count,2)
        self.assertTrue(raw.Date.is_monotonic_increasing)
        self.assertEqual(raw.Date.max(),pd.Timestamp('2025-02-02'))

    def test_matched_exposure_benchmark(self):
        raw=demo_data()
        signals=pd.DataFrame({'Date':raw.Date.iloc[:-1],'probability_up':.5})
        result=simulate(raw,signals,allocation=.25)
        self.assertAlmostEqual(result['matched_benchmark_return'], .25*result['benchmark_return'])

    def test_walk_forward_train_precedes_holdout(self):
        result, ledger=walk_forward(demo_data())
        self.assertEqual(len(result),3)
        self.assertTrue((result.train_end < result.start).all())
        self.assertFalse(ledger.Date.duplicated().any())
        self.assertEqual(len(ledger),result.test_samples.sum())
        self.assertEqual(set(result.market_regime)-{'UP','DOWN','SIDEWAYS'},set())
