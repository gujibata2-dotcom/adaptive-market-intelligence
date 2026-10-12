import unittest
from pathlib import Path
import numpy as np
from streamlit.testing.v1 import AppTest
from app.dashboard_service import analyze, demo_data, validate_market_data


class DashboardTests(unittest.TestCase):
    def test_real_training_produces_finite_holdout_results(self):
        result = analyze(demo_data())
        self.assertTrue(np.isfinite(result['log_loss']))
        self.assertTrue(result['train_end'] < result['test_start'])
        self.assertEqual(len(result['ledger']), 106)

    def test_bad_inputs_rejected(self):
        raw = demo_data()
        for bad in [raw.drop(columns='Volume'), raw.iloc[:100],
                    raw.assign(Volume=-1), raw.assign(Close=np.inf),
                    raw.assign(Date=raw.Date.iloc[0]), raw.drop(index=60)]:
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                validate_market_data(bad)

    def test_screen_analyze_and_source_change(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'streamlit_app.py')).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        app.button[0].click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 3)
        self.assertEqual(len(app.dataframe), 1)
        app.button[1].click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 7)
        self.assertEqual(len(app.dataframe), 2)
        app.radio[0].set_value('อัปโหลด CSV').run()
        self.assertEqual(len(app.metric), 0)
        app.button[0].click().run()
        self.assertEqual(len(app.error), 1)
        self.assertEqual(len(app.exception), 0)
