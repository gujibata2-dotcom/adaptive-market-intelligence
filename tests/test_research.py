import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from app.main import create_features, run_experiment
from experiments.h005_price_range_market_regime import create_h005_dataset, run_h005


def market(rows=180):
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0, .02, rows)))
    return pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=rows),
                         "Close": close, "High": close * 1.02,
                         "Low": close * .98, "Volume": rng.integers(100, 1000, rows)})


class ResearchTests(unittest.TestCase):
    def test_unknown_target_and_infinite_features_removed(self):
        raw = market()
        raw.loc[80, "Volume"] = 0
        result = create_features(raw)
        self.assertLess(result.Date.max(), raw.Date.max())
        self.assertTrue(np.isfinite(result.select_dtypes("number")).all().all())

    def test_future_window_is_exactly_next_seven_returns(self):
        raw = market()
        result = create_h005_dataset(raw)
        i = raw.index[raw.Date == result.Date.iloc[0]][0]
        expected = raw.Close.pct_change().iloc[i + 1:i + 8].std()
        self.assertAlmostEqual(result.future_volatility_7d.iloc[0], expected)
        self.assertEqual(result.Date.max(), raw.Date.iloc[-8])

    def test_future_changes_do_not_change_past_regimes(self):
        raw = market()
        original = create_h005_dataset(raw)
        changed = raw.copy()
        changed.loc[120:, "Close"] *= np.exp(np.arange(60) * .1)
        updated = create_h005_dataset(changed)
        cols = ["Date", "volatility_threshold", "volatility_regime"]
        pd.testing.assert_frame_equal(original.loc[original.Date < raw.Date.iloc[120], cols],
                                      updated.loc[updated.Date < raw.Date.iloc[120], cols])

    def test_classifier_purges_boundary_and_writes_ledger(self):
        raw = market()
        dataset = create_features(raw)
        split = int(len(dataset) * .7)
        with tempfile.TemporaryDirectory() as folder, \
                patch("app.main.load_market_data", return_value=raw), \
                patch("app.main.XGBClassifier") as classifier, \
                patch("pandas.DataFrame.to_csv") as save:
            classifier.return_value.predict_proba.return_value = np.tile([.4, .6], (len(dataset) - split, 1))
            run_experiment()
            train_x = classifier.return_value.fit.call_args.args[0]
            self.assertEqual(len(train_x), split - 1)
            self.assertEqual(train_x.index.max(), split - 2)
            save.assert_called_once()

    def test_h005_runs_offline_and_exports_holdout_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "experiments").mkdir()
            with patch("experiments.h005_price_range_market_regime.download_market_data", return_value=market()), \
                    patch("experiments.h005_price_range_market_regime.ROOT", root):
                run_h005()
            metrics = pd.read_csv(root / "experiments/h005_validation.csv")
            self.assertEqual(len(metrics), 2)
            self.assertTrue((metrics.purged_samples == 7).all())
            self.assertTrue(np.isfinite(metrics[["oos_r2", "oos_rmse", "baseline_rmse"]]).all().all())


if __name__ == "__main__":
    unittest.main()
