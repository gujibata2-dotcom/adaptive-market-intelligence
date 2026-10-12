import unittest
import numpy as np
import pandas as pd
from app.features import create_features, NORMALIZED_FEATURES
from app.dashboard_service import demo_data


class FeatureTests(unittest.TestCase):
    def test_causal_prefix(self):
        raw=demo_data()
        original=create_features(raw)
        changed=raw.copy()
        changed.loc[200:,['Close','High','Low','Volume']]*=5
        updated=create_features(changed)
        cols=['Date']+NORMALIZED_FEATURES
        pd.testing.assert_frame_equal(original.loc[original.Date < raw.Date.iloc[200],cols],
                                      updated.loc[updated.Date < raw.Date.iloc[200],cols])

    def test_price_scale_invariance(self):
        raw=demo_data()
        scaled=raw.copy()
        scaled[['Close','High','Low']]*=100
        a=create_features(raw)[NORMALIZED_FEATURES]
        b=create_features(scaled)[NORMALIZED_FEATURES]
        np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-12)

    def test_zero_volume_and_zero_range_handled(self):
        raw=demo_data()
        raw.loc[100,'Volume']=0
        raw.loc[110,['High','Low']]=raw.Close.iloc[110]
        df=create_features(raw)
        self.assertTrue(np.isfinite(df[NORMALIZED_FEATURES]).all().all())
        self.assertIn(raw.Date.iloc[101],df.Date.tolist())
        self.assertEqual(df.loc[df.Date==raw.Date.iloc[110],'close_location'].iloc[0],.5)
        self.assertNotIn(raw.Date.iloc[-1],df.Date.tolist())
