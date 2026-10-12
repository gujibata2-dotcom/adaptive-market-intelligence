"""Causal, scale-independent OHLCV features available after each daily close."""
import numpy as np
import pandas as pd

LEGACY_FEATURES = ['return_1','return_3','return_7','sma_7','sma_20','sma_50','volatility_7','volume_change']
NORMALIZED_FEATURES = ['return_1','return_3','return_7','distance_sma_7','distance_sma_20',
    'distance_sma_50','volatility_7','volatility_20','price_range','close_location',
    'volume_log_change','volume_relative_20']
FEATURE_SETS = {'legacy': LEGACY_FEATURES, 'normalized': NORMALIZED_FEATURES}


def create_features(raw, feature_set='normalized'):
    if feature_set not in FEATURE_SETS:
        raise ValueError('Unknown feature set')
    df=raw.copy()
    price=df.Close
    for days in [1,3,7]:
        df[f'return_{days}']=price.pct_change(days,fill_method=None)
    for days in [7,20,50]:
        average=price.rolling(days).mean()
        df[f'sma_{days}']=average
        df[f'distance_sma_{days}']=price/average-1
    for days in [7,20]:
        df[f'volatility_{days}']=df.return_1.rolling(days).std()
    df['volume_change']=df.Volume.pct_change(fill_method=None)
    df['price_range']=(df.High-df.Low)/price
    width=df.High-df.Low
    # A zero-range candle has neutral close location rather than infinity.
    df['close_location']=((price-df.Low)/width.where(width>0)).fillna(.5)
    df['volume_log_change']=np.log1p(df.Volume).diff()
    historical_volume=df.Volume.rolling(20).mean().shift(1)
    df['volume_relative_20']=df.Volume/historical_volume.where(historical_volume>0)-1
    df['future_return']=price.shift(-1)/price-1
    df['target']=(df.future_return>0).astype(int)
    df=df.replace([np.inf,-np.inf],np.nan)
    # Unused legacy features must not discard otherwise valid normalized rows.
    return df.dropna(subset=FEATURE_SETS[feature_set]+['future_return']).reset_index(drop=True)
