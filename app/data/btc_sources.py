"""Whole-source BTC daily downloads; never stitch exchange volumes together."""
from datetime import datetime, timezone
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yfinance as yf
from app.dashboard_service import validate_market_data


def coinbase_daily(days=1825, end=None):
    end = pd.Timestamp(end or datetime.now(timezone.utc)).tz_convert('UTC').normalize() if end is None or pd.Timestamp(end).tzinfo else pd.Timestamp(end).tz_localize('UTC').normalize()
    start = end - pd.Timedelta(days=days)
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=Retry(total=2, backoff_factor=.5, status_forcelist=[429, 500, 502, 503, 504])))
    rows = []
    cursor = start
    try:
        while cursor < end:
            stop = min(cursor + pd.Timedelta(days=299), end)
            response = session.get('https://api.exchange.coinbase.com/products/BTC-USD/candles',
                params={'granularity':86400, 'start':cursor.isoformat(), 'end':stop.isoformat()}, timeout=15)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list) or any(not isinstance(row, list) or len(row) != 6 for row in payload):
                raise ValueError('Coinbase candle response has invalid schema')
            rows.extend(payload)
            cursor = stop
    finally:
        session.close()
    df = pd.DataFrame(rows, columns=['timestamp','Low','High','Open','Close','Volume'])
    df['Date'] = pd.to_datetime(df.timestamp, unit='s', utc=True)
    df = df.loc[(df.Date >= start) & (df.Date < end)].drop_duplicates('Date')
    df = validate_market_data(df)
    if len(df) != days:
        raise ValueError('Coinbase returned incomplete requested history')
    return df


def yahoo_daily():
    df = yf.download('BTC-USD', period='5y', interval='1d', auto_adjust=True, progress=False, timeout=10)
    if df.empty:
        raise ValueError('Yahoo returned no data')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    today = pd.Timestamp(datetime.now(timezone.utc)).normalize()
    df = df.loc[pd.to_datetime(df.Date, utc=True) < today]
    return validate_market_data(df)


def load_btc(source='auto'):
    if source not in ['auto','coinbase','yahoo']:
        raise ValueError('Unknown source')
    errors = []
    # Exchange candles first: bounded requests, no Yahoo crumb dependency.
    choices = [('Coinbase BTC-USD', coinbase_daily), ('Yahoo Finance BTC-USD', yahoo_daily)]
    for name, loader in choices:
        if source != 'auto' and not name.lower().startswith(source):
            continue
        try:
            return loader(), name
        except (ValueError, requests.RequestException, RuntimeError) as exc:
            errors.append(f'{name}: {type(exc).__name__}')
    raise ValueError('โหลดข้อมูลจริงไม่สำเร็จ ใช้ CSV แทนได้ (' + '; '.join(errors) + ')')
