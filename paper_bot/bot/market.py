import requests

BASE_URL = "https://api.bitkub.com"

def get_last_price(symbol="BTC_THB", timeout=10):
    url = f"{BASE_URL}/api/v3/market/ticker"
    r = requests.get(url, params={"sym": symbol.lower()}, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if data.get("error", 0) != 0:
        raise RuntimeError(f"Bitkub API error: {data.get('error')}")
    result = data.get("result")
    if isinstance(result, list):
        result = result[0] if result else None
    if isinstance(result, dict):
        for k in ("last", "last_price", "close"):
            if k in result:
                return float(result[k])
        for v in result.values():
            if isinstance(v, dict):
                for k in ("last", "last_price", "close"):
                    if k in v:
                        return float(v[k])
    raise RuntimeError(f"Unexpected ticker response: {data}")
