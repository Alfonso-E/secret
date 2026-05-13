"""Fetch kline (candle) data from Bitget V2 public market API.

Spot:    GET /api/v2/spot/market/candles
Futures: GET /api/v2/mix/market/candles  (productType=usdt-futures)

Both return arrays of [ts, open, high, low, close, base_vol, quote_vol, ...].
Granularity values differ from Bybit:
  Bitget: '1min', '5min', '15min', '30min', '1h', '4h', '6h', '12h', '1day', '1week'
  Bybit:  '1', '15', '60', 'D'

This module exposes a Bybit-compatible interface (interval='60' -> '1h', etc.)
so the rest of the codebase doesn't need to know the difference.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.bitget.com"
_MAX_LIMIT = 200

# Bitget uses different granularity strings for spot vs futures (mix).
#   spot   -> '1min', '5min', '15min', '30min', '1h', '4h', '1day', ...
#   mix    -> '1m',   '5m',   '15m',   '30m',   '1H', '4H', '1D',  ...
_INTERVAL_TO_SPOT: dict[str, str] = {
    "1":   "1min",
    "3":   "3min",
    "5":   "5min",
    "15":  "15min",
    "30":  "30min",
    "60":  "1h",
    "120": "2h",
    "240": "4h",
    "360": "6h",
    "720": "12h",
    "D":   "1day",
    "W":   "1week",
}
_INTERVAL_TO_MIX: dict[str, str] = {
    "1":   "1m",
    "3":   "3m",
    "5":   "5m",
    "15":  "15m",
    "30":  "30m",
    "60":  "1H",
    "120": "2H",
    "240": "4H",
    "360": "6H",
    "720": "12H",
    "D":   "1D",
    "W":   "1W",
}

# Milliseconds per interval — used by the paginator.
_INTERVAL_MS: dict[str, int] = {
    "1":   60_000,
    "3":   3 * 60_000,
    "5":   5 * 60_000,
    "15":  15 * 60_000,
    "30":  30 * 60_000,
    "60":  60 * 60_000,
    "120": 120 * 60_000,
    "240": 240 * 60_000,
    "360": 360 * 60_000,
    "720": 720 * 60_000,
    "D":   24 * 60 * 60_000,
    "W":   7 * 24 * 60 * 60_000,
}

# Aliases so the existing scripts using the Bybit-style "spot"/"linear" category
# work unchanged.
TIMEFRAMES: dict[str, str] = {"1m": "1", "15m": "15", "1h": "60", "1d": "D"}
_CATEGORY_TO_PATH: dict[str, str] = {
    "spot":   "/api/v2/spot/market/candles",
    "linear": "/api/v2/mix/market/candles",
}
_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "turnover"]


def _coerce_utc(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")


def fetch_klines(
    symbol: str = "BTCUSDT",
    interval: str = "60",
    category: str = "spot",
    limit: int = 200,
    start_ms: int | None = None,
    end_ms: int | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch a single batch of klines. Returns DataFrame indexed by open_time (UTC), oldest-first."""
    if category not in _CATEGORY_TO_PATH:
        raise ValueError(f"Unsupported category {category!r}")
    interval_map = _INTERVAL_TO_SPOT if category == "spot" else _INTERVAL_TO_MIX
    if interval not in interval_map:
        raise ValueError(f"Unsupported interval {interval!r} for category {category!r}")

    params: dict[str, object] = {
        "symbol":      symbol,
        "granularity": interval_map[interval],
        "limit":       min(limit, _MAX_LIMIT),
    }
    if category == "linear":
        params["productType"] = "usdt-futures"
    if start_ms is not None:
        params["startTime"] = start_ms
    if end_ms is not None:
        params["endTime"] = end_ms

    url = f"{BASE_URL}{_CATEGORY_TO_PATH[category]}"
    http = session or requests
    resp = http.get(url, params=params, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    if str(payload.get("code")) != "00000":
        raise RuntimeError(f"Bitget API error code={payload.get('code')} msg={payload.get('msg')!r}")

    rows = payload.get("data") or []
    if not rows:
        return pd.DataFrame(columns=_COLUMNS[1:]).rename_axis("open_time")

    # Bitget rows: [ts, open, high, low, close, baseVol, quoteVol(, usdtVol)]
    df = pd.DataFrame(rows)
    df = df.iloc[:, :7]
    df.columns = _COLUMNS
    df["open_time"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume", "turnover"):
        df[col] = df[col].astype("float64")
    return df.set_index("open_time").sort_index()


def fetch_historical(
    symbol: str = "BTCUSDT",
    interval: str = "60",
    category: str = "spot",
    start: str | pd.Timestamp = "365D",
    end: str | pd.Timestamp | None = None,
    rate_limit_sleep: float = 0.1,
) -> pd.DataFrame:
    """Paginate the public klines endpoint to cover the requested range."""
    if interval not in _INTERVAL_MS:
        raise ValueError(f"Unsupported interval {interval!r}")
    step_ms = _INTERVAL_MS[interval]

    end_ts = _coerce_utc(end) if end is not None else pd.Timestamp.now(tz="UTC")
    if isinstance(start, str) and start[-1].isalpha() and start[0].isdigit():
        start_ts = end_ts - pd.Timedelta(start)
    else:
        start_ts = _coerce_utc(start)

    target_start_ms = int(start_ts.timestamp() * 1000)
    cursor_end_ms   = int(end_ts.timestamp() * 1000)

    session = requests.Session()
    chunks: list[pd.DataFrame] = []

    while True:
        batch = fetch_klines(
            symbol=symbol, interval=interval, category=category,
            limit=_MAX_LIMIT, end_ms=cursor_end_ms, session=session,
        )
        if batch.empty:
            break
        chunks.append(batch)
        oldest_ms = int(batch.index[0].timestamp() * 1000)
        if oldest_ms <= target_start_ms or len(batch) < _MAX_LIMIT:
            break
        cursor_end_ms = oldest_ms - step_ms
        time.sleep(rate_limit_sleep)

    if not chunks:
        return pd.DataFrame(columns=_COLUMNS[1:]).rename_axis("open_time")

    full = pd.concat(chunks).sort_index()
    full = full[~full.index.duplicated(keep="first")]
    return full.loc[full.index >= start_ts]


def load_or_fetch(
    csv_path: str | Path,
    symbol: str = "BTCUSDT",
    interval: str = "60",
    category: str = "spot",
    lookback: str = "400D",
    refresh: bool = True,
) -> pd.DataFrame:
    """Load cached history from CSV; fetch (or top up) from Bitget as needed."""
    path = Path(csv_path)
    cached = pd.DataFrame()
    if path.exists():
        cached = pd.read_csv(path, parse_dates=["open_time"]).set_index("open_time")
        cached.index = pd.to_datetime(cached.index, utc=True)

    if not refresh and not cached.empty:
        return cached

    if cached.empty:
        fresh = fetch_historical(symbol, interval, category, start=lookback)
    else:
        last_ts = cached.index[-1]
        if pd.Timestamp.now(tz="UTC") - last_ts < pd.Timedelta(_INTERVAL_MS[interval], unit="ms"):
            return cached
        fresh = fetch_historical(symbol, interval, category, start=last_ts)

    combined = pd.concat([cached, fresh]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path)
    return combined


def load_or_fetch_range(
    csv_path: str | Path,
    symbol: str = "BTCUSDT",
    interval: str = "60",
    category: str = "spot",
    start: str | pd.Timestamp = "2023-01-01",
    end: str | pd.Timestamp = "2024-12-31",
) -> pd.DataFrame:
    """Load a fixed historical window from CSV cache, fetching once if absent."""
    path = Path(csv_path)
    start_ts = _coerce_utc(start)
    end_ts = _coerce_utc(end)

    if path.exists():
        cached = pd.read_csv(path, parse_dates=["open_time"]).set_index("open_time")
        cached.index = pd.to_datetime(cached.index, utc=True)
        if not cached.empty and cached.index[0] <= start_ts and cached.index[-1] >= end_ts:
            return cached.loc[start_ts:end_ts]

    fresh = fetch_historical(symbol, interval, category, start=start_ts, end=end_ts)
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh.to_csv(path)
    return fresh.loc[start_ts:end_ts]
