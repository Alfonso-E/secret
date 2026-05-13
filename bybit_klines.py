"""Fetch BTC kline (candle) data from Bybit v5 public market API.

Docs: https://bybit-exchange.github.io/docs/v5/market/kline
Public endpoint, no API key required.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.bybit.com/v5/market/kline"

# Bybit interval codes: 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M.
TIMEFRAMES: dict[str, str] = {
    "1m":  "1",
    "15m": "15",
    "1h":  "60",
    "1d":  "D",
}

# Milliseconds per interval — used by the paginator. Approximate for D/W/M.
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
    "M":   30 * 24 * 60 * 60_000,
}

_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "turnover"]
_MAX_LIMIT = 1000


def _coerce_utc(value: str | pd.Timestamp) -> pd.Timestamp:
    """Accept naive or tz-aware input and return a tz-aware UTC Timestamp."""
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
    """Fetch a single batch of klines. Returns DataFrame oldest-first.

    Columns: open, high, low, close, volume, turnover (float64),
    indexed by open_time (UTC).
    """
    params: dict[str, object] = {
        "category": category,
        "symbol":   symbol,
        "interval": interval,
        "limit":    limit,
    }
    if start_ms is not None:
        params["start"] = start_ms
    if end_ms is not None:
        params["end"] = end_ms

    http = session or requests
    resp = http.get(BASE_URL, params=params, timeout=15)
    resp.raise_for_status()
    payload = resp.json()

    if payload.get("retCode") != 0:
        raise RuntimeError(
            f"Bybit API error retCode={payload.get('retCode')} "
            f"retMsg={payload.get('retMsg')!r}"
        )

    rows = payload["result"]["list"]
    if not rows:
        return pd.DataFrame(columns=_COLUMNS[1:]).rename_axis("open_time")

    df = pd.DataFrame(rows, columns=_COLUMNS)
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
    """Fetch a long history range, paginating in 1000-candle batches.

    `start` can be an absolute timestamp ("2024-01-01") or a relative offset
    ("365D", "18M") interpreted as "this far back from `end`".
    `end` defaults to now (UTC).
    """
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
    full = full.loc[full.index >= start_ts]
    return full


def load_or_fetch(
    csv_path: str | Path,
    symbol: str = "BTCUSDT",
    interval: str = "60",
    category: str = "spot",
    lookback: str = "400D",
    refresh: bool = True,
) -> pd.DataFrame:
    """Load cached history from CSV; fetch (or top up) from Bybit as needed.

    If the CSV exists, only the newest candles since its last timestamp are pulled.
    Set `refresh=False` to skip the network call entirely.
    """
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
    """Load a fixed historical window from CSV cache, fetching once if absent.

    Unlike `load_or_fetch`, this is for stable backtest windows (start and end
    both in the past) — no incremental top-up needed.
    """
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


def fetch_all_timeframes(
    symbol: str = "BTCUSDT",
    category: str = "spot",
    limit: int = 200,
) -> dict[str, pd.DataFrame]:
    """Fetch the four target timeframes (1m, 15m, 1h, 1d) for a symbol."""
    session = requests.Session()
    return {
        label: fetch_klines(symbol, interval, category, limit, session=session)
        for label, interval in TIMEFRAMES.items()
    }


def _main() -> None:
    data = fetch_all_timeframes()
    for label, df in data.items():
        latest = df.iloc[-1]
        print(
            f"{label:<4} rows={len(df):<4} "
            f"latest={latest.name:%Y-%m-%d %H:%M:%S} UTC  "
            f"O={latest.open:<10.2f} H={latest.high:<10.2f} "
            f"L={latest.low:<10.2f} C={latest.close:<10.2f} "
            f"vol={latest.volume:.4f}"
        )


if __name__ == "__main__":
    _main()
