"""Fetch historical funding rates from Bybit v5 public API.

Endpoint: GET /v5/market/funding/history
Docs: https://bybit-exchange.github.io/docs/v5/market/history-fund-rate

Funding rate is expressed as a decimal fraction (0.0001 = 0.01% per funding interval).
Most USDT perpetuals on Bybit have an 8-hour funding interval, so positive
funding ~ longs pay shorts every 8 hours.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.bybit.com/v5/market/funding/history"
_MAX_LIMIT = 200


def _coerce_utc(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")


def fetch_funding_batch(
    symbol: str,
    category: str = "linear",
    limit: int = _MAX_LIMIT,
    end_ms: int | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch a single batch of funding history rows (newest-first)."""
    params: dict[str, object] = {"category": category, "symbol": symbol, "limit": limit}
    if end_ms is not None:
        params["endTime"] = end_ms

    http = session or requests
    resp = http.get(BASE_URL, params=params, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("retCode") != 0:
        raise RuntimeError(f"Bybit API error retCode={payload.get('retCode')} retMsg={payload.get('retMsg')!r}")

    rows = payload["result"]["list"]
    if not rows:
        return pd.DataFrame(columns=["funding_time", "funding_rate"]).set_index("funding_time")

    df = pd.DataFrame(rows)
    df["funding_time"] = pd.to_datetime(df["fundingRateTimestamp"].astype("int64"), unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype("float64")
    return df[["funding_time", "funding_rate"]].set_index("funding_time").sort_index()


def fetch_funding_history(
    symbol: str,
    category: str = "linear",
    start: str | pd.Timestamp = "2023-01-01",
    end: str | pd.Timestamp | None = None,
    rate_limit_sleep: float = 0.15,
) -> pd.DataFrame:
    """Paginate the funding history endpoint back to `start`."""
    end_ts = _coerce_utc(end) if end is not None else pd.Timestamp.now(tz="UTC")
    start_ts = _coerce_utc(start)
    start_ms = int(start_ts.timestamp() * 1000)
    cursor_end_ms = int(end_ts.timestamp() * 1000)

    session = requests.Session()
    chunks: list[pd.DataFrame] = []

    while True:
        batch = fetch_funding_batch(symbol, category, _MAX_LIMIT, cursor_end_ms, session)
        if batch.empty:
            break
        chunks.append(batch)
        oldest_ms = int(batch.index[0].timestamp() * 1000)
        if oldest_ms <= start_ms or len(batch) < _MAX_LIMIT:
            break
        cursor_end_ms = oldest_ms - 1
        time.sleep(rate_limit_sleep)

    if not chunks:
        return pd.DataFrame(columns=["funding_rate"]).rename_axis("funding_time")

    full = pd.concat(chunks).sort_index()
    full = full[~full.index.duplicated(keep="first")]
    return full.loc[full.index >= start_ts]


def load_or_fetch_funding(
    csv_path: str | Path,
    symbol: str,
    category: str = "linear",
    start: str | pd.Timestamp = "2023-01-01",
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load cached funding from CSV; fetch from Bybit if absent or insufficient."""
    path = Path(csv_path)
    start_ts = _coerce_utc(start)
    end_ts = _coerce_utc(end) if end is not None else pd.Timestamp.now(tz="UTC")

    if path.exists():
        cached = pd.read_csv(path, parse_dates=["funding_time"]).set_index("funding_time")
        cached.index = pd.to_datetime(cached.index, utc=True)
        if not cached.empty and cached.index[0] <= start_ts and cached.index[-1] >= end_ts - pd.Timedelta(hours=12):
            return cached.loc[start_ts:end_ts]

    fresh = fetch_funding_history(symbol, category, start_ts, end_ts)
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh.to_csv(path)
    return fresh.loc[start_ts:end_ts]


if __name__ == "__main__":
    # Smoke test on BTCUSDT
    df = fetch_funding_batch("BTCUSDT")
    print(f"Latest {len(df)} funding rows for BTCUSDT:")
    print(df.tail(5))
    avg = df["funding_rate"].mean()
    print(f"\nMean funding rate (sample): {avg:.6f}  ({avg * 100:.4f}% per 8h)")
    print(f"Annualized (always-on, no fees): {avg * 3 * 365 * 100:.2f}%")
