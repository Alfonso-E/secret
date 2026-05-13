"""Fetch historical funding rates from Bitget V2 public API.

Endpoint: GET /api/v2/mix/market/history-fund-rate
Public, no auth. Paginated by pageNo/pageSize (max 100/page).

Response shape:
  data: [ { symbol, fundingRate, fundingTime } ]

Funding interval varies per symbol — most majors are 8 hours.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.bitget.com"
_PATH = "/api/v2/mix/market/history-fund-rate"
_MAX_PAGE_SIZE = 100


def _coerce_utc(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")


def fetch_funding_page(
    symbol: str,
    product_type: str = "usdt-futures",
    page_no: int = 1,
    page_size: int = _MAX_PAGE_SIZE,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch one page of funding history. Pages are newest-first."""
    http = session or requests
    resp = http.get(
        f"{BASE_URL}{_PATH}",
        params={
            "symbol": symbol,
            "productType": product_type,
            "pageNo": page_no,
            "pageSize": page_size,
        },
        timeout=15,
    )
    resp.raise_for_status()
    payload = resp.json()
    if str(payload.get("code")) != "00000":
        raise RuntimeError(f"Bitget API error code={payload.get('code')} msg={payload.get('msg')!r}")

    rows = payload.get("data") or []
    if not rows:
        return pd.DataFrame(columns=["funding_rate"]).rename_axis("funding_time")

    df = pd.DataFrame(rows)
    df["funding_time"] = pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype("float64")
    return df[["funding_time", "funding_rate"]].set_index("funding_time").sort_index()


def fetch_funding_history(
    symbol: str,
    product_type: str = "usdt-futures",
    start: str | pd.Timestamp = "2023-01-01",
    end: str | pd.Timestamp | None = None,
    rate_limit_sleep: float = 0.12,
) -> pd.DataFrame:
    """Paginate funding history back to `start` (inclusive)."""
    end_ts = _coerce_utc(end) if end is not None else pd.Timestamp.now(tz="UTC")
    start_ts = _coerce_utc(start)

    session = requests.Session()
    chunks: list[pd.DataFrame] = []
    page = 1

    while True:
        batch = fetch_funding_page(symbol, product_type, page, _MAX_PAGE_SIZE, session)
        if batch.empty:
            break
        chunks.append(batch)
        oldest = batch.index[0]
        if oldest <= start_ts or len(batch) < _MAX_PAGE_SIZE:
            break
        page += 1
        time.sleep(rate_limit_sleep)

    if not chunks:
        return pd.DataFrame(columns=["funding_rate"]).rename_axis("funding_time")

    full = pd.concat(chunks).sort_index()
    full = full[~full.index.duplicated(keep="first")]
    return full.loc[(full.index >= start_ts) & (full.index <= end_ts)]


def load_or_fetch_funding(
    csv_path: str | Path,
    symbol: str,
    product_type: str = "usdt-futures",
    start: str | pd.Timestamp = "2023-01-01",
    end: str | pd.Timestamp | None = None,
    # accept Bybit-style `category` arg for drop-in compatibility
    category: str | None = None,
) -> pd.DataFrame:
    """Load cached funding from CSV; fetch from Bitget if absent or stale."""
    del category  # ignored — Bitget uses productType instead
    path = Path(csv_path)
    start_ts = _coerce_utc(start)
    end_ts = _coerce_utc(end) if end is not None else pd.Timestamp.now(tz="UTC")

    if path.exists():
        cached = pd.read_csv(path, parse_dates=["funding_time"]).set_index("funding_time")
        cached.index = pd.to_datetime(cached.index, utc=True)
        if not cached.empty and cached.index[0] <= start_ts and cached.index[-1] >= end_ts - pd.Timedelta(hours=12):
            return cached.loc[start_ts:end_ts]

    fresh = fetch_funding_history(symbol, product_type, start_ts, end_ts)
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh.to_csv(path)
    return fresh.loc[start_ts:end_ts]
