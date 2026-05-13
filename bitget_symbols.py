"""Bitget symbol metadata + rounding helpers.

Spot field names    (from /api/v2/spot/public/symbols):
  minTradeAmount, minTradeUSDT, pricePrecision, quantityPrecision
Futures field names (from /api/v2/mix/market/contracts):
  minTradeNum, sizeMultiplier, volumePlace, priceEndStep, pricePlace,
  minTradeUSDT, maxLever, fundInterval

The SymbolInfo dataclass returned here is the SAME shape as the Bybit version
so the rest of the bot (sizing.py, scheduler.py) doesn't care which venue
provided it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import requests

BASE_URL = "https://api.bitget.com"
_SPOT_PATH = "/api/v2/spot/public/symbols"
_MIX_PATH  = "/api/v2/mix/market/contracts"


@dataclass(frozen=True)
class SymbolInfo:
    symbol: str
    category: str           # "spot" | "linear"
    base_coin: str
    quote_coin: str
    qty_step: float
    min_order_qty: float
    max_order_qty: float
    tick_size: float
    min_notional: float
    max_leverage: float


def _step_places(step: float) -> int:
    if step <= 0:
        return 0
    s = f"{step:.10f}".rstrip("0")
    if "." in s:
        return len(s.split(".")[1])
    return 0


def round_down(value: float, step: float) -> float:
    if step <= 0:
        return value
    return math.floor(value / step) * step


def round_nearest(value: float, step: float) -> float:
    if step <= 0:
        return value
    return round(value / step) * step


def _to_float(v: object, default: float = 0.0) -> float:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def fetch_symbol_info(symbol: str, category: str, config=None) -> SymbolInfo:
    """One-shot public lookup. `config` accepted for parity with the Bybit version but unused."""
    del config
    if category == "spot":
        r = requests.get(f"{BASE_URL}{_SPOT_PATH}", params={"symbol": symbol}, timeout=15)
        r.raise_for_status()
        payload = r.json()
        if str(payload.get("code")) != "00000":
            raise RuntimeError(f"Bitget symbols error: {payload.get('msg')!r}")
        rows = payload.get("data") or []
        if not rows:
            raise ValueError(f"No spot symbol info for {symbol}")
        row = rows[0]
        qty_decimals   = int(_to_float(row.get("quantityPrecision"), 6))
        price_decimals = int(_to_float(row.get("pricePrecision"), 2))
        return SymbolInfo(
            symbol=row["symbol"], category="spot",
            base_coin=row.get("baseCoin", ""), quote_coin=row.get("quoteCoin", ""),
            qty_step=10 ** -qty_decimals,
            min_order_qty=_to_float(row.get("minTradeAmount")),
            max_order_qty=0.0,
            tick_size=10 ** -price_decimals,
            min_notional=_to_float(row.get("minTradeUSDT"), 1.0),
            max_leverage=1.0,
        )

    if category == "linear":
        r = requests.get(f"{BASE_URL}{_MIX_PATH}",
                         params={"productType": "usdt-futures", "symbol": symbol},
                         timeout=15)
        r.raise_for_status()
        payload = r.json()
        if str(payload.get("code")) != "00000":
            raise RuntimeError(f"Bitget contracts error: {payload.get('msg')!r}")
        rows = payload.get("data") or []
        if not rows:
            raise ValueError(f"No futures contract for {symbol}")
        row = rows[0]
        price_decimals = int(_to_float(row.get("pricePlace"), 1))
        price_end_step = _to_float(row.get("priceEndStep"), 1.0)
        tick = price_end_step * (10 ** -price_decimals)
        size_mult = _to_float(row.get("sizeMultiplier"), 0.0001)
        return SymbolInfo(
            symbol=row["symbol"], category="linear",
            base_coin=row.get("baseCoin", ""), quote_coin=row.get("quoteCoin", ""),
            qty_step=size_mult,
            min_order_qty=_to_float(row.get("minTradeNum"), size_mult),
            max_order_qty=_to_float(row.get("maxOrderQty"), 0.0),
            tick_size=tick,
            min_notional=_to_float(row.get("minTradeUSDT"), 5.0),
            max_leverage=_to_float(row.get("maxLever"), 1.0),
        )

    raise ValueError(f"Unsupported category {category!r}")


def format_qty(info: SymbolInfo, qty: float) -> str:
    rounded = round_down(qty, info.qty_step)
    return f"{rounded:.{_step_places(info.qty_step)}f}"


def format_price(info: SymbolInfo, price: float) -> str:
    rounded = round_nearest(price, info.tick_size)
    return f"{rounded:.{_step_places(info.tick_size)}f}"


def validate_order(info: SymbolInfo, qty: float, price: float | None = None) -> str | None:
    if qty <= 0:
        return f"qty must be > 0 (got {qty})"
    if qty < info.min_order_qty:
        return f"qty {qty} < min_order_qty {info.min_order_qty}"
    if info.max_order_qty > 0 and qty > info.max_order_qty:
        return f"qty {qty} > max_order_qty {info.max_order_qty}"
    if price is not None and price <= 0:
        return f"price must be > 0 (got {price})"
    if price is not None and info.min_notional > 0:
        if qty * price < info.min_notional:
            return f"notional {qty * price:.2f} < min_notional {info.min_notional}"
    return None
