"""Symbol metadata + rounding helpers.

Every exchange has lot-size and tick-size rules: e.g., for BTCUSDT spot, qty
must be a multiple of 0.000001 and price a multiple of 0.01. Submitting an
order that violates these rules is the most common reason for rejection.

This module pulls the rules via /v5/market/instruments-info (PUBLIC endpoint
— no auth needed) and provides helpers to round any (qty, price) pair into
something the exchange will accept.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import requests

from config import BybitConfig

_PUBLIC = "/v5/market/instruments-info"


@dataclass(frozen=True)
class SymbolInfo:
    symbol: str
    category: str           # "spot" | "linear" | "inverse"
    base_coin: str
    quote_coin: str
    qty_step: float         # minimum increment for quantity
    min_order_qty: float
    max_order_qty: float
    tick_size: float        # minimum increment for price
    min_notional: float     # minimum order value (quote coin)
    max_leverage: float     # only for derivatives; 1.0 for spot


def _step_places(step: float) -> int:
    if step <= 0:
        return 0
    s = f"{step:.10f}".rstrip("0")
    if "." in s:
        return len(s.split(".")[1])
    return 0


def round_down(value: float, step: float) -> float:
    """Floor to nearest multiple of step. Avoids accidental overshoot."""
    if step <= 0:
        return value
    return math.floor(value / step) * step


def round_nearest(value: float, step: float) -> float:
    if step <= 0:
        return value
    return round(value / step) * step


def fetch_symbol_info(symbol: str, category: str, config: BybitConfig) -> SymbolInfo:
    """One-shot public call. Cached at the caller level if needed."""
    url = f"{config.base_url}{_PUBLIC}"
    resp = requests.get(url, params={"category": category, "symbol": symbol}, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("retCode") != 0:
        raise RuntimeError(
            f"instruments-info error retCode={payload.get('retCode')} retMsg={payload.get('retMsg')!r}"
        )
    rows = payload["result"]["list"]
    if not rows:
        raise ValueError(f"No instrument info for {symbol} (category={category})")
    r = rows[0]

    if category == "spot":
        lot = r["lotSizeFilter"]
        price_filter = r["priceFilter"]
        return SymbolInfo(
            symbol=r["symbol"], category=category,
            base_coin=r["baseCoin"], quote_coin=r["quoteCoin"],
            qty_step=float(lot.get("basePrecision", "0")),
            min_order_qty=float(lot.get("minOrderQty", "0")),
            max_order_qty=float(lot.get("maxOrderQty", "0") or 0),
            tick_size=float(price_filter.get("tickSize", "0")),
            min_notional=float(lot.get("minOrderAmt", "0")),
            max_leverage=1.0,
        )

    # linear / inverse derivatives
    lot = r["lotSizeFilter"]
    price_filter = r["priceFilter"]
    leverage_filter = r.get("leverageFilter", {})
    return SymbolInfo(
        symbol=r["symbol"], category=category,
        base_coin=r.get("baseCoin", ""), quote_coin=r.get("quoteCoin", ""),
        qty_step=float(lot.get("qtyStep", "0")),
        min_order_qty=float(lot.get("minOrderQty", "0")),
        max_order_qty=float(lot.get("maxOrderQty", "0") or 0),
        tick_size=float(price_filter.get("tickSize", "0")),
        min_notional=float(lot.get("minNotionalValue", "0") or 0),
        max_leverage=float(leverage_filter.get("maxLeverage", "1") or 1),
    )


def format_qty(info: SymbolInfo, qty: float) -> str:
    """Round qty DOWN to qty_step and format as fixed-decimal string for the API."""
    rounded = round_down(qty, info.qty_step)
    return f"{rounded:.{_step_places(info.qty_step)}f}"


def format_price(info: SymbolInfo, price: float) -> str:
    """Round price to nearest tick and format as fixed-decimal string."""
    rounded = round_nearest(price, info.tick_size)
    return f"{rounded:.{_step_places(info.tick_size)}f}"


def validate_order(info: SymbolInfo, qty: float, price: float | None = None) -> str | None:
    """Return None if the order would pass exchange filters, else a human reason."""
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
