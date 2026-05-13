"""Order placement primitives — all default to `dry_run=True`.

A dry-run call prints the exact JSON body that WOULD be POSTed to /v5/order/create
and returns a synthetic OrderResult. Nothing crosses the network. Useful for
verifying the bot's intended behavior before flipping to live.

Live mode (`dry_run=False`) signs and sends the request via BybitClient. The
return shape is identical so calling code doesn't branch.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Literal

from bybit_client import BybitClient
from bybit_symbols import SymbolInfo, format_price, format_qty, validate_order

Side = Literal["Buy", "Sell"]
OrderType = Literal["Market", "Limit"]
Category = Literal["spot", "linear", "inverse"]


@dataclass
class OrderResult:
    order_id:    str
    symbol:      str
    side:        Side
    order_type:  OrderType
    qty:         str
    price:       str | None
    category:    Category
    dry_run:     bool
    raw_request: dict = field(default_factory=dict)
    raw_response: dict | None = None

    def __repr__(self) -> str:
        tag = "[DRY-RUN]" if self.dry_run else "[LIVE]"
        price_part = f" @ {self.price}" if self.price else ""
        return (f"{tag} {self.order_type} {self.side} {self.qty} {self.symbol} ({self.category}){price_part} "
                f"id={self.order_id}")


def _build_payload(
    *, category: Category, symbol: str, side: Side, order_type: OrderType,
    qty: str, price: str | None, reduce_only: bool, time_in_force: str,
) -> dict:
    payload: dict = {
        "category":   category,
        "symbol":     symbol,
        "side":       side,
        "orderType":  order_type,
        "qty":        qty,
        "timeInForce": time_in_force,
    }
    if price is not None:
        payload["price"] = price
    if reduce_only:
        payload["reduceOnly"] = True
    return payload


def place_order(
    *,
    info:        SymbolInfo,
    side:        Side,
    qty:         float,
    order_type:  OrderType = "Market",
    price:       float | None = None,
    reduce_only: bool = False,
    time_in_force: str | None = None,
    client:      BybitClient | None = None,
    dry_run:     bool = True,
) -> OrderResult:
    """Submit (or pretend-submit) a single order.

    Validation runs even in dry-run so we catch lot-size / min-notional issues early.
    """
    if order_type == "Limit" and price is None:
        raise ValueError("Limit orders require a price")
    if order_type == "Market" and price is not None:
        raise ValueError("Market orders should not pass a price")

    err = validate_order(info, qty, price)
    if err:
        raise ValueError(f"Order rejected by local validation: {err}")

    qty_str = format_qty(info, qty)
    price_str = format_price(info, price) if price is not None else None

    if time_in_force is None:
        time_in_force = "IOC" if order_type == "Market" else "GTC"

    payload = _build_payload(
        category=info.category, symbol=info.symbol, side=side,
        order_type=order_type, qty=qty_str, price=price_str,
        reduce_only=reduce_only, time_in_force=time_in_force,
    )

    if dry_run:
        print(f"  [DRY-RUN] POST /v5/order/create  body={json.dumps(payload)}")
        return OrderResult(
            order_id=f"dryrun-{uuid.uuid4().hex[:12]}",
            symbol=info.symbol, side=side, order_type=order_type,
            qty=qty_str, price=price_str, category=info.category,
            dry_run=True, raw_request=payload,
        )

    if client is None:
        raise ValueError("dry_run=False requires a BybitClient instance")
    resp = client.post("/v5/order/create", payload)
    return OrderResult(
        order_id=resp["result"]["orderId"],
        symbol=info.symbol, side=side, order_type=order_type,
        qty=qty_str, price=price_str, category=info.category,
        dry_run=False, raw_request=payload, raw_response=resp,
    )


def cancel_order(
    *,
    category:  Category,
    symbol:    str,
    order_id:  str,
    client:    BybitClient | None = None,
    dry_run:   bool = True,
) -> dict:
    payload = {"category": category, "symbol": symbol, "orderId": order_id}
    if dry_run:
        print(f"  [DRY-RUN] POST /v5/order/cancel  body={json.dumps(payload)}")
        return {"retCode": 0, "result": {"orderId": order_id}, "dry_run": True}
    if client is None:
        raise ValueError("dry_run=False requires a BybitClient instance")
    return client.post("/v5/order/cancel", payload)


def close_position_market(
    *,
    info:    SymbolInfo,
    current_side: Side,   # if we are LONG, we close by SELLING; if SHORT, by BUYING
    qty:     float,
    client:  BybitClient | None = None,
    dry_run: bool = True,
) -> OrderResult:
    """Market-close an existing position. `qty` is the size to close."""
    close_side: Side = "Sell" if current_side == "Buy" else "Buy"
    return place_order(
        info=info, side=close_side, qty=qty, order_type="Market",
        reduce_only=True, client=client, dry_run=dry_run,
    )
