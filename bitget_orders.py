"""Order placement primitives for Bitget — all default to `dry_run=True`.

Bitget V2 quirks vs Bybit:
  - Sides/order types use LOWERCASE strings ("buy", "sell", "market", "limit").
  - Spot orders go to /api/v2/spot/trade/place-order. CRITICAL: for spot
    MARKET BUY, `size` is interpreted as QUOTE (USDT) amount, NOT base coin.
    Spot market SELL and all limit orders use base coin as expected. We
    handle this branch inside place_order().
  - Futures orders go to /api/v2/mix/order/place-order with `tradeSide` and
    require `marginCoin` + `marginMode`.
  - The "force" field replaces Bybit's "timeInForce".

A dry-run call prints the exact JSON body that WOULD be POSTed and returns a
synthetic OrderResult. Nothing crosses the network. The OrderResult shape
matches the Bybit version so scheduler.py is venue-agnostic.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Literal

from bitget_client import BitgetClient
from bitget_symbols import SymbolInfo, format_price, format_qty, validate_order
from logger import log

Side = Literal["Buy", "Sell"]
OrderType = Literal["Market", "Limit"]
Category = Literal["spot", "linear"]


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


def _spot_payload(*, symbol: str, side: Side, order_type: OrderType, qty: str,
                  price: str | None, force: str) -> dict:
    body: dict = {
        "symbol":   symbol,
        "side":     side.lower(),                # "buy" / "sell"
        "orderType": order_type.lower(),         # "market" / "limit"
        "force":    force,
        "size":     qty,
    }
    if price is not None:
        body["price"] = price
    return body


def _futures_payload(*, symbol: str, side: Side, order_type: OrderType, qty: str,
                     price: str | None, force: str, reduce_only: bool,
                     margin_mode: str = "isolated") -> dict:
    body: dict = {
        "symbol":      symbol,
        "productType": "usdt-futures",
        "marginCoin":  "USDT",
        "marginMode":  margin_mode,
        "size":        qty,
        "side":        side.lower(),
        "tradeSide":   "close" if reduce_only else "open",
        "orderType":   order_type.lower(),
        "force":       force,
    }
    if price is not None:
        body["price"] = price
    return body


def place_order(
    *,
    info:        SymbolInfo,
    side:        Side,
    qty:         float,
    order_type:  OrderType = "Market",
    price:       float | None = None,
    reduce_only: bool = False,
    time_in_force: str | None = None,
    reference_price: float | None = None,   # required for spot MARKET BUY (converts qty -> USDT)
    client:      BitgetClient | None = None,
    dry_run:     bool = True,
) -> OrderResult:
    if order_type == "Limit" and price is None:
        raise ValueError("Limit orders require a price")
    if order_type == "Market" and price is not None:
        raise ValueError("Market orders should not pass a price")

    err = validate_order(info, qty, price or reference_price)
    if err:
        raise ValueError(f"Order rejected by local validation: {err}")

    qty_str = format_qty(info, qty)         # base-coin string, used for our records
    price_str = format_price(info, price) if price is not None else None

    if time_in_force is None:
        # Bitget uses 'ioc' / 'gtc' / 'fok' / 'post_only' in V2.
        time_in_force = "ioc" if order_type == "Market" else "gtc"

    if info.category == "spot":
        path = "/api/v2/spot/trade/place-order"
        if order_type == "Market" and side == "Buy":
            # Bitget V2 spot market BUY expects `size` as QUOTE (USDT) amount.
            if reference_price is None or reference_price <= 0:
                raise ValueError(
                    "Spot market BUY requires a positive reference_price so we can "
                    "compute the USDT amount Bitget expects in `size`."
                )
            usdt_size = qty * reference_price
            body = {
                "symbol":    info.symbol,
                "side":      "buy",
                "orderType": "market",
                "force":     time_in_force,
                "size":      f"{usdt_size:.4f}",
            }
        else:
            body = _spot_payload(symbol=info.symbol, side=side, order_type=order_type,
                                 qty=qty_str, price=price_str, force=time_in_force)
    else:
        path = "/api/v2/mix/order/place-order"
        body = _futures_payload(symbol=info.symbol, side=side, order_type=order_type,
                                qty=qty_str, price=price_str, force=time_in_force,
                                reduce_only=reduce_only)

    if dry_run:
        log.info(f"  [DRY-RUN] POST {path}  body={json.dumps(body)}")
        return OrderResult(
            order_id=f"dryrun-{uuid.uuid4().hex[:12]}",
            symbol=info.symbol, side=side, order_type=order_type,
            qty=qty_str, price=price_str, category=info.category,
            dry_run=True, raw_request=body,
        )

    if client is None:
        raise ValueError("dry_run=False requires a BitgetClient instance")
    resp = client.post(path, body)
    return OrderResult(
        order_id=(resp.get("data") or {}).get("orderId", ""),
        symbol=info.symbol, side=side, order_type=order_type,
        qty=qty_str, price=price_str, category=info.category,
        dry_run=False, raw_request=body, raw_response=resp,
    )


def cancel_order(
    *,
    category: Category,
    symbol:   str,
    order_id: str,
    client:   BitgetClient | None = None,
    dry_run:  bool = True,
) -> dict:
    if category == "spot":
        path = "/api/v2/spot/trade/cancel-order"
        body = {"symbol": symbol, "orderId": order_id}
    else:
        path = "/api/v2/mix/order/cancel-order"
        body = {"symbol": symbol, "orderId": order_id, "productType": "usdt-futures", "marginCoin": "USDT"}
    if dry_run:
        log.info(f"  [DRY-RUN] POST {path}  body={json.dumps(body)}")
        return {"code": "00000", "data": {"orderId": order_id}, "dry_run": True}
    if client is None:
        raise ValueError("dry_run=False requires a BitgetClient instance")
    return client.post(path, body)


def close_position_market(
    *,
    info:    SymbolInfo,
    current_side: Side,
    qty:     float,
    client:  BitgetClient | None = None,
    dry_run: bool = True,
) -> OrderResult:
    close_side: Side = "Sell" if current_side == "Buy" else "Buy"
    return place_order(
        info=info, side=close_side, qty=qty, order_type="Market",
        reduce_only=True, client=client, dry_run=dry_run,
    )


# ---------- Spot plan (trigger / stop-loss) orders ----------

@dataclass
class PlanOrderResult:
    plan_order_id: str
    symbol:        str
    side:          Side
    size:          str
    trigger_price: str
    dry_run:       bool
    raw_request:   dict = field(default_factory=dict)
    raw_response:  dict | None = None

    def __repr__(self) -> str:
        tag = "[DRY-RUN]" if self.dry_run else "[LIVE]"
        return (f"{tag} PLAN {self.side} {self.size} {self.symbol} "
                f"triggered@{self.trigger_price} id={self.plan_order_id}")


def place_spot_stop_loss(
    *,
    info:          SymbolInfo,
    side:          Side,                 # "Sell" for typical stop-loss on a long
    qty:           float,
    trigger_price: float,
    trigger_type:  str = "fill_price",   # "fill_price" | "mark_price"
    client:        BitgetClient | None = None,
    dry_run:       bool = True,
) -> PlanOrderResult:
    """Place a SPOT trigger order that market-executes when triggerPrice is hit.

    For an EMA long position, side="Sell" and trigger_price = entry - 1.5*ATR
    gives us a hard stop that fires intra-bar (not just at hour close).
    """
    if info.category != "spot":
        raise ValueError("place_spot_stop_loss only supports spot symbols")
    err = validate_order(info, qty, trigger_price)
    if err:
        raise ValueError(f"Plan order rejected by local validation: {err}")

    qty_str = format_qty(info, qty)
    trigger_str = format_price(info, trigger_price)

    # Bitget V2 spot plan order body. Bitget rejected planType="normal_plan"
    # with code=40020 'Parameter {0} error' in a real live run. Correct V2
    # value for a size-in-base-coin trigger order is planType="amount".
    # "executePrice": "0" is the documented marker for market-type execution.
    body = {
        "symbol":       info.symbol,
        "side":         side.lower(),
        "size":         qty_str,
        "triggerPrice": trigger_str,
        "triggerType":  trigger_type,
        "orderType":    "market",
        "executePrice": "0",
        "planType":     "amount",
    }
    path = "/api/v2/spot/trade/place-plan-order"

    if dry_run:
        log.info(f"  [DRY-RUN] POST {path}  body={json.dumps(body)}")
        return PlanOrderResult(
            plan_order_id=f"dryrun-plan-{uuid.uuid4().hex[:10]}",
            symbol=info.symbol, side=side, size=qty_str,
            trigger_price=trigger_str, dry_run=True, raw_request=body,
        )

    if client is None:
        raise ValueError("dry_run=False requires a BitgetClient instance")
    resp = client.post(path, body)
    plan_id = (resp.get("data") or {}).get("orderId", "")
    return PlanOrderResult(
        plan_order_id=plan_id, symbol=info.symbol, side=side,
        size=qty_str, trigger_price=trigger_str,
        dry_run=False, raw_request=body, raw_response=resp,
    )


def transfer_internal(
    *,
    from_type: str,                       # "spot" | "usdt_futures" | "crossed_margin" | "isolated_margin" | ...
    to_type:   str,
    amount:    float,
    coin:      str = "USDT",
    client:    BitgetClient | None = None,
    dry_run:   bool = True,
) -> dict:
    """Move funds between sub-wallets within the same account.

    Carry trade needs USDT in the spot wallet (to buy the spot leg) AND in the
    USDT-futures wallet (to back perp short margin). Bitget demo accounts only
    fund one of these by default, so the bot auto-rebalances before opening.
    """
    if from_type == to_type:
        raise ValueError(f"transfer from_type == to_type ({from_type!r})")
    body = {
        "fromType": from_type,
        "toType":   to_type,
        "amount":   f"{amount:.4f}",
        "coin":     coin,
    }
    path = "/api/v2/spot/wallet/transfer"
    if dry_run:
        log.info(f"  [DRY-RUN] POST {path}  body={json.dumps(body)}")
        return {"code": "00000", "dry_run": True}
    if client is None:
        raise ValueError("dry_run=False requires a BitgetClient")
    log.info(f"    Transferring ${amount:.2f} {coin}  {from_type} -> {to_type}")
    return client.post(path, body)


def cancel_spot_plan_order(
    *,
    symbol:        str,
    plan_order_id: str,
    client:        BitgetClient | None = None,
    dry_run:       bool = True,
) -> dict:
    body = {"symbol": symbol, "orderId": plan_order_id}
    path = "/api/v2/spot/trade/cancel-plan-order"
    if dry_run:
        log.info(f"  [DRY-RUN] POST {path}  body={json.dumps(body)}")
        return {"code": "00000", "data": {"orderId": plan_order_id}, "dry_run": True}
    if client is None:
        raise ValueError("dry_run=False requires a BitgetClient instance")
    return client.post(path, body)
