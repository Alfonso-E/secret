"""Read-only account queries against Bitget V2.

These all hit authenticated GET endpoints. Nothing here changes state on the
exchange. The returned dataclasses match the shape used by the Bybit version
so the rest of the bot doesn't care which venue it's running against.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from bitget_client import BitgetClient


@dataclass
class WalletCoin:
    coin:           str
    wallet_balance: float
    equity:         float
    usd_value:      float
    available_to_withdraw: float


@dataclass
class WalletSnapshot:
    account_type:        str
    total_equity_usd:    float
    total_available_usd: float
    total_margin_usd:    float
    coins:               list[WalletCoin]

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([vars(c) for c in self.coins])


def _to_float(v: object, default: float = 0.0) -> float:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def get_wallet_balance(client: BitgetClient) -> WalletSnapshot:
    """Aggregate USDT-futures account + spot assets into a single snapshot.

    Bitget splits balances across product types. For the carry-trade bot we
    need both the spot inventory (long leg collateral) and the USDT-futures
    margin (perp short side). We sum them into one combined snapshot.
    """
    # USDT-futures account
    fut = client.get(
        "/api/v2/mix/account/accounts",
        {"productType": "usdt-futures"},
    )
    fut_entry = (fut.get("data") or [{}])[0]
    fut_equity = _to_float(fut_entry.get("usdtEquity"))
    fut_available = _to_float(fut_entry.get("available") or fut_entry.get("crossedMaxAvailable"))
    fut_margin = _to_float(fut_entry.get("locked")) + _to_float(fut_entry.get("unionTotalMargin"))

    # Spot assets (per-coin list)
    spot = client.get("/api/v2/spot/account/assets")
    spot_coins_raw = spot.get("data") or []
    spot_equity = sum(_to_float(c.get("usdValue") or c.get("usdtValue")) for c in spot_coins_raw)
    spot_available = sum(_to_float(c.get("available")) for c in spot_coins_raw)

    coins: list[WalletCoin] = []
    for c in spot_coins_raw:
        bal = _to_float(c.get("available")) + _to_float(c.get("frozen")) + _to_float(c.get("locked"))
        if bal == 0:
            continue
        coins.append(WalletCoin(
            coin=c.get("coin", ""),
            wallet_balance=bal,
            equity=_to_float(c.get("available")) + _to_float(c.get("frozen")),
            usd_value=_to_float(c.get("usdValue") or c.get("usdtValue")),
            available_to_withdraw=_to_float(c.get("available")),
        ))
    if fut_equity > 0:
        coins.append(WalletCoin(
            coin="USDT (futures account)",
            wallet_balance=fut_equity,
            equity=fut_equity,
            usd_value=fut_equity,
            available_to_withdraw=fut_available,
        ))

    return WalletSnapshot(
        account_type="UNIFIED (spot+futures aggregated)",
        total_equity_usd=spot_equity + fut_equity,
        total_available_usd=spot_available + fut_available,
        total_margin_usd=fut_margin,
        coins=coins,
    )


def get_spot_coin_balance(client: BitgetClient, coin: str = "USDT") -> float:
    """Return spendable spot balance of a single coin (e.g., USDT)."""
    data = client.get("/api/v2/spot/account/assets", {"coin": coin})
    for r in (data.get("data") or []):
        if r.get("coin") == coin:
            return _to_float(r.get("available"))
    return 0.0


def get_futures_usdt_available(client: BitgetClient) -> float:
    """USDT in the futures wallet that's free to transfer out (not locked in margin)."""
    data = client.get("/api/v2/mix/account/accounts", {"productType": "usdt-futures"})
    entry = (data.get("data") or [{}])[0]
    return _to_float(entry.get("available") or entry.get("crossedMaxAvailable"))


def get_positions(client: BitgetClient, product_type: str = "usdt-futures") -> pd.DataFrame:
    """List open USDT-perp positions."""
    data = client.get(
        "/api/v2/mix/position/all-position",
        {"productType": product_type, "marginCoin": "USDT"},
    )
    rows = data.get("data") or []
    if not rows:
        return pd.DataFrame(columns=[
            "symbol", "side", "size", "avg_price", "unrealized_pnl",
            "position_value", "leverage", "liq_price",
        ])
    out = []
    for r in rows:
        size = _to_float(r.get("total"))
        if size == 0:
            continue
        hold_side = r.get("holdSide", "")
        out.append({
            "symbol":        r.get("symbol", ""),
            "side":          "Buy" if hold_side == "long" else "Sell" if hold_side == "short" else hold_side,
            "size":          size,
            "avg_price":     _to_float(r.get("openPriceAvg")),
            "unrealized_pnl":_to_float(r.get("unrealizedPL")),
            "position_value":_to_float(r.get("marginSize")) * _to_float(r.get("leverage"), 1.0),
            "leverage":      _to_float(r.get("leverage"), 1.0),
            "liq_price":     _to_float(r.get("liquidationPrice")),
        })
    return pd.DataFrame(out) if out else pd.DataFrame(columns=[
        "symbol", "side", "size", "avg_price", "unrealized_pnl",
        "position_value", "leverage", "liq_price",
    ])


def get_spot_plan_orders(client: BitgetClient, symbol: str | None = None) -> pd.DataFrame:
    """List active spot plan (trigger / stop-loss) orders."""
    params: dict[str, object] = {}
    if symbol:
        params["symbol"] = symbol
    data = client.get("/api/v2/spot/trade/current-plan-order", params)
    rows = (data.get("data") or {}).get("orderList") or data.get("data") or []
    if isinstance(rows, dict):
        rows = rows.get("orderList") or []
    if not rows:
        return pd.DataFrame(columns=[
            "plan_order_id", "symbol", "side", "size", "trigger_price", "status",
        ])
    return pd.DataFrame([
        {
            "plan_order_id": r.get("orderId"),
            "symbol":        r.get("symbol"),
            "side":          (r.get("side") or "").capitalize(),
            "size":          _to_float(r.get("size")),
            "trigger_price": _to_float(r.get("triggerPrice")),
            "status":        r.get("status") or r.get("state"),
        }
        for r in rows
    ])


def get_open_orders(client: BitgetClient) -> pd.DataFrame:
    """Open USDT-perp orders. (Spot open orders use a different endpoint.)"""
    data = client.get(
        "/api/v2/mix/order/orders-pending",
        {"productType": "usdt-futures"},
    )
    rows = (data.get("data") or {}).get("entrustedList") or []
    if not rows:
        return pd.DataFrame(columns=[
            "order_id", "symbol", "side", "qty", "price", "order_type", "order_status",
        ])
    return pd.DataFrame([
        {
            "order_id":     r.get("orderId"),
            "symbol":       r.get("symbol"),
            "side":         r.get("side"),
            "qty":          _to_float(r.get("size")),
            "price":        _to_float(r.get("price")),
            "order_type":   r.get("orderType"),
            "order_status": r.get("state"),
        }
        for r in rows
    ])
