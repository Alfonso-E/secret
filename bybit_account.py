"""Read-only account queries against Bybit V5.

These all hit authenticated GET endpoints under /v5/account/* and /v5/position/*.
Nothing here changes state on the exchange — safe to call freely.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from bybit_client import BybitClient


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
        rows = [vars(c) for c in self.coins]
        return pd.DataFrame(rows)


def _safe_float(s: str | None) -> float:
    if s is None or s == "":
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def get_wallet_balance(client: BybitClient, account_type: str = "UNIFIED") -> WalletSnapshot:
    """Fetch the Unified Trading Account snapshot.

    For demo accounts the same endpoint returns virtual balances.
    """
    data = client.get("/v5/account/wallet-balance", {"accountType": account_type})
    entry = data["result"]["list"][0]
    coins = [
        WalletCoin(
            coin=c["coin"],
            wallet_balance=_safe_float(c.get("walletBalance")),
            equity=_safe_float(c.get("equity")),
            usd_value=_safe_float(c.get("usdValue")),
            available_to_withdraw=_safe_float(c.get("availableToWithdraw")),
        )
        for c in entry["coin"]
    ]
    return WalletSnapshot(
        account_type=entry["accountType"],
        total_equity_usd=_safe_float(entry.get("totalEquity")),
        total_available_usd=_safe_float(entry.get("totalAvailableBalance")),
        total_margin_usd=_safe_float(entry.get("totalInitialMargin")),
        coins=coins,
    )


def get_positions(client: BybitClient, category: str = "linear", settle_coin: str = "USDT") -> pd.DataFrame:
    """List open derivative positions (perp / futures)."""
    data = client.get(
        "/v5/position/list",
        {"category": category, "settleCoin": settle_coin},
    )
    rows = data["result"]["list"]
    if not rows:
        return pd.DataFrame(columns=[
            "symbol", "side", "size", "avg_price", "unrealized_pnl",
            "position_value", "leverage", "liq_price",
        ])
    return pd.DataFrame([
        {
            "symbol":        r["symbol"],
            "side":          r["side"],
            "size":          _safe_float(r.get("size")),
            "avg_price":     _safe_float(r.get("avgPrice")),
            "unrealized_pnl":_safe_float(r.get("unrealisedPnl")),
            "position_value":_safe_float(r.get("positionValue")),
            "leverage":      _safe_float(r.get("leverage")),
            "liq_price":     _safe_float(r.get("liqPrice")),
        }
        for r in rows
    ])


def get_open_orders(
    client: BybitClient, category: str = "linear", settle_coin: str = "USDT",
) -> pd.DataFrame:
    """List open (unfilled) orders for derivatives."""
    data = client.get(
        "/v5/order/realtime",
        {"category": category, "settleCoin": settle_coin},
    )
    rows = data["result"]["list"]
    if not rows:
        return pd.DataFrame(columns=[
            "order_id", "symbol", "side", "qty", "price", "order_type", "order_status",
        ])
    return pd.DataFrame([
        {
            "order_id":     r["orderId"],
            "symbol":       r["symbol"],
            "side":         r["side"],
            "qty":          _safe_float(r.get("qty")),
            "price":        _safe_float(r.get("price")),
            "order_type":   r.get("orderType"),
            "order_status": r.get("orderStatus"),
        }
        for r in rows
    ])
