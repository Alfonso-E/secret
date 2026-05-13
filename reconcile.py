"""Reconcile current exchange state against strategy intent.

Why this exists: without state diffing, the bot would re-open positions every
funding cycle even when it's already holding them. The reconciler queries
Bitget for actual balances + positions, compares against the strategy's
desired state, and emits the MINIMAL set of orders needed to bridge the gap.

Design rules:
  - BTC is reserved for the EMA overlay. Any BTC spot balance is treated as
    EMA position; the carry universe excludes BTC.
  - The carry trade always has shape: long-spot + short-perp on the SAME
    asset. We only consider short perpetuals as carry positions.
  - Diff orders close OLD positions before opening NEW ones, so we don't
    accidentally double up notional.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bitget_account import get_positions, get_spot_plan_orders, get_wallet_balance
from bitget_client import BitgetClient


@dataclass
class PlanOrder:
    plan_order_id: str
    symbol:        str
    side:          str       # 'Sell' for a long-position stop-loss
    size:          float
    trigger_price: float


@dataclass
class PositionView:
    """Snapshot of what we currently hold on the exchange."""
    spot_balances:  dict[str, float] = field(default_factory=dict)  # 'ETH' -> 4.5
    perp_shorts:    dict[str, float] = field(default_factory=dict)  # 'ETHUSDT' -> 4.5 (size)
    perp_longs:     dict[str, float] = field(default_factory=dict)  # rare but possible
    perp_entries:   dict[str, float] = field(default_factory=dict)  # 'ETHUSDT' -> avg entry price
    plan_orders:    list[PlanOrder] = field(default_factory=list)   # active spot trigger orders
    total_equity_usd: float = 0.0

    def plan_orders_for(self, symbol: str) -> list[PlanOrder]:
        return [p for p in self.plan_orders if p.symbol == symbol]


def fetch_state(client: BitgetClient) -> PositionView:
    """Aggregate spot wallet + USDT-perp positions into one view."""
    snap = get_wallet_balance(client)
    pv = PositionView(total_equity_usd=snap.total_equity_usd)
    for c in snap.coins:
        if "futures" in c.coin.lower():
            continue
        if c.coin.upper() == "USDT":
            continue
        if c.wallet_balance > 0:
            pv.spot_balances[c.coin.upper()] = c.wallet_balance

    pos = get_positions(client)
    for _, row in pos.iterrows():
        sym = row["symbol"]
        size = float(row["size"])
        entry = float(row["avg_price"])
        if size <= 0:
            continue
        if row["side"] == "Sell":
            pv.perp_shorts[sym] = size
        elif row["side"] == "Buy":
            pv.perp_longs[sym] = size
        pv.perp_entries[sym] = entry

    try:
        plan_df = get_spot_plan_orders(client)
        for _, r in plan_df.iterrows():
            pv.plan_orders.append(PlanOrder(
                plan_order_id=str(r["plan_order_id"]),
                symbol=str(r["symbol"]),
                side=str(r["side"]),
                size=float(r["size"]),
                trigger_price=float(r["trigger_price"]),
            ))
    except Exception:
        # Plan-orders endpoint occasionally rejects on fresh demo accounts. Non-fatal.
        pass

    return pv


@dataclass
class CarryIntent:
    """What the strategy wants the carry portion to be at."""
    target_symbol: str | None       # which asset to be long-spot + short-perp; None = flat
    target_notional_usd: float       # USD notional per leg


@dataclass
class EmaIntent:
    """What the strategy wants the EMA portion to be at."""
    want_long_btc: bool             # True = hold BTC spot, False = flat
    target_notional_usd: float


@dataclass
class DiffAction:
    """One atomic action emitted by the reconciler."""
    kind: str                        # 'close_carry' | 'open_carry' | 'open_ema' | 'close_ema'
    symbol: str                      # for carry: the pair symbol; for EMA: BTCUSDT
    spot_qty: float = 0.0            # base coin to buy (+) or sell (-) on spot
    perp_qty: float = 0.0            # base coin to short-open (+) or short-close (+, with reduce_only=True)
    spot_price: float = 0.0          # latest reference price (for sizing min-notional checks)
    perp_price: float = 0.0
    notional_usd: float = 0.0
    reason: str = ""


def compute_diff(
    current: PositionView,
    carry_intent: CarryIntent,
    ema_intent: EmaIntent,
    spot_prices: dict[str, float],
    perp_prices: dict[str, float],
    carry_universe: list[str],
    ema_symbol: str = "BTCUSDT",
    size_tolerance_pct: float = 5.0,
) -> list[DiffAction]:
    """Return the actions needed to go from `current` to the intents.

    `size_tolerance_pct` avoids churning on micro-drift: a position within X%
    of the target size is considered correct and untouched.
    """
    actions: list[DiffAction] = []

    # --- Step 1: close carry pairs we shouldn't be holding ----------------
    held_carry_syms = set(current.perp_shorts.keys()) & set(carry_universe)
    target_sym = carry_intent.target_symbol if carry_intent.target_notional_usd > 0 else None

    for sym in sorted(held_carry_syms):
        if sym == target_sym:
            continue  # we'll handle resizing in step 2
        perp_size = current.perp_shorts.get(sym, 0.0)
        base_coin = sym.replace("USDT", "")
        spot_size = current.spot_balances.get(base_coin, 0.0)
        actions.append(DiffAction(
            kind="close_carry",
            symbol=sym,
            spot_qty=spot_size,
            perp_qty=perp_size,
            spot_price=spot_prices.get(sym, 0.0),
            perp_price=perp_prices.get(sym, 0.0),
            notional_usd=perp_size * perp_prices.get(sym, 0.0),
            reason=f"rotation away from {sym}",
        ))

    # --- Step 2: open / resize the target carry pair ---------------------
    if target_sym is not None and target_sym in spot_prices and target_sym in perp_prices:
        sp, pp = spot_prices[target_sym], perp_prices[target_sym]
        notional = carry_intent.target_notional_usd
        desired_spot = notional / sp
        desired_perp = notional / pp

        base_coin = target_sym.replace("USDT", "")
        cur_spot = current.spot_balances.get(base_coin, 0.0)
        cur_perp = current.perp_shorts.get(target_sym, 0.0)

        # Tolerance check: if within X% on both legs, leave alone
        spot_drift = abs(cur_spot - desired_spot) / desired_spot if desired_spot > 0 else 0
        perp_drift = abs(cur_perp - desired_perp) / desired_perp if desired_perp > 0 else 0
        within_tol = (spot_drift * 100 < size_tolerance_pct
                      and perp_drift * 100 < size_tolerance_pct)

        if not within_tol:
            needed_spot = max(0.0, desired_spot - cur_spot)
            needed_perp = max(0.0, desired_perp - cur_perp)
            if needed_spot > 0 or needed_perp > 0:
                actions.append(DiffAction(
                    kind="open_carry",
                    symbol=target_sym,
                    spot_qty=needed_spot,
                    perp_qty=needed_perp,
                    spot_price=sp, perp_price=pp,
                    notional_usd=notional,
                    reason=(f"open carry on {target_sym}" if cur_perp == 0
                            else f"top up carry on {target_sym}"),
                ))

    # --- Step 3: EMA on BTC ----------------------------------------------
    btc_bal = current.spot_balances.get("BTC", 0.0)
    btc_price = spot_prices.get(ema_symbol, 0.0)
    if ema_intent.want_long_btc and ema_intent.target_notional_usd > 0 and btc_price > 0:
        desired_btc = ema_intent.target_notional_usd / btc_price
        if btc_bal < desired_btc * (1 - size_tolerance_pct / 100):
            needed = desired_btc - btc_bal
            actions.append(DiffAction(
                kind="open_ema",
                symbol=ema_symbol,
                spot_qty=needed, perp_qty=0.0,
                spot_price=btc_price, perp_price=0.0,
                notional_usd=needed * btc_price,
                reason="EMA entry signal",
            ))
    elif not ema_intent.want_long_btc and btc_bal > 0 and btc_price > 0:
        actions.append(DiffAction(
            kind="close_ema",
            symbol=ema_symbol,
            spot_qty=btc_bal, perp_qty=0.0,
            spot_price=btc_price, perp_price=0.0,
            notional_usd=btc_bal * btc_price,
            reason="EMA exit signal",
        ))

    return actions
