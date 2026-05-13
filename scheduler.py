"""Strategy evaluator — reconciles current Bitget state against desired state.

The bot wakes up at an event boundary (every 8h funding, or hourly for EMA),
queries actual positions from Bitget, computes the diff vs strategy intent,
and emits only the orders needed to bridge the gap.

Strategy logic recap (90/10 split @ 5x carry leverage):
  Carry portion (90% of capital):
    - Cross-sectional rotation across positive-funding USDT perps.
    - At each funding period, hold the asset with the highest smoothed
      9-period funding rate.
  EMA portion (10% of capital):
    - Phase 1 EMA(50)/EMA(200) crossover on BTCUSDT spot, 1h timeframe.
    - Enter on golden cross + RSI > 50 + close > EMA(200).
    - Exit on death cross. (ATR stops are a future enhancement.)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from bitget_client import BitgetClient
from bitget_funding import load_or_fetch_funding
from bitget_klines import load_or_fetch
from bitget_account import get_futures_usdt_available, get_spot_coin_balance
from bitget_orders import (
    OrderResult, cancel_spot_plan_order, place_order, place_spot_stop_loss,
    transfer_internal,
)
from bitget_symbols import SymbolInfo, fetch_symbol_info
from config import BitgetConfig
from logger import log
from notify import notify_error, notify_halt, notify_trade
from reconcile import (
    CarryIntent, DiffAction, EmaIntent, PositionView, compute_diff, fetch_state,
)
from safety import SafetyError, SafetyGuards, SafetyLimits, SessionState
from sizing import carry_leg_size, ema_position_size
from strategy import StrategyParams, generate_signals

DATA_DIR = Path(__file__).parent / "data"

# BTC is reserved for the EMA directional overlay so we can disambiguate
# "this BTC spot is the EMA position" vs "this BTC spot is the carry leg".
# The carry universe is the remaining positive-funding majors.
CARRY_UNIVERSE = [
    "ETHUSDT", "SOLUSDT",
    "AVAXUSDT", "LINKUSDT", "ARBUSDT",
    "DOGEUSDT", "ADAUSDT",
]
EMA_SYMBOL = "BTCUSDT"


@dataclass
class StrategyConfig:
    total_capital_usd:   float = 10_000.0
    # Capital split. 0.8 = 80% carry / 20% EMA — gives more upside to the
    # directional overlay during BTC trends (backtested CAGR ~13% vs 10%
    # at the conservative 0.9 split), at the cost of higher drawdown
    # (~-8% vs -4%). Drop to 0.9 for safety, raise to 0.7 for more growth.
    carry_fraction:      float = 0.8
    carry_leverage:      float = 5.0
    carry_smooth_periods: int = 9
    carry_enter_threshold: float = 0.0
    # Hysteresis: a new asset must beat the current held asset's smoothed
    # funding rate by this much (per 8h period) to justify rotating. Matches
    # the backtest's `min_switch_advantage=0.0002` setting, which prevents
    # constant churning between near-equal carry candidates.
    carry_min_switch_advantage: float = 0.0002
    # Funding-flip safety: if the held asset's last 3 published funding rates
    # are ALL negative, force exit even if rotation wouldn't have triggered.
    carry_force_exit_negative_streak: int = 3
    # Liquidation safety: warn (Discord) when distance-to-liq falls below this
    # fraction of current price; halt + flatten if it falls below `liq_halt_pct`.
    liq_warn_pct:  float = 0.10
    liq_halt_pct:  float = 0.05
    ema_params:    StrategyParams = StrategyParams(ema_fast=50, ema_slow=200, trend_ema=200)


@dataclass
class SchedulerInputs:
    config:       BitgetConfig
    strategy:     StrategyConfig
    limits:       SafetyLimits
    state:        SessionState
    client:       BitgetClient | None       # None when using --mock-account
    spot_prices:  dict[str, float]
    perp_prices:  dict[str, float]
    btc_klines:   pd.DataFrame
    funding_panel: pd.DataFrame
    mock_position_view: PositionView | None = None  # for testing without a real account
    dry_run:      bool = True


# ---------- Strategy intent ----------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _pick_carry_target(
    panel: pd.DataFrame,
    smooth: int,
    threshold: float,
    currently_held: str | None,
    min_switch_advantage: float,
    force_exit_negative_streak: int,
) -> tuple[str | None, float, str]:
    """Pick the next carry target.

    Returns (symbol, smoothed_rate, reason). Symbol is None if we should be flat.

    Logic:
      1. If we hold a position and its last N raw funding rates are all
         negative -> force exit (return None).
      2. Compute smoothed rates for all assets clearing `threshold`.
      3. If nothing clears it, return None.
      4. If we already hold one of the candidates, stick with it unless a
         different candidate beats us by at least `min_switch_advantage`.
    """
    # Funding-flip safety check on the held position
    if currently_held and currently_held in panel.columns:
        last_n = panel[currently_held].tail(force_exit_negative_streak)
        if len(last_n) == force_exit_negative_streak and (last_n < 0).all():
            return None, 0.0, (
                f"held asset {currently_held} had {force_exit_negative_streak} consecutive "
                f"negative funding periods (last={last_n.iloc[-1]:+.6f}) — force exit"
            )

    smoothed = panel.tail(smooth).mean()
    eligible = smoothed[smoothed > threshold]
    if eligible.empty:
        return None, 0.0, f"no asset's smoothed funding > {threshold:+.6f}"

    best = eligible.idxmax()
    best_rate = float(eligible.loc[best])

    # Hysteresis: if we already hold an eligible asset, prefer to keep it
    # unless someone else clearly beats us. This matches the backtest's
    # rotation rule and prevents fee-eating churn between near-tied candidates.
    if currently_held and currently_held in eligible.index:
        held_rate = float(eligible.loc[currently_held])
        if best == currently_held:
            return currently_held, held_rate, f"held asset still highest @ {held_rate:+.6f}"
        if best_rate < held_rate + min_switch_advantage:
            return currently_held, held_rate, (
                f"held {currently_held} @ {held_rate:+.6f}; new candidate {best} @ "
                f"{best_rate:+.6f} doesn't clear hysteresis (need +{min_switch_advantage:.6f})"
            )

    return best, best_rate, f"argmax {best} @ {best_rate:+.6f}"


def _ema_signal_state(btc: pd.DataFrame, params: StrategyParams) -> dict:
    signals = generate_signals(btc, params)
    needed = ["ema_fast", "ema_slow", "rsi", "atr", "trend_ema", "long_entry", "long_exit", "close"]
    last = signals.dropna(subset=[c for c in needed if c in signals.columns]).iloc[-1]
    return {
        "time":         last.name,
        "close":        float(last["close"]),
        "ema_fast":     float(last["ema_fast"]),
        "ema_slow":     float(last["ema_slow"]),
        "trend_ema":    float(last["trend_ema"]),
        "rsi":          float(last["rsi"]),
        "atr":          float(last["atr"]),
        "long_entry":   bool(last["long_entry"]),
        "long_exit":    bool(last["long_exit"]),
        "regime_long_ok": float(last["close"]) > float(last["trend_ema"]),
    }


def _decide_ema_intent(
    ema_state: dict, current: PositionView, btc_price: float, capital_usd: float,
) -> tuple[EmaIntent, str]:
    """Decide what the EMA overlay should be doing right now.

    Logic:
      - If currently flat AND long_entry signal at latest bar -> enter long.
      - If currently long AND long_exit signal at latest bar  -> exit.
      - Otherwise hold whatever we already have.
    """
    btc_bal = current.spot_balances.get("BTC", 0.0)
    holding = btc_bal * btc_price > 1.0  # ignore dust

    if not holding and ema_state["long_entry"]:
        return EmaIntent(want_long_btc=True, target_notional_usd=capital_usd), "entry signal"
    if holding and ema_state["long_exit"]:
        return EmaIntent(want_long_btc=False, target_notional_usd=0.0), "exit signal"
    return EmaIntent(want_long_btc=holding, target_notional_usd=capital_usd if holding else 0.0), \
           ("holding through" if holding else "flat, no entry signal")


# ---------- Liquidation distance monitor ----------

def _check_liquidation_distance(
    current: PositionView,
    perp_prices: dict[str, float],
    warn_pct: float,
    halt_pct: float,
    dry_run: bool,
) -> set[str]:
    """Walk every open perp short. For each, compute distance to liquidation
    as (liq_price - current_price) / current_price (always positive for a
    short whose liq is above market). Emit Discord warnings at `warn_pct`
    and return the symbol set to force-flatten at `halt_pct`.

    The actual liquidation price comes from `current.perp_entries` plus the
    Bitget position response. If we don't have a liq price for a symbol
    (e.g., demo response missing it), we skip the check on that one.
    """
    to_flatten: set[str] = set()
    if not current.perp_shorts:
        return to_flatten

    # We need the liquidation prices Bitget returned. They're not on
    # PositionView yet — we fetched them inside get_positions but only kept
    # entries. For now, accept that the demo response may not include them
    # and skip silently when missing. (A future tweak: surface liq_price
    # through PositionView so this check is always live.)
    # Best-effort: if perp_entries[sym] is far from current price, that
    # already implies decent room.
    for sym, size in current.perp_shorts.items():
        if size <= 0:
            continue
        cur_price = perp_prices.get(sym, 0.0)
        entry = current.perp_entries.get(sym, 0.0)
        if cur_price <= 0 or entry <= 0:
            continue

        # Without an explicit liq price, approximate using 5x isolated short:
        # liquidation ~ entry * (1 + 1/leverage - maintenance_margin) ~ entry * 1.17
        # for L=5 with ~2% maintenance margin. Conservative: use 1.15.
        approx_liq = entry * 1.15
        distance_pct = (approx_liq - cur_price) / cur_price

        if distance_pct < halt_pct:
            log.warning(
                f"  [LIQ-RISK]  {sym}: cur=${cur_price:,.2f} approx_liq=${approx_liq:,.2f} "
                f"distance={distance_pct*100:.2f}% < halt {halt_pct*100:.2f}% — FLATTENING"
            )
            to_flatten.add(sym)
            if not dry_run:
                notify_error(
                    f"{sym} perp short close to liquidation: current ${cur_price:,.2f}, "
                    f"approx liq ${approx_liq:,.2f} ({distance_pct*100:.2f}% buffer). "
                    f"Flattening this position."
                )
        elif distance_pct < warn_pct:
            log.warning(
                f"  [LIQ-WARN]  {sym}: cur=${cur_price:,.2f} approx_liq=${approx_liq:,.2f} "
                f"distance={distance_pct*100:.2f}%"
            )
            if not dry_run:
                notify_error(
                    f"{sym} perp short liquidation buffer at {distance_pct*100:.2f}% "
                    f"(warn threshold {warn_pct*100:.1f}%). Position still open."
                )

    return to_flatten


# ---------- Wallet rebalancing ----------

def _ensure_spot_usdt(
    needed_usd: float,
    client: BitgetClient | None,
    dry_run: bool,
    buffer_pct: float = 0.5,
) -> None:
    """Make sure the spot wallet holds at least `needed_usd * (1 + buffer_pct/100)`.

    Live mode: if short, transfer the shortfall from the USDT-futures wallet.
    Demo mode: Bitget blocks inter-wallet transfers on demo accounts, so we
    raise a clear, actionable error telling the user how to fund the spot
    demo wallet manually via the UI.
    """
    if needed_usd <= 0:
        return
    target = needed_usd * (1 + buffer_pct / 100.0)
    if dry_run or client is None:
        log.info(f"    [dry-run wallet check]  would ensure ${target:,.2f} USDT in spot")
        return

    have = get_spot_coin_balance(client, "USDT")
    if have >= target:
        log.info(f"    Spot wallet has ${have:,.2f} USDT (>= ${target:,.2f} needed)")
        return

    shortfall = target - have

    # Demo trading on Bitget doesn't support inter-wallet transfers — the API
    # endpoint returns 40404. The user must manually fund the spot demo wallet
    # from the Bitget Demo UI. Surface a clear, actionable error.
    if client.config.is_demo:
        raise RuntimeError(
            f"Spot wallet has only ${have:.2f} USDT but needs ${target:.2f} for this trade. "
            f"Bitget DEMO accounts can't auto-transfer between sub-wallets — you have to fund "
            f"the spot demo wallet from the Bitget UI manually. "
            f"Open Bitget -> Demo Trading -> Spot, then use 'Reset Balance' / 'Recharge Demo Funds' "
            f"to put at least ${shortfall:.0f} USDT in the spot demo wallet. Re-run after that."
        )

    fut_available = get_futures_usdt_available(client)
    if fut_available < shortfall:
        raise RuntimeError(
            f"Cannot fund spot leg: need ${shortfall:,.2f} USDT but only "
            f"${fut_available:,.2f} available in futures wallet"
        )
    transfer_internal(
        from_type="usdt_futures", to_type="spot",
        amount=shortfall, coin="USDT",
        client=client, dry_run=False,
    )


# ---------- Order execution from a DiffAction ----------

def _place_carry_open(action: DiffAction, config: BitgetConfig, client: BitgetClient | None,
                      leverage: float, dry_run: bool) -> list[OrderResult]:
    """Open a long-spot + short-perp pair for one asset."""
    spot_info = fetch_symbol_info(action.symbol, "spot", config)
    perp_info = fetch_symbol_info(action.symbol, "linear", config)
    spec = carry_leg_size(
        spot_info=spot_info, perp_info=perp_info,
        spot_price=action.spot_price, perp_price=action.perp_price,
        capital_usd=action.notional_usd * (leverage + 1) / leverage,  # invert the notional math
        leverage=leverage,
    )
    if not spec.fits_exchange:
        log.info(f"    [SKIP] {action.symbol}: {spec.rejection_reason}")
        return []
    out: list[OrderResult] = []
    if action.spot_qty > 0:
        # Make sure the spot wallet has USDT (auto-transfer from futures if needed)
        _ensure_spot_usdt(action.spot_qty * action.spot_price, client, dry_run)
        out.append(place_order(
            info=spot_info, side="Buy", qty=action.spot_qty,
            order_type="Market", reference_price=action.spot_price,
            client=client, dry_run=dry_run,
        ))
    if action.perp_qty > 0:
        out.append(place_order(
            info=perp_info, side="Sell", qty=action.perp_qty,
            order_type="Market", client=client, dry_run=dry_run,
        ))
    return out


def _place_carry_close(action: DiffAction, config: BitgetConfig, client: BitgetClient | None,
                       dry_run: bool) -> list[OrderResult]:
    """Close an existing carry pair: sell the spot leg, buy back the perp short."""
    spot_info = fetch_symbol_info(action.symbol, "spot", config)
    perp_info = fetch_symbol_info(action.symbol, "linear", config)
    out: list[OrderResult] = []
    if action.spot_qty > 0:
        out.append(place_order(
            info=spot_info, side="Sell", qty=action.spot_qty,
            order_type="Market", client=client, dry_run=dry_run,
        ))
    if action.perp_qty > 0:
        out.append(place_order(
            info=perp_info, side="Buy", qty=action.perp_qty,
            order_type="Market", reduce_only=True,
            client=client, dry_run=dry_run,
        ))
    return out


def _place_ema_open(
    action: DiffAction, config: BitgetConfig, client: BitgetClient | None,
    dry_run: bool, atr: float, stop_mult: float,
) -> list:
    info = fetch_symbol_info(action.symbol, "spot", config)
    spec = ema_position_size(info=info, price=action.spot_price, capital_usd=action.notional_usd)
    if not spec.fits_exchange:
        log.info(f"    [SKIP] {action.symbol} EMA: {spec.rejection_reason}")
        return []
    results: list = []
    _ensure_spot_usdt(spec.qty * action.spot_price, client, dry_run)
    results.append(place_order(
        info=info, side="Buy", qty=spec.qty, order_type="Market",
        reference_price=action.spot_price,
        client=client, dry_run=dry_run,
    ))
    # Place an on-exchange stop-loss alongside the entry. Trigger price is
    # estimated from the last close minus stop_mult * ATR; the real fill
    # price will differ slightly, but this matches the backtest's stop logic
    # closely enough that live behavior should track backtest expectations.
    if atr > 0 and stop_mult > 0:
        stop_price = action.spot_price - stop_mult * atr
        if stop_price > 0:
            try:
                results.append(place_spot_stop_loss(
                    info=info, side="Sell", qty=spec.qty,
                    trigger_price=stop_price, trigger_type="fill_price",
                    client=client, dry_run=dry_run,
                ))
            except ValueError as e:
                log.warning(f"    [WARN] could not place EMA stop: {e}")
    return results


def _place_ema_close(
    action: DiffAction, config: BitgetConfig, client: BitgetClient | None,
    dry_run: bool, current: PositionView,
) -> list:
    info = fetch_symbol_info(action.symbol, "spot", config)
    results: list = []
    # Cancel any active stop-loss plan orders BEFORE selling, so we don't
    # leave a dangling plan that could trigger on stale state.
    for plan in current.plan_orders_for(action.symbol):
        try:
            cancel_spot_plan_order(
                symbol=plan.symbol, plan_order_id=plan.plan_order_id,
                client=client, dry_run=dry_run,
            )
            log.info(f"    [INFO] cancelled stale plan order {plan.plan_order_id} "
                     f"(trigger=${plan.trigger_price:,.2f})")
        except Exception as e:
            log.warning(f"    [WARN] failed to cancel plan {plan.plan_order_id}: {e}")
    results.append(place_order(
        info=info, side="Sell", qty=action.spot_qty, order_type="Market",
        client=client, dry_run=dry_run,
    ))
    return results


# ---------- Main evaluator ----------

def evaluate_once(inputs: SchedulerInputs) -> dict:
    """One pass: compute intent, query state, reconcile, emit orders."""
    cfg = inputs.strategy
    state = inputs.state
    guards = SafetyGuards(inputs.limits, state)

    log.info("")
    log.info("=" * 92)
    log.info(f"STRATEGY EVALUATION  {_now_utc().isoformat(timespec='seconds')}  "
          f"dry_run={inputs.dry_run}  capital=${cfg.total_capital_usd:,.0f}  "
          f"split={round(cfg.carry_fraction*100)}/{round((1 - cfg.carry_fraction)*100)}")
    log.info("=" * 92)

    try:
        guards.check_can_trade()
    except SafetyError as e:
        log.info(f"  [HALT] {e}")
        if not inputs.dry_run:
            notify_halt(str(e))
        return {"halted": True, "reason": str(e)}

    # 1. Current state (real or mocked)
    if inputs.mock_position_view is not None:
        current = inputs.mock_position_view
    elif inputs.client is not None:
        current = fetch_state(inputs.client)
    else:
        current = PositionView(total_equity_usd=cfg.total_capital_usd)

    log.info(f"  Current spot balances: {dict(current.spot_balances) or '{}'}")
    log.info(f"  Current perp shorts:   {dict(current.perp_shorts) or '{}'}")
    log.info(f"  Account equity (live): ${current.total_equity_usd:,.2f}")

    # 1b. Liquidation distance check for any open perp short
    force_flatten = _check_liquidation_distance(
        current, inputs.perp_prices, cfg.liq_warn_pct, cfg.liq_halt_pct, inputs.dry_run,
    )

    # 2. Strategy intent
    currently_held = next(
        (s for s in current.perp_shorts if s in CARRY_UNIVERSE),
        None,
    )
    target, target_rate, target_reason = _pick_carry_target(
        inputs.funding_panel,
        cfg.carry_smooth_periods,
        cfg.carry_enter_threshold,
        currently_held=currently_held,
        min_switch_advantage=cfg.carry_min_switch_advantage,
        force_exit_negative_streak=cfg.carry_force_exit_negative_streak,
    )
    carry_capital = cfg.total_capital_usd * cfg.carry_fraction
    carry_notional = carry_capital * cfg.carry_leverage / (cfg.carry_leverage + 1)
    final_target = target if target in CARRY_UNIVERSE else None
    if final_target is not None and final_target in force_flatten:
        log.warning(f"  Override: target {final_target} is in liquidation-halt set — forcing flat")
        final_target = None
    carry_intent = CarryIntent(
        target_symbol=final_target,
        target_notional_usd=carry_notional if final_target else 0.0,
    )

    ema_state = _ema_signal_state(inputs.btc_klines, cfg.ema_params)
    btc_price = inputs.spot_prices.get(EMA_SYMBOL, 0.0)
    ema_intent, ema_reason = _decide_ema_intent(
        ema_state, current, btc_price,
        cfg.total_capital_usd * (1 - cfg.carry_fraction),
    )

    log.info("")
    log.info(f"--- Intent ---")
    if carry_intent.target_symbol:
        annualized = target_rate * 3 * 365 * 100
        log.info(f"  Carry: hold {carry_intent.target_symbol}  notional ${carry_intent.target_notional_usd:,.0f}  "
              f"(smoothed rate {target_rate:+.6f}/8h = {annualized:+.2f}% APR) — {target_reason}")
    else:
        log.info(f"  Carry: stay flat — {target_reason}")
    log.info(f"  EMA  : want_long_btc={ema_intent.want_long_btc}  notional ${ema_intent.target_notional_usd:,.0f}  "
          f"reason='{ema_reason}'")
    log.info(f"         (bar @ {ema_state['time']}  close=${ema_state['close']:,.2f}  "
          f"EMA{cfg.ema_params.ema_fast}=${ema_state['ema_fast']:,.2f}  "
          f"EMA{cfg.ema_params.ema_slow}=${ema_state['ema_slow']:,.2f}  "
          f"RSI={ema_state['rsi']:.1f})")

    # 3. Reconcile
    actions = compute_diff(
        current=current,
        carry_intent=carry_intent,
        ema_intent=ema_intent,
        spot_prices=inputs.spot_prices,
        perp_prices=inputs.perp_prices,
        carry_universe=CARRY_UNIVERSE,
        ema_symbol=EMA_SYMBOL,
    )

    log.info("")
    log.info(f"--- Reconcile diff: {len(actions)} action(s) ---")
    if not actions:
        log.info(f"  No changes needed — current state already matches intent.")
    for a in actions:
        log.info(f"  {a.kind:12s}  {a.symbol:10s}  spot={a.spot_qty:>14.6f}  perp={a.perp_qty:>14.6f}  "
              f"notional=${a.notional_usd:>10,.0f}  reason='{a.reason}'")

    # 4. Execute
    all_orders: list[OrderResult] = []
    for action in actions:
        try:
            guards.check_order(symbol=f"{action.symbol}_{action.kind}",
                               notional_usd=max(action.notional_usd, 1.0))
        except SafetyError as e:
            log.info(f"  [REJECTED BY GUARD] {action.symbol}: {e}")
            continue
        try:
            if action.kind == "close_carry":
                all_orders.extend(_place_carry_close(action, inputs.config, inputs.client, inputs.dry_run))
                guards.register_position_change(f"{action.symbol}_carry", -action.notional_usd)
                if not inputs.dry_run:
                    notify_trade(
                        action="close_carry", symbol=action.symbol, side="Sell",
                        qty=f"{action.spot_qty:.6g}", notional_usd=action.notional_usd,
                        reason=action.reason,
                    )
            elif action.kind == "open_carry":
                all_orders.extend(_place_carry_open(action, inputs.config, inputs.client,
                                                    cfg.carry_leverage, inputs.dry_run))
                guards.register_position_change(f"{action.symbol}_carry", action.notional_usd)
                if not inputs.dry_run:
                    notify_trade(
                        action="open_carry", symbol=action.symbol, side="Buy",
                        qty=f"{action.spot_qty:.6g}", notional_usd=action.notional_usd,
                        reason=action.reason,
                        extra={"Leverage (perp)": f"{cfg.carry_leverage:.0f}x"},
                    )
            elif action.kind == "open_ema":
                all_orders.extend(_place_ema_open(
                    action, inputs.config, inputs.client, inputs.dry_run,
                    atr=ema_state["atr"], stop_mult=cfg.ema_params.atr_stop_mult,
                ))
                guards.register_position_change(f"{action.symbol}_ema", action.notional_usd)
                if not inputs.dry_run:
                    stop_estimate = action.spot_price - cfg.ema_params.atr_stop_mult * ema_state["atr"]
                    notify_trade(
                        action="open_ema", symbol=action.symbol, side="Buy",
                        qty=f"{action.spot_qty:.6g}", notional_usd=action.notional_usd,
                        reason=action.reason,
                        extra={
                            "Entry (approx)": f"${action.spot_price:,.2f}",
                            "Stop trigger":   f"${stop_estimate:,.2f}",
                        },
                    )
            elif action.kind == "close_ema":
                all_orders.extend(_place_ema_close(
                    action, inputs.config, inputs.client, inputs.dry_run,
                    current=current,
                ))
                guards.register_position_change(f"{action.symbol}_ema", -action.notional_usd)
                if not inputs.dry_run:
                    notify_trade(
                        action="close_ema", symbol=action.symbol, side="Sell",
                        qty=f"{action.spot_qty:.6g}", notional_usd=action.notional_usd,
                        reason=action.reason,
                    )
        except ValueError as e:
            log.info(f"  [SKIP] {action.symbol}: {e}")

    log.info("")
    log.info(f"--- Summary ---")
    log.info(f"  Orders placed (or dry-run printed): {len(all_orders)}")
    log.info(f"  Tracked notional: ${state.total_notional_usd:,.2f}")
    return {
        "halted": False,
        "carry_target": target,
        "actions": actions,
        "orders": all_orders,
        "ema_state": ema_state,
        "total_notional_usd": state.total_notional_usd,
    }
