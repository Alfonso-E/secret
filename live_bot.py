"""Live bot entrypoint — single pass by default, continuous with --loop.

Modes:
  --mock-account    Use a synthetic $10k account; skip Bitget credentials.
                    Useful for verifying logic without funding the demo wallet.
  (default auth)    Load credentials from .env / env vars, fetch real demo balance.

  --live            Actually send orders. Without this flag, every order is dry-run.
  --loop            Run continuously: wake on the next funding (8h) or hour (EMA)
                    boundary, evaluate, sleep again. Ctrl+C to stop cleanly.

Examples:
  run.bat live_bot.py --mock-account              (no creds, no orders, one pass)
  run.bat live_bot.py                              (real demo balance, one pass, dry-run)
  run.bat live_bot.py --loop                       (real balance, continuous, still dry-run)
  run.bat live_bot.py --loop --live                (CONTINUOUS LIVE DEMO TRADING — final mode)
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from bitget_client import BitgetClient
from bitget_funding import load_or_fetch_funding
from bitget_klines import fetch_klines, load_or_fetch
from config import BitgetConfig, load_bitget_config
from logger import heartbeat, log
from notify import notify_daily_summary, notify_error, notify_halt, notify_test
from reconcile import PositionView, fetch_state
from safety import SafetyLimits, SessionState
from scheduler import (
    CARRY_UNIVERSE, EMA_SYMBOL, SchedulerInputs, StrategyConfig, evaluate_once,
)
from state import load_state, record_halt, save_state, update_for_cycle

DATA_DIR = Path(__file__).parent / "data"

# When to wake up. Funding settles at the 8h marks; EMA is checked hourly.
FUNDING_HOURS = (0, 8, 16)


def _latest_price(symbol: str, category: str, attempts: int = 3) -> float | None:
    """Latest price from 1m candles. Falls back to 1h if 1m is empty. None on failure."""
    last_err: Exception | None = None
    for attempt in range(attempts):
        try:
            df = fetch_klines(symbol=symbol, interval="1", category=category, limit=1)
            if df.empty:
                df = fetch_klines(symbol=symbol, interval="60", category=category, limit=1)
            if not df.empty:
                return float(df["close"].iloc[-1])
        except Exception as e:
            last_err = e
            time.sleep(0.3)
    log.warning(f"  [WARN] {symbol} ({category}): could not fetch price after {attempts} attempts ({last_err})")
    return None


def _latest_spot_price(symbol: str) -> float | None:
    return _latest_price(symbol, "spot")


def _latest_perp_price(symbol: str) -> float | None:
    return _latest_price(symbol, "linear")


def _build_funding_panel(symbols: list[str]) -> pd.DataFrame:
    series: dict[str, pd.Series] = {}
    for sym in symbols:
        path = DATA_DIR / f"funding_{sym.lower()}.csv"
        df = load_or_fetch_funding(path, symbol=sym, start="2024-06-01")
        if not df.empty:
            series[sym] = df["funding_rate"]
    return pd.concat(series, axis=1).sort_index()


def _load_market_data(symbols_with_btc: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    log.info("  Loading market data...")
    btc_klines = load_or_fetch(
        DATA_DIR / "btcusdt_1h_spot.csv",
        symbol="BTCUSDT", interval="60", category="spot",
        lookback="540D", refresh=True,
    )
    funding_panel = _build_funding_panel(CARRY_UNIVERSE)
    spot_prices, perp_prices = {}, {}
    for sym in symbols_with_btc:
        sp = _latest_spot_price(sym)
        pp = _latest_perp_price(sym)
        if sp is not None:
            spot_prices[sym] = sp
        if pp is not None:
            perp_prices[sym] = pp
    return btc_klines, funding_panel, spot_prices, perp_prices


def _build_inputs(
    *,
    cfg: BitgetConfig,
    client: BitgetClient | None,
    mock_account: bool,
    capital_override: float,
    carry_frac: float,
    dry_run: bool,
) -> SchedulerInputs:
    symbols = list(CARRY_UNIVERSE) + [EMA_SYMBOL]
    btc_klines, funding_panel, spot_prices, perp_prices = _load_market_data(symbols)

    # Capital: from real wallet (if available) or fall back to override
    if mock_account or client is None:
        total_cap = capital_override
        mock_pv = PositionView(total_equity_usd=capital_override)
    else:
        from bitget_account import get_wallet_balance
        snap = get_wallet_balance(client)
        total_cap = snap.total_equity_usd if snap.total_equity_usd > 0 else capital_override
        mock_pv = None

    strategy = StrategyConfig(total_capital_usd=total_cap, carry_fraction=carry_frac)
    limits = SafetyLimits()
    state = SessionState()
    state.update_equity(total_cap)

    return SchedulerInputs(
        config=cfg, strategy=strategy, limits=limits, state=state, client=client,
        spot_prices=spot_prices, perp_prices=perp_prices,
        btc_klines=btc_klines, funding_panel=funding_panel,
        mock_position_view=mock_pv, dry_run=dry_run,
    )


# ---------- Continuous loop ----------

def _next_wake_time(now: datetime) -> tuple[datetime, str]:
    """Earliest of: next funding moment, top of next hour, in UTC."""
    next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    candidates: list[tuple[datetime, str]] = [(next_hour, "hourly-EMA-check")]
    base = now.replace(minute=0, second=0, microsecond=0)
    for h in FUNDING_HOURS:
        for delta in range(2):
            t = base.replace(hour=h) + timedelta(days=delta)
            if t > now:
                candidates.append((t, f"funding-{h:02d}:00-UTC"))
                break
    return min(candidates, key=lambda c: c[0])


class _Stopper:
    def __init__(self): self.requested = False
    def request(self, *_): log.info("\n  [SIGNAL] shutdown requested — finishing current cycle then exiting."); self.requested = True


def run_continuous(
    cfg: BitgetConfig, client: BitgetClient | None,
    mock_account: bool, capital: float, carry_frac: float, dry_run: bool,
) -> int:
    """Loop forever (or until Ctrl+C), evaluating at each event boundary."""
    stopper = _Stopper()
    try:
        signal.signal(signal.SIGINT, stopper.request)
        signal.signal(signal.SIGTERM, stopper.request)
    except Exception:
        pass

    cycle = 0
    while not stopper.requested:
        cycle += 1
        now = datetime.now(timezone.utc)
        log.info("")
        log.info("#" * 92)
        log.info(f"# CYCLE {cycle}  at {now.isoformat(timespec='seconds')}")
        log.info("#" * 92)
        try:
            inputs = _build_inputs(
                cfg=cfg, client=client, mock_account=mock_account,
                capital_override=capital, carry_frac=carry_frac, dry_run=dry_run,
            )
            result = evaluate_once(inputs)
            heartbeat(extra=f"cycle={cycle} target={result.get('carry_target')} actions={len(result.get('actions', []))}")
            if result.get("halted"):
                log.info(f"  [HALTED] {result.get('reason')} — exiting loop.")
                return 1
        except Exception as e:
            log.info(f"  [ERROR] cycle {cycle} crashed: {type(e).__name__}: {e}")
            log.info("  [INFO] continuing — next cycle will retry.")
            if not dry_run:
                notify_error(f"{type(e).__name__}: {e}", cycle=cycle)

        if stopper.requested:
            break

        wake, label = _next_wake_time(datetime.now(timezone.utc))
        sleep_s = max(1.0, (wake - datetime.now(timezone.utc)).total_seconds())
        log.info(f"\n  Next wake: {wake.isoformat(timespec='seconds')} UTC  ({label})  "
              f"sleeping {sleep_s:,.0f}s")
        # sleep in small steps so Ctrl+C is responsive
        end = time.time() + sleep_s
        while time.time() < end and not stopper.requested:
            time.sleep(min(2.0, end - time.time()))

    log.info("\n  Loop exited cleanly. Positions left as-is on the exchange.")
    return 0


# ---------- CLI ----------

def main() -> int:
    p = argparse.ArgumentParser(description="Live bot entrypoint (dry-run by default)")
    p.add_argument("--mock-account", action="store_true",
                   help="Use synthetic $10k account; skip Bitget credentials.")
    p.add_argument("--live", action="store_true",
                   help="Actually send orders. DEFAULT IS DRY-RUN.")
    p.add_argument("--loop", action="store_true",
                   help="Run continuously; sleep until next funding/hour event.")
    p.add_argument("--capital", type=float, default=10_000.0,
                   help="Override total capital in USD (default 10000; ignored if real balance found).")
    p.add_argument("--carry-frac", type=float, default=0.9,
                   help="Carry portion of capital (0.0 to 1.0). Default 0.9 = 90%% carry / 10%% EMA.")
    args = p.parse_args()

    dry_run = not args.live
    if args.live and args.mock_account:
        log.info("[FAIL] --live and --mock-account are mutually exclusive.", file=sys.stderr)
        return 2

    if args.mock_account:
        cfg = BitgetConfig(api_key="MOCK", api_secret="MOCK", passphrase="MOCK", env="demo")
        client: BitgetClient | None = None
        log.info(f"[MODE] mock-account  base_url={cfg.base_url}  (no auth, no orders sent)")
    else:
        try:
            cfg = load_bitget_config()
        except RuntimeError as e:
            log.info(f"[FAIL] {e}", file=sys.stderr)
            log.info("(Tip: pass --mock-account to run without credentials.)", file=sys.stderr)
            return 2
        client = BitgetClient(cfg)
        log.info(f"[MODE] {cfg.env}  base_url={cfg.base_url}  dry_run_orders={dry_run}  loop={args.loop}")

    if args.loop:
        return run_continuous(
            cfg=cfg, client=client, mock_account=args.mock_account,
            capital=args.capital, carry_frac=args.carry_frac, dry_run=dry_run,
        )

    # Single-pass mode
    persistent = load_state()
    log.info(f"  Persistent state: runs={persistent.total_runs}  "
             f"peak_equity=${persistent.peak_equity_usd:,.2f}  "
             f"peak_at={persistent.peak_equity_utc or '(never)'}")
    current_eq = 0.0  # set before any exception so finally block can still record
    try:
        inputs = _build_inputs(
            cfg=cfg, client=client, mock_account=args.mock_account,
            capital_override=args.capital, carry_frac=args.carry_frac, dry_run=dry_run,
        )
        current_eq = inputs.strategy.total_capital_usd

        # Cross-run drawdown check — replaces the broken session-state version.
        dd = persistent.drawdown_from_peak(current_eq)
        max_dd_allowed = -inputs.limits.max_daily_loss_pct / 100.0
        if persistent.peak_equity_usd > 0 and dd <= max_dd_allowed:
            reason = (f"Cross-run drawdown {dd*100:+.2f}% from peak "
                      f"${persistent.peak_equity_usd:,.2f} ({persistent.peak_equity_utc}) "
                      f"breached limit {max_dd_allowed*100:+.2f}%")
            log.info(f"  [HALT] {reason}")
            if not dry_run:
                notify_halt(reason)
            persistent = record_halt(persistent, reason)
            save_state(persistent)
            return 1

        result = evaluate_once(inputs)
    except Exception as e:
        log.info(f"  [ERROR] single-pass crashed: {type(e).__name__}: {e}")
        if not dry_run:
            notify_error(f"{type(e).__name__}: {e}")
        raise
    finally:
        # Always bookkeep run count + peak, even on crash, so we don't lose history.
        try:
            persistent = update_for_cycle(persistent, current_eq)
            save_state(persistent)
        except Exception as save_err:
            log.warning(f"  Could not save persistent state: {save_err}")
    heartbeat(extra=f"single_pass target={result.get('carry_target')} actions={len(result.get('actions', []))}")

    # Daily check-in: send once when the cron lands in the 0:xx UTC hour.
    # GH Actions cron has 5-15 min jitter, so anchor on the hour rather than exact minute.
    now_utc = datetime.now(timezone.utc)
    if not dry_run and now_utc.hour == 0:
        equity = inputs.strategy.total_capital_usd
        ema_holding = False
        extra_lines: list[str] = []
        if client is not None:
            try:
                pv = fetch_state(client)
                ema_holding = pv.spot_balances.get("BTC", 0.0) > 0
                if pv.total_equity_usd > 0:
                    equity = pv.total_equity_usd

                # Drawdown vs peak (helps you see when the bot has bled vs is at fresh highs)
                if persistent.peak_equity_usd > 0:
                    dd = persistent.drawdown_from_peak(equity) * 100
                    extra_lines.append(
                        f"Peak ${persistent.peak_equity_usd:,.2f} ({persistent.peak_equity_utc[:10]}) "
                        f"| DD {dd:+.2f}%"
                    )

                # Current funding rate of the held carry asset (annualized for context)
                held = next(iter(pv.perp_shorts.keys()), None)
                if held and held in inputs.funding_panel.columns:
                    smoothed = float(inputs.funding_panel[held].tail(9).mean())
                    apr = smoothed * 3 * 365 * 100
                    extra_lines.append(f"{held} funding: {smoothed:+.6f}/8h ({apr:+.2f}% APR)")

                # Liquidation buffer for the perp short
                for sym, size in pv.perp_shorts.items():
                    entry = pv.perp_entries.get(sym, 0.0)
                    cur = inputs.perp_prices.get(sym, 0.0)
                    if entry > 0 and cur > 0:
                        approx_liq = entry * 1.15  # rough for 5x isolated
                        buffer = (approx_liq - cur) / cur * 100
                        extra_lines.append(f"{sym} liq buffer ~{buffer:.1f}%")
            except Exception as e:
                extra_lines.append(f"(could not enrich: {type(e).__name__})")
        extra_lines.append(f"Runs since start: {persistent.total_runs}")

        notify_daily_summary(
            equity_usd=equity,
            carry_target=result.get("carry_target"),
            ema_holding=ema_holding,
            extra_lines=extra_lines,
        )

    if result.get("halted"):
        log.info(f"\n[HALTED] {result.get('reason')}")
        return 1
    log.info(f"\n[OK] Evaluation complete. dry_run={dry_run}")
    if dry_run:
        log.info("     No orders were actually placed. Inspect the body= lines above to verify intent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
