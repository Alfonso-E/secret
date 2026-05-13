"""Phase 2 backtester: regime-aware router.

For each bar, the regime detector classifies the market as TREND or RANGE.
  - TREND  -> EMA crossover strategy (same rules as Phase 1)
  - RANGE  -> grid bot

When the regime changes between bars, we liquidate the previous strategy's
open position(s) at the next bar's open before activating the new strategy.

Cash is shared between the two strategies. Only one strategy is "live" at a
time, so there is never overlap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest import BacktestConfig, Trade, _make_trade
from grid_strategy import (
    GridFill, GridParams, GridState, build_grid,
    grid_mark_to_market, liquidate_grid, step_grid,
)
from regime import RANGE, TREND, RegimeParams, detect_regime
from strategy import StrategyParams, generate_signals


@dataclass
class Phase2Result:
    equity:      pd.Series
    regime:      pd.Series
    ema_trades:  list[Trade]
    grid_fills:  list[GridFill]
    grid_cycles: int
    ema_params:    StrategyParams
    grid_params:   GridParams
    regime_params: RegimeParams
    config:        BacktestConfig
    buy_hold_equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


def run_phase2(
    df: pd.DataFrame,
    ema_params:    StrategyParams = StrategyParams(),
    grid_params:   GridParams     = GridParams(),
    regime_params: RegimeParams   = RegimeParams(),
    config:        BacktestConfig = BacktestConfig(),
    range_mode:    str            = "grid",   # "grid" | "cash"
) -> Phase2Result:
    signals = generate_signals(df, ema_params)
    signals = detect_regime(signals, regime_params)

    warmup_cols = [c for c in ("ema_fast", "ema_slow", "rsi", "atr", "trend_ema", "adx") if c in signals.columns]
    sig = signals.dropna(subset=warmup_cols).copy()

    times    = sig.index
    opens    = sig["open"].to_numpy()
    highs    = sig["high"].to_numpy()
    lows     = sig["low"].to_numpy()
    closes   = sig["close"].to_numpy()
    atrs     = sig["atr"].to_numpy()
    entries  = sig["long_entry"].to_numpy()
    exits    = sig["long_exit"].to_numpy()
    regimes  = sig["regime"].to_numpy()

    cash = config.initial_capital

    # EMA strategy state
    ema_size = 0.0
    ema_entry_price = 0.0
    ema_entry_time: pd.Timestamp | None = None
    ema_entry_idx = -1
    ema_stop = 0.0
    ema_pending: str | None = None
    ema_pending_stop_dist = 0.0
    ema_trades: list[Trade] = []

    # Grid strategy state
    grid_state: GridState | None = None
    grid_fills_all: list[GridFill] = []

    equity_curve = np.empty(len(sig), dtype=float)
    prev_regime: str = regimes[0]  # treat first bar as already-in this regime

    for i in range(len(sig)):
        bar_time = times[i]
        regime_now = regimes[i]

        # 1) Regime change at this bar's open: liquidate previous-mode positions.
        if regime_now != prev_regime:
            if prev_regime == TREND and ema_size > 0.0:
                exit_price = opens[i]
                proceeds = ema_size * exit_price
                cash += proceeds - proceeds * config.fee_rate
                ema_trades.append(_make_trade(
                    ema_entry_time, ema_entry_price, bar_time, exit_price,
                    ema_size, config.fee_rate, "regime_change", i - ema_entry_idx,
                ))
                ema_size = 0.0
                ema_pending = None
            elif prev_regime == RANGE and grid_state is not None:
                fills, cash_delta = liquidate_grid(grid_state, bar_time, opens[i], config.fee_rate)
                cash += cash_delta
                grid_fills_all.extend(fills)
                grid_state = None

            if regime_now == RANGE and range_mode == "grid":
                grid_state = build_grid(
                    anchor_price=opens[i],
                    anchor_time=bar_time,
                    atr_at_anchor=atrs[i],
                    available_cash=cash,
                    params=grid_params,
                )

        # 2) TREND mode: process pending EMA actions + intra-bar stop.
        if regime_now == TREND:
            if ema_pending == "open" and ema_size == 0.0:
                entry_price = opens[i]
                notional = cash * config.position_fraction
                ema_size = notional / entry_price
                cash -= notional + notional * config.fee_rate
                ema_stop = entry_price - ema_pending_stop_dist
                ema_entry_price = entry_price
                ema_entry_time = bar_time
                ema_entry_idx = i
                ema_pending = None
            elif ema_pending == "close" and ema_size > 0.0:
                exit_price = opens[i]
                proceeds = ema_size * exit_price
                cash += proceeds - proceeds * config.fee_rate
                ema_trades.append(_make_trade(
                    ema_entry_time, ema_entry_price, bar_time, exit_price,
                    ema_size, config.fee_rate, "signal", i - ema_entry_idx,
                ))
                ema_size = 0.0
                ema_pending = None

            if ema_size > 0.0 and lows[i] <= ema_stop:
                exit_price = ema_stop
                proceeds = ema_size * exit_price
                cash += proceeds - proceeds * config.fee_rate
                ema_trades.append(_make_trade(
                    ema_entry_time, ema_entry_price, bar_time, exit_price,
                    ema_size, config.fee_rate, "stop", i - ema_entry_idx,
                ))
                ema_size = 0.0
                ema_pending = None

            # mark to market for equity curve
            mark_value = ema_size * closes[i]

            # set up next-bar action from this bar's close signals
            if ema_size == 0.0 and entries[i]:
                ema_pending = "open"
                ema_pending_stop_dist = atrs[i] * ema_params.atr_stop_mult
            elif ema_size > 0.0 and exits[i]:
                ema_pending = "close"

        # 3) RANGE mode: walk the bar through the grid.
        elif regime_now == RANGE and grid_state is not None:
            fills, cash_delta = step_grid(
                state=grid_state, bar_time=bar_time,
                bar_high=highs[i], bar_low=lows[i], fee_rate=config.fee_rate,
            )
            cash += cash_delta
            grid_fills_all.extend(fills)
            mark_value = grid_mark_to_market(grid_state, closes[i])
        else:
            mark_value = 0.0

        equity_curve[i] = cash + mark_value
        prev_regime = regime_now

    # Final liquidation at last close (for fair end-of-period MTM)
    final_cash = cash
    if grid_state is not None:
        fills, cash_delta = liquidate_grid(grid_state, times[-1], closes[-1], config.fee_rate)
        final_cash += cash_delta
        grid_fills_all.extend(fills)
    if ema_size > 0.0 and ema_entry_time is not None:
        exit_price = closes[-1]
        proceeds = ema_size * exit_price
        final_cash += proceeds - proceeds * config.fee_rate
        ema_trades.append(_make_trade(
            ema_entry_time, ema_entry_price, times[-1], exit_price,
            ema_size, config.fee_rate, "end_of_data", len(sig) - 1 - ema_entry_idx,
        ))
    equity_curve[-1] = final_cash

    equity = pd.Series(equity_curve, index=times, name="equity")
    regime_series = pd.Series(regimes, index=times, name="regime")
    bh = (closes / closes[0]) * config.initial_capital
    buy_hold_eq = pd.Series(bh, index=times, name="buy_hold")
    grid_cycles = grid_state.cycles if grid_state else sum(1 for f in grid_fills_all if f.side == "sell")

    return Phase2Result(
        equity=equity, regime=regime_series,
        ema_trades=ema_trades, grid_fills=grid_fills_all,
        grid_cycles=grid_cycles,
        ema_params=ema_params, grid_params=grid_params,
        regime_params=regime_params, config=config,
        buy_hold_equity=buy_hold_eq,
    )


def compute_phase2_metrics(result: Phase2Result) -> dict[str, float]:
    eq = result.equity
    cfg = result.config

    total_return = eq.iloc[-1] / cfg.initial_capital - 1.0
    bars = len(eq)
    cagr = (eq.iloc[-1] / cfg.initial_capital) ** (cfg.bars_per_year / max(bars, 1)) - 1.0
    drawdown = eq / eq.cummax() - 1.0
    max_dd = drawdown.min()
    bar_returns = eq.pct_change().dropna()
    sharpe = (bar_returns.mean() / bar_returns.std()) * np.sqrt(cfg.bars_per_year) if bar_returns.std() > 0 else 0.0

    ema_n = len(result.ema_trades)
    ema_wins = [t for t in result.ema_trades if t.pnl_usd > 0]
    ema_win_rate = len(ema_wins) / ema_n if ema_n else 0.0
    grid_buys = sum(1 for f in result.grid_fills if f.side == "buy")
    grid_sells = sum(1 for f in result.grid_fills if f.side == "sell")
    grid_pnl = sum(f.pnl for f in result.grid_fills if f.side == "sell")

    regime_counts = result.regime.value_counts()
    pct_trend = regime_counts.get(TREND, 0) / len(result.regime) * 100
    pct_range = regime_counts.get(RANGE, 0) / len(result.regime) * 100

    bh_return = result.buy_hold_equity.iloc[-1] / cfg.initial_capital - 1.0

    return {
        "total_return_pct":   total_return * 100,
        "cagr_pct":           cagr * 100,
        "max_drawdown_pct":   max_dd * 100,
        "sharpe":             sharpe,
        "final_equity":       eq.iloc[-1],
        "buy_hold_return_pct": bh_return * 100,
        "ema_trades":         ema_n,
        "ema_win_rate_pct":   ema_win_rate * 100,
        "grid_buys":          grid_buys,
        "grid_sells":         grid_sells,
        "grid_pnl_usd":       grid_pnl,
        "pct_time_trend":     pct_trend,
        "pct_time_range":     pct_range,
    }
