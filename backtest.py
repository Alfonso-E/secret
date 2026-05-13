"""Event-driven backtest engine for long-only spot strategies.

Execution model:
  - Signals are evaluated at bar close.
  - Entry/exit triggered by a signal execute on the NEXT bar's open
    (this is what avoids look-ahead bias).
  - ATR stop-loss is monitored intra-bar: if low <= stop_price, exit at stop_price.
  - Fees are applied to both entry and exit notional.

The engine is deliberately single-asset / single-position. No leverage,
no shorting, no pyramiding. That matches Phase 1 of the research doc.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from strategy import StrategyParams


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 10_000.0
    fee_rate: float = 0.001            # Bybit spot taker fee, 0.1%
    position_fraction: float = 1.0     # fraction of equity to deploy per trade
    bars_per_year: float = 365 * 24    # 1h bars; used for annualization


@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    size: float                # coin units
    pnl_usd: float
    return_pct: float          # net of fees
    exit_reason: str           # "stop" | "signal"
    bars_held: int


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: list[Trade]
    params: StrategyParams
    config: BacktestConfig
    buy_hold_equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


def run_backtest(
    signals: pd.DataFrame,
    params: StrategyParams,
    config: BacktestConfig = BacktestConfig(),
) -> BacktestResult:
    """Walk the dataframe forward, simulate trades, return equity + trade log."""
    needed = {"open", "high", "low", "close", "atr", "long_entry", "long_exit"}
    missing = needed - set(signals.columns)
    if missing:
        raise ValueError(f"Missing columns in signals: {missing}")

    warmup_cols = [c for c in ("ema_fast", "ema_slow", "rsi", "atr", "trend_ema") if c in signals.columns]
    df = signals.dropna(subset=warmup_cols).copy()

    times    = df.index
    opens    = df["open"].to_numpy()
    highs    = df["high"].to_numpy()
    lows     = df["low"].to_numpy()
    closes   = df["close"].to_numpy()
    atrs     = df["atr"].to_numpy()
    entries  = df["long_entry"].to_numpy()
    exits    = df["long_exit"].to_numpy()

    cash = config.initial_capital
    size = 0.0
    entry_price = 0.0
    entry_time: pd.Timestamp | None = None
    entry_index = -1
    stop_price = 0.0

    pending: str | None = None       # "open" | "close" | None
    pending_stop_dist = 0.0          # ATR distance captured at signal bar

    trades: list[Trade] = []
    equity_curve = np.empty(len(df), dtype=float)

    for i in range(len(df)):
        # 1) Execute pending actions at this bar's open.
        if pending == "open" and size == 0.0:
            entry_price = opens[i]
            notional = cash * config.position_fraction
            size = notional / entry_price
            cash -= notional
            cash -= notional * config.fee_rate
            stop_price = entry_price - pending_stop_dist
            entry_time = times[i]
            entry_index = i
            pending = None
        elif pending == "close" and size > 0.0:
            exit_price = opens[i]
            proceeds = size * exit_price
            cash += proceeds - proceeds * config.fee_rate
            trades.append(_make_trade(
                entry_time, entry_price, times[i], exit_price,
                size, config.fee_rate, "signal", i - entry_index,
            ))
            size = 0.0
            pending = None

        # 2) Intra-bar stop check (only relevant if we're in a position).
        if size > 0.0 and lows[i] <= stop_price:
            exit_price = stop_price
            proceeds = size * exit_price
            cash += proceeds - proceeds * config.fee_rate
            trades.append(_make_trade(
                entry_time, entry_price, times[i], exit_price,
                size, config.fee_rate, "stop", i - entry_index,
            ))
            size = 0.0
            pending = None  # cancel any pending close — we're already out

        # 3) Mark-to-market for equity curve.
        equity_curve[i] = cash + size * closes[i]

        # 4) Read this bar's close-time signals to set up next bar's action.
        if size == 0.0 and entries[i]:
            pending = "open"
            pending_stop_dist = atrs[i] * params.atr_stop_mult
        elif size > 0.0 and exits[i]:
            pending = "close"

    equity = pd.Series(equity_curve, index=times, name="equity")
    buy_hold = (closes / closes[0]) * config.initial_capital
    buy_hold_eq = pd.Series(buy_hold, index=times, name="buy_hold")
    return BacktestResult(equity=equity, trades=trades, params=params, config=config, buy_hold_equity=buy_hold_eq)


def _make_trade(
    entry_time: pd.Timestamp, entry_price: float,
    exit_time: pd.Timestamp,  exit_price: float,
    size: float, fee_rate: float, reason: str, bars_held: int,
) -> Trade:
    gross_pnl = (exit_price - entry_price) * size
    fees = (entry_price + exit_price) * size * fee_rate
    pnl = gross_pnl - fees
    ret = (exit_price / entry_price - 1.0) - 2 * fee_rate
    return Trade(
        entry_time=entry_time, entry_price=entry_price,
        exit_time=exit_time,   exit_price=exit_price,
        size=size, pnl_usd=pnl, return_pct=ret,
        exit_reason=reason, bars_held=bars_held,
    )


def compute_metrics(result: BacktestResult) -> dict[str, float]:
    eq = result.equity
    cfg = result.config
    trades = result.trades

    total_return = eq.iloc[-1] / cfg.initial_capital - 1.0
    bars = len(eq)
    cagr = (eq.iloc[-1] / cfg.initial_capital) ** (cfg.bars_per_year / max(bars, 1)) - 1.0

    rolling_max = eq.cummax()
    drawdown = eq / rolling_max - 1.0
    max_dd = drawdown.min()

    bar_returns = eq.pct_change().dropna()
    sharpe = (bar_returns.mean() / bar_returns.std()) * np.sqrt(cfg.bars_per_year) if bar_returns.std() > 0 else 0.0

    if trades:
        wins = [t for t in trades if t.pnl_usd > 0]
        losses = [t for t in trades if t.pnl_usd <= 0]
        win_rate = len(wins) / len(trades)
        avg_win = float(np.mean([t.return_pct for t in wins])) if wins else 0.0
        avg_loss = float(np.mean([t.return_pct for t in losses])) if losses else 0.0
        gross_win = sum(t.pnl_usd for t in wins)
        gross_loss = -sum(t.pnl_usd for t in losses)
        profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")
        stop_exits = sum(1 for t in trades if t.exit_reason == "stop")
    else:
        win_rate = avg_win = avg_loss = profit_factor = stop_exits = 0.0

    bh_return = result.buy_hold_equity.iloc[-1] / cfg.initial_capital - 1.0

    return {
        "total_return_pct":   total_return * 100,
        "cagr_pct":           cagr * 100,
        "max_drawdown_pct":   max_dd * 100,
        "sharpe":             sharpe,
        "num_trades":         len(trades),
        "win_rate_pct":       win_rate * 100,
        "avg_win_pct":        avg_win * 100,
        "avg_loss_pct":       avg_loss * 100,
        "profit_factor":      profit_factor,
        "stop_exits":         stop_exits,
        "buy_hold_return_pct": bh_return * 100,
        "final_equity":       eq.iloc[-1],
    }
