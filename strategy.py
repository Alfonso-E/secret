"""Signal generation for the EMA crossover strategy.

Phase 1 (per the research doc):
  - Long entry  : EMA(fast) crosses above EMA(slow) AND RSI > rsi_threshold
  - Long exit   : EMA(fast) crosses below EMA(slow)  -- a 'death cross'
  - Stop loss   : entry_price - atr_stop_mult * ATR(atr_length) at entry
  - Long-only (spot, no shorting)

All signals are computed using values known at the close of each bar.
The backtester executes on the *next* bar's open to avoid look-ahead bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from indicators import atr, ema, rsi


@dataclass(frozen=True)
class StrategyParams:
    ema_fast: int = 9
    ema_slow: int = 21
    rsi_length: int = 14
    rsi_threshold: float = 50.0
    atr_length: int = 14
    atr_stop_mult: float = 1.5
    trend_ema: int | None = 200   # regime filter: only long when close > EMA(trend_ema). None disables.


def add_indicators(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = ema(df["close"], params.ema_fast)
    out["ema_slow"] = ema(df["close"], params.ema_slow)
    out["rsi"] = rsi(df["close"], params.rsi_length)
    out["atr"] = atr(df["high"], df["low"], df["close"], params.atr_length)
    if params.trend_ema is not None:
        out["trend_ema"] = ema(df["close"], params.trend_ema)
    return out


def generate_signals(df: pd.DataFrame, params: StrategyParams = StrategyParams()) -> pd.DataFrame:
    """Return df with indicator columns plus boolean signal columns:

        golden_cross : EMA fast crossed above EMA slow this bar
        death_cross  : EMA fast crossed below EMA slow this bar
        long_entry   : golden_cross AND rsi > threshold
        long_exit    : death_cross
    """
    out = add_indicators(df, params)
    above = out["ema_fast"] > out["ema_slow"]
    prev_above = above.shift(1, fill_value=False)
    out["golden_cross"] = above & ~prev_above
    out["death_cross"] = ~above & prev_above

    entry = out["golden_cross"] & (out["rsi"] > params.rsi_threshold)
    if params.trend_ema is not None:
        entry &= out["close"] > out["trend_ema"]
    out["long_entry"] = entry
    out["long_exit"] = out["death_cross"]
    return out
