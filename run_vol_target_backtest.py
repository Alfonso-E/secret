"""Backtest volatility-targeted position sizing against the current strategy.

We compare 4 variants on each of two windows:
  A. Baseline                    — no vol targeting (current bot)
  B. Vol-targeted carry only     — scale carry notional by BTC vol multiplier
  C. Vol-targeted EMA only       — scale EMA entry notional by BTC vol multiplier
  D. Vol-targeted both           — scale both legs

The 80/20 carry/EMA capital split and 5x leverage on carry mirror the
current live config.

Target annualized vol = 0.50 (a touch below typical BTC realized vol of
55-70%) — multiplier averages slightly above 1.0 most of the time, scaling
down hard during pumps/dumps when BTC vol blows past 1.0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtest import BacktestConfig, compute_metrics, run_backtest
from bybit_funding import load_or_fetch_funding
from bybit_klines import load_or_fetch_range
from funding_backtest import (
    CarryConfig, backtest_cross_sectional, compute_carry_metrics,
)
from strategy import StrategyParams, generate_signals
from vol_targeting import compute_vol_multipliers

DATA_DIR = Path(__file__).parent / "data"
SYMBOLS = [
    "ETHUSDT", "SOLUSDT", "AVAXUSDT",
    "LINKUSDT", "ARBUSDT", "DOGEUSDT", "ADAUSDT",
]
LEVERAGE = 5.0
TOTAL_CAPITAL = 10_000.0
CARRY_FRAC = 0.8
TARGET_VOL = 0.50
MAX_MULT   = 1.00   # cap: scale DOWN only; never above 5x effective leverage


def _combine(carry_eq: pd.Series, ema_eq: pd.Series) -> pd.Series:
    """Sum carry and EMA equity curves on the EMA index (1h)."""
    if ema_eq.empty:
        return carry_eq
    if carry_eq.empty:
        return ema_eq
    aligned = carry_eq.reindex(ema_eq.index, method="ffill").fillna(carry_eq.iloc[0])
    return (aligned + ema_eq).rename("combined")


def _metrics(equity: pd.Series, initial_capital: float, bars_per_year: float) -> dict:
    total_return = equity.iloc[-1] / initial_capital - 1.0
    bars = len(equity)
    cagr = (equity.iloc[-1] / initial_capital) ** (bars_per_year / max(bars, 1)) - 1.0
    dd = equity / equity.cummax() - 1.0
    max_dd = dd.min()
    ret = equity.pct_change().dropna()
    sharpe = (ret.mean() / ret.std()) * np.sqrt(bars_per_year) if ret.std() > 0 else 0.0
    return {
        "return_%":  round(total_return * 100, 2),
        "cagr_%":    round(cagr * 100, 2),
        "max_dd_%":  round(max_dd * 100, 2),
        "sharpe":    round(sharpe, 2),
        "final_$":   round(equity.iloc[-1], 2),
    }


def _evaluate(
    label: str,
    btc_klines: pd.DataFrame,
    funding: dict[str, pd.DataFrame],
    vol_mult_carry: pd.Series | None,
    vol_mult_ema:   pd.Series | None,
) -> dict:
    """Run carry + EMA backtests with optional vol multipliers and combine."""
    ccfg = CarryConfig(initial_capital=TOTAL_CAPITAL * CARRY_FRAC, perp_leverage=LEVERAGE)
    cres = backtest_cross_sectional(
        funding, smooth_periods=9, enter_threshold=0.0,
        rebalance_every=21, min_switch_advantage=0.0002,
        config=ccfg, vol_multipliers=vol_mult_carry,
    )

    ema_params = StrategyParams(ema_fast=50, ema_slow=200, trend_ema=200)
    btc_signals = generate_signals(btc_klines, ema_params)
    ecfg = BacktestConfig(initial_capital=TOTAL_CAPITAL * (1 - CARRY_FRAC))
    eres = run_backtest(btc_signals, ema_params, ecfg, vol_multipliers=vol_mult_ema)

    combined = _combine(cres.equity, eres.equity)
    bars_per_year = 365 * 24  # combined is at 1h
    m = _metrics(combined, TOTAL_CAPITAL, bars_per_year)
    return {"variant": label, **m}


def main() -> None:
    bull = load_or_fetch_range(
        DATA_DIR / "btcusdt_1h_spot_2023_2024.csv",
        symbol="BTCUSDT", interval="60", category="spot",
        start="2023-01-01", end="2024-12-31",
    )
    recent = load_or_fetch_range(
        DATA_DIR / "btcusdt_1h_spot.csv",
        symbol="BTCUSDT", interval="60", category="spot",
        start="2024-11-01", end="2026-05-12",
    )
    funding_full: dict[str, pd.DataFrame] = {}
    for sym in SYMBOLS:
        path = DATA_DIR / f"funding_{sym.lower()}.csv"
        df = load_or_fetch_funding(path, symbol=sym, category="linear", start="2023-01-01")
        if not df.empty:
            funding_full[sym] = df

    print(f"BTC bull window:   {len(bull)} bars  {bull.index[0]} -> {bull.index[-1]}")
    print(f"BTC recent window: {len(recent)} bars  {recent.index[0]} -> {recent.index[-1]}")
    print(f"Funding symbols loaded: {list(funding_full.keys())}")

    # Vol multipliers per window — capped at 1.0 to match the live bot's
    # safety constraint (no scale-up above configured leverage).
    mult_bull   = compute_vol_multipliers(bull,   target_annualized_vol=TARGET_VOL, max_mult=MAX_MULT)
    mult_recent = compute_vol_multipliers(recent, target_annualized_vol=TARGET_VOL, max_mult=MAX_MULT)

    print(f"\nVol multiplier stats (BULL):")
    print(f"  mean={mult_bull.mean():.3f}  median={mult_bull.median():.3f}  "
          f"min={mult_bull.min():.3f}  max={mult_bull.max():.3f}")
    print(f"Vol multiplier stats (RECENT):")
    print(f"  mean={mult_recent.mean():.3f}  median={mult_recent.median():.3f}  "
          f"min={mult_recent.min():.3f}  max={mult_recent.max():.3f}")

    for label, btc, mult, window_name in [
        (None, bull,   mult_bull,   "BULL WINDOW   2023-01 -> 2024-12"),
        (None, recent, mult_recent, "RECENT WINDOW 2024-11 -> 2026-05"),
    ]:
        # Subset funding to this window
        funding_window = {
            s: d.loc[btc.index[0]:btc.index[-1]] for s, d in funding_full.items()
        }
        rows = [
            _evaluate("A: baseline (no vol)",       btc, funding_window, None,  None),
            _evaluate("B: vol-target carry only",   btc, funding_window, mult,  None),
            _evaluate("C: vol-target EMA only",     btc, funding_window, None,  mult),
            _evaluate("D: vol-target both",         btc, funding_window, mult,  mult),
        ]
        print()
        print("=" * 92)
        print(f"{window_name}  (target vol {TARGET_VOL*100:.0f}% annualized)")
        print("=" * 92)
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
