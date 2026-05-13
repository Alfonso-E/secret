"""Combined strategy: cross-sectional funding carry + directional EMA overlay.

Capital is split between two independent strategies that run in parallel:
  - Carry portion: cross-sectional rotation across 10 USDT-perp funding rates
                   at 5x leverage (the strongest pure-carry variant we measured).
  - Directional portion: Phase 1 EMA(50)/(200) crossover + EMA(200) trend filter
                         on BTCUSDT spot, the best directional variant from
                         earlier sweeps.

Window is the same OOS funding window (2023-07-30 -> 2026-05-12), so the
combined results are directly comparable to the ML-driven runs from run_ml.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtest import BacktestConfig, compute_metrics, run_backtest
from bybit_funding import load_or_fetch_funding
from bybit_klines import load_or_fetch_range
from funding_backtest import (
    CarryConfig, backtest_cross_sectional, backtest_equal_weight_basket,
    compute_carry_metrics,
)
from strategy import StrategyParams, generate_signals

DATA_DIR = Path(__file__).parent / "data"
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "AVAXUSDT", "LINKUSDT", "ARBUSDT", "DOGEUSDT",
    "ADAUSDT", "INJUSDT",
]
WINDOW_START = "2023-07-30"
WINDOW_END   = "2026-05-12"
LEVERAGE = 5.0
TOTAL_CAPITAL = 10_000.0


def _combine_curves(carry_eq: pd.Series, ema_eq: pd.Series) -> pd.Series:
    """Sum two equity curves on a common time index (1h, EMA's native frequency)."""
    if ema_eq.empty and carry_eq.empty:
        raise ValueError("Both equity curves empty")
    if ema_eq.empty:
        return carry_eq
    if carry_eq.empty:
        return ema_eq
    idx = ema_eq.index
    carry_aligned = carry_eq.reindex(idx, method="ffill").fillna(carry_eq.iloc[0])
    return (carry_aligned + ema_eq).rename("combined")


def _metrics(equity: pd.Series, initial_capital: float, bars_per_year: float) -> dict[str, float]:
    total_return = equity.iloc[-1] / initial_capital - 1.0
    bars = len(equity)
    cagr = (equity.iloc[-1] / initial_capital) ** (bars_per_year / max(bars, 1)) - 1.0
    dd = equity / equity.cummax() - 1.0
    max_dd = dd.min()
    ret = equity.pct_change().dropna()
    sharpe = (ret.mean() / ret.std()) * np.sqrt(bars_per_year) if ret.std() > 0 else 0.0
    return {
        "total_return_pct":  total_return * 100,
        "cagr_pct":          cagr * 100,
        "max_drawdown_pct":  max_dd * 100,
        "sharpe":            sharpe,
        "final_equity":      equity.iloc[-1],
    }


def main() -> None:
    # 1. Load BTC 1h klines for the window
    btc_path = DATA_DIR / "btcusdt_1h_2023_07_2026_05.csv"
    btc = load_or_fetch_range(
        btc_path, symbol="BTCUSDT", interval="60", category="spot",
        start=WINDOW_START, end=WINDOW_END,
    )
    print(f"BTC 1h: {len(btc)} bars  {btc.index[0]} -> {btc.index[-1]}")

    # 2. Load funding for all assets, restrict to OOS window
    funding_by_symbol: dict[str, pd.DataFrame] = {}
    for sym in SYMBOLS:
        path = DATA_DIR / f"funding_{sym.lower()}.csv"
        df = load_or_fetch_funding(path, symbol=sym, category="linear", start="2023-01-01")
        if not df.empty:
            funding_by_symbol[sym] = df.loc[WINDOW_START:WINDOW_END]
    print(f"Funding: {len(funding_by_symbol)} symbols loaded for window")

    # 3. Build EMA signals once
    ema_params = StrategyParams(ema_fast=50, ema_slow=200, trend_ema=200)
    btc_signals = generate_signals(btc, ema_params)

    # 4. Run pure-strategy baselines for sanity check
    print()
    print("=" * 92)
    print(f"COMBINED CARRY + DIRECTIONAL  total=${TOTAL_CAPITAL:,.0f}  carry leverage={LEVERAGE:.0f}x")
    print("=" * 92)
    rows: list[dict[str, object]] = []

    # Pure carry (100% in cross-sectional) at this capital
    pure_carry_cfg = CarryConfig(initial_capital=TOTAL_CAPITAL, perp_leverage=LEVERAGE)
    pure_carry = backtest_cross_sectional(funding_by_symbol, smooth_periods=9, config=pure_carry_cfg)
    pcm = compute_carry_metrics(pure_carry)
    rows.append({
        "carry_split": "100% carry / 0% EMA",
        "return_%":  round(pcm["total_return_pct"], 2),
        "cagr_%":    round(pcm["cagr_pct"], 2),
        "max_dd_%":  round(pcm["max_drawdown_pct"], 2),
        "sharpe":    round(pcm["sharpe"], 2),
    })

    # Pure EMA (100% in Phase 1)
    pure_ema_cfg = BacktestConfig(initial_capital=TOTAL_CAPITAL)
    pure_ema = run_backtest(btc_signals, ema_params, pure_ema_cfg)
    pem = compute_metrics(pure_ema)
    rows.append({
        "carry_split": "0% carry / 100% EMA",
        "return_%":  round(pem["total_return_pct"], 2),
        "cagr_%":    round(pem["cagr_pct"], 2),
        "max_dd_%":  round(pem["max_drawdown_pct"], 2),
        "sharpe":    round(pem["sharpe"], 2),
    })

    # 5. Combined splits
    bars_per_year_hourly = 365 * 24
    for carry_frac in [0.9, 0.8, 0.7, 0.5]:
        ema_frac = 1.0 - carry_frac
        carry_cap = TOTAL_CAPITAL * carry_frac
        ema_cap   = TOTAL_CAPITAL * ema_frac

        ccfg = CarryConfig(initial_capital=carry_cap, perp_leverage=LEVERAGE)
        cres = backtest_cross_sectional(funding_by_symbol, smooth_periods=9, config=ccfg)

        ecfg = BacktestConfig(initial_capital=ema_cap)
        eres = run_backtest(btc_signals, ema_params, ecfg)

        combined = _combine_curves(cres.equity, eres.equity)
        m = _metrics(combined, TOTAL_CAPITAL, bars_per_year_hourly)
        rows.append({
            "carry_split": f"{int(carry_frac*100):d}% carry / {int(ema_frac*100):d}% EMA",
            "return_%":  round(m["total_return_pct"], 2),
            "cagr_%":    round(m["cagr_pct"], 2),
            "max_dd_%":  round(m["max_drawdown_pct"], 2),
            "sharpe":    round(m["sharpe"], 2),
        })

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    out_path = DATA_DIR / "combined_results.csv"
    table.to_csv(out_path, index=False)
    print(f"\nResults -> {out_path}")

    # --- Year-by-year breakdown of the 70/30 split ---
    print()
    print("=" * 92)
    print("YEAR-BY-YEAR BREAKDOWN  (70% carry / 30% EMA, leverage=5x)")
    print("=" * 92)

    ccfg = CarryConfig(initial_capital=TOTAL_CAPITAL * 0.7, perp_leverage=LEVERAGE)
    cres = backtest_cross_sectional(funding_by_symbol, smooth_periods=9, config=ccfg)
    ecfg = BacktestConfig(initial_capital=TOTAL_CAPITAL * 0.3)
    eres = run_backtest(btc_signals, ema_params, ecfg)
    combined = _combine_curves(cres.equity, eres.equity)

    year_rows: list[dict[str, object]] = []
    for year, year_eq in combined.groupby(combined.index.year):
        if len(year_eq) < 24 * 30:
            continue
        start_v, end_v = year_eq.iloc[0], year_eq.iloc[-1]
        ret = (end_v / start_v - 1) * 100
        dd = (year_eq / year_eq.cummax() - 1).min() * 100
        year_rows.append({
            "year":    int(year),
            "bars":    len(year_eq),
            "start_$": round(start_v, 2),
            "end_$":   round(end_v, 2),
            "return_%": round(ret, 2),
            "max_dd_%": round(dd, 2),
        })
    print(pd.DataFrame(year_rows).to_string(index=False))

    # Carry and EMA contribution separately for the same split
    print()
    print("CONTRIBUTION BREAKDOWN  (70/30 split)")
    carry_only_m = _metrics(cres.equity, TOTAL_CAPITAL * 0.7, 3 * 365)
    ema_only_m = _metrics(eres.equity, TOTAL_CAPITAL * 0.3, 365 * 24)
    print(f"  Carry portion ($7000 -> ${cres.equity.iloc[-1]:,.2f})  "
          f"return {carry_only_m['total_return_pct']:+.2f}%  CAGR {carry_only_m['cagr_pct']:+.2f}%  "
          f"max DD {carry_only_m['max_drawdown_pct']:.2f}%")
    print(f"  EMA portion   ($3000 -> ${eres.equity.iloc[-1]:,.2f})  "
          f"return {ema_only_m['total_return_pct']:+.2f}%  CAGR {ema_only_m['cagr_pct']:+.2f}%  "
          f"max DD {ema_only_m['max_drawdown_pct']:.2f}%")


if __name__ == "__main__":
    main()
