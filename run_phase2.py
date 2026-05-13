"""Run Phase 2 (regime-routed EMA + grid) on both windows and compare to Phase 1."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest import BacktestConfig, compute_metrics, run_backtest
from bybit_klines import load_or_fetch, load_or_fetch_range
from grid_strategy import GridParams
from phase2_backtest import compute_phase2_metrics, run_phase2
from regime import RegimeParams
from strategy import StrategyParams, generate_signals

DATA_DIR = Path(__file__).parent / "data"


def phase1_metrics(df: pd.DataFrame, params: StrategyParams) -> dict[str, float]:
    sig = generate_signals(df, params)
    return _trim_metrics(_phase1(sig, params))


def _phase1(sig: pd.DataFrame, params: StrategyParams) -> dict[str, float]:
    result = run_backtest(sig, params, BacktestConfig())
    from backtest import compute_metrics as _cm
    return _cm(result)


def _trim_metrics(m: dict[str, float]) -> dict[str, float]:
    return m


def summarize_phase2(label: str, df: pd.DataFrame) -> None:
    bh = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100

    # Phase 1 reference: best two from earlier sweeps
    p1_9_21 = phase1_metrics(df, StrategyParams(ema_fast=9, ema_slow=21, trend_ema=200))
    p1_50_200 = phase1_metrics(df, StrategyParams(ema_fast=50, ema_slow=200, trend_ema=200))

    # Phase 2 variants — both range_modes for each EMA pair
    p2_9_21_grid = compute_phase2_metrics(run_phase2(
        df, ema_params=StrategyParams(ema_fast=9, ema_slow=21, trend_ema=200),
        grid_params=GridParams(n_levels=10, spacing_atr_mult=0.5),
        regime_params=RegimeParams(), range_mode="grid",
    ))
    p2_9_21_cash = compute_phase2_metrics(run_phase2(
        df, ema_params=StrategyParams(ema_fast=9, ema_slow=21, trend_ema=200),
        regime_params=RegimeParams(), range_mode="cash",
    ))
    p2_50_200_grid = compute_phase2_metrics(run_phase2(
        df, ema_params=StrategyParams(ema_fast=50, ema_slow=200, trend_ema=200),
        grid_params=GridParams(n_levels=10, spacing_atr_mult=0.5),
        regime_params=RegimeParams(), range_mode="grid",
    ))
    p2_50_200_cash = compute_phase2_metrics(run_phase2(
        df, ema_params=StrategyParams(ema_fast=50, ema_slow=200, trend_ema=200),
        regime_params=RegimeParams(), range_mode="cash",
    ))

    rows = [
        {"variant": "Phase 1: 9/21 + filter",        **_pick_p1(p1_9_21)},
        {"variant": "Phase 1: 50/200 + filter",      **_pick_p1(p1_50_200)},
        {"variant": "Phase 2a: 9/21   cash-in-range", **_pick_p2(p2_9_21_cash)},
        {"variant": "Phase 2a: 50/200 cash-in-range", **_pick_p2(p2_50_200_cash)},
        {"variant": "Phase 2b: 9/21   grid-in-range", **_pick_p2(p2_9_21_grid)},
        {"variant": "Phase 2b: 50/200 grid-in-range", **_pick_p2(p2_50_200_grid)},
    ]
    table = pd.DataFrame(rows)

    print()
    print("=" * 92)
    print(f"{label}")
    print(f"  Buy & hold over this window: {bh:+.2f}%")
    print("=" * 92)
    print(table.to_string(index=False))


def _pick_p1(m: dict[str, float]) -> dict[str, object]:
    return {
        "return_%":   round(m["total_return_pct"], 2),
        "max_dd_%":   round(m["max_drawdown_pct"], 2),
        "sharpe":     round(m["sharpe"], 2),
        "trades":     int(m["num_trades"]),
        "grid_sells": "-",
        "%time_trend": "-",
    }


def _pick_p2(m: dict[str, float]) -> dict[str, object]:
    return {
        "return_%":   round(m["total_return_pct"], 2),
        "max_dd_%":   round(m["max_drawdown_pct"], 2),
        "sharpe":     round(m["sharpe"], 2),
        "trades":     int(m["ema_trades"]),
        "grid_sells": int(m["grid_sells"]),
        "%time_trend": round(m["pct_time_trend"], 1),
    }


def main() -> None:
    bull = load_or_fetch_range(
        DATA_DIR / "btcusdt_1h_spot_2023_2024.csv",
        symbol="BTCUSDT", interval="60", category="spot",
        start="2023-01-01", end="2024-12-31",
    )
    recent = load_or_fetch(
        DATA_DIR / "btcusdt_1h_spot.csv",
        symbol="BTCUSDT", interval="60", category="spot",
        lookback="540D", refresh=False,
    )
    summarize_phase2("BULL window (Jan 2023 - Dec 2024)", bull)
    summarize_phase2("RECENT window (Nov 2024 - May 2026)", recent)


if __name__ == "__main__":
    main()
