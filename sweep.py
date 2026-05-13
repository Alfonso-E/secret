"""Parameter sweep over EMA pairs for the Phase 1 strategy.

Reuses the cached 1h history and the same backtester. For each (fast, slow)
pair we run the strategy twice — with and without the EMA(200) trend filter
— so we can see the filter's effect directly.

Results are printed as a table sorted by total return (descending) and
written to data/sweep_results.csv for later analysis.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import pandas as pd

from backtest import BacktestConfig, compute_metrics, run_backtest
from bybit_klines import load_or_fetch
from strategy import StrategyParams, generate_signals

DATA_PATH = Path(__file__).parent / "data" / "btcusdt_1h_spot.csv"
OUT_PATH  = Path(__file__).parent / "data" / "sweep_results.csv"

# Pairs to test. Each pair must satisfy fast < slow.
EMA_PAIRS: list[tuple[int, int]] = [
    (9,  21),
    (12, 26),
    (20, 50),
    (21, 55),
    (50, 100),
    (50, 200),
]

# Trend filter lengths to try. None = filter disabled.
TREND_FILTERS: list[int | None] = [None, 200]


def run_sweep(
    df: pd.DataFrame,
    ema_pairs: list[tuple[int, int]] = EMA_PAIRS,
    trend_filters: list[int | None] = TREND_FILTERS,
    config: BacktestConfig = BacktestConfig(),
) -> pd.DataFrame:
    """Sweep the strategy across (ema_fast, ema_slow) x trend_filter combinations."""
    rows: list[dict[str, object]] = []
    for (fast, slow), trend in product(ema_pairs, trend_filters):
        params = StrategyParams(ema_fast=fast, ema_slow=slow, trend_ema=trend)
        signals = generate_signals(df, params)
        result = run_backtest(signals, params, config)
        m = compute_metrics(result)
        rows.append({
            "ema_pair":     f"{fast}/{slow}",
            "trend_filter": str(trend) if trend else "off",
            "trades":       int(m["num_trades"]),
            "return_%":     round(m["total_return_pct"], 2),
            "max_dd_%":     round(m["max_drawdown_pct"], 2),
            "win_rate_%":   round(m["win_rate_pct"], 2),
            "profit_factor": round(m["profit_factor"], 2),
            "sharpe":       round(m["sharpe"], 2),
            "stops":        int(m["stop_exits"]),
        })
    return pd.DataFrame(rows).sort_values("return_%", ascending=False).reset_index(drop=True)


def main() -> None:
    df = load_or_fetch(
        DATA_PATH, symbol="BTCUSDT", interval="60", category="spot",
        lookback="540D", refresh=False,
    )
    print(f"Data: {len(df)} bars  {df.index[0]} -> {df.index[-1]}\n")

    table = run_sweep(df)
    print(table.to_string(index=False))

    bh_return = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
    print(f"\nBuy & hold over same period: {bh_return:+.2f}%")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_PATH, index=False)
    print(f"\nResults -> {OUT_PATH}")


if __name__ == "__main__":
    main()
