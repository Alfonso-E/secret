"""Run the funding-rate carry backtest on BTC, ETH, SOL.

Tests three strategies:
  1. always_on        : long spot + short perp from day 1, never rebalance
  2. threshold        : enter when smoothed rate > 0.0001 per 8h, exit when <= 0
  3. cross_sectional  : at each funding period, rotate into the asset with the
                         highest smoothed rate (must clear fee threshold to switch)

Window: same 2023-01-01 -> latest as the directional backtests, so results are
comparable to Phase 1.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bybit_funding import load_or_fetch_funding
from funding_backtest import (
    CarryConfig, backtest_always_on, backtest_cross_sectional,
    backtest_equal_weight_basket, backtest_threshold, compute_carry_metrics,
)

DATA_DIR = Path(__file__).parent / "data"
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "AVAXUSDT", "LINKUSDT", "ARBUSDT", "DOGEUSDT",
    "ADAUSDT", "INJUSDT",
]
START = "2023-01-01"
END = None   # to "now"
LEVERAGES_TO_TEST = [1.0, 3.0, 5.0]


def main() -> None:
    funding_by_symbol: dict[str, pd.DataFrame] = {}
    for sym in SYMBOLS:
        path = DATA_DIR / f"funding_{sym.lower()}.csv"
        try:
            df = load_or_fetch_funding(path, symbol=sym, category="linear", start=START, end=END)
            if df.empty:
                print(f"{sym}: no data, skipping")
                continue
            print(f"{sym:10s} {len(df):4d} rows  {df.index[0].date()} -> {df.index[-1].date()}  "
                  f"mean {df['funding_rate'].mean():+.6f}/8h  "
                  f"({df['funding_rate'].mean() * 3 * 365 * 100:+.2f}% APR raw)")
            funding_by_symbol[sym] = df
        except Exception as e:
            print(f"{sym}: failed -> {e}")

    print()
    rows: list[dict[str, object]] = []

    for leverage in LEVERAGES_TO_TEST:
        cfg = CarryConfig(perp_leverage=leverage)

        # Per-symbol always-on at this leverage
        for sym, df in funding_by_symbol.items():
            r = backtest_always_on(df, cfg)
            m = compute_carry_metrics(r)
            rows.append({
                "leverage": f"{leverage:.0f}x",
                "strategy": f"{sym} always-on",
                "return_%":  round(m["total_return_pct"], 2),
                "cagr_%":    round(m["cagr_pct"], 2),
                "max_dd_%":  round(m["max_drawdown_pct"], 2),
                "sharpe":    round(m["sharpe"], 2),
            })

        # Equal-weight basket of all available symbols
        eb = backtest_equal_weight_basket(funding_by_symbol, cfg)
        eb_m = compute_carry_metrics(eb)
        rows.append({
            "leverage": f"{leverage:.0f}x",
            "strategy": f"basket[{len(funding_by_symbol)} assets] always-on",
            "return_%":  round(eb_m["total_return_pct"], 2),
            "cagr_%":    round(eb_m["cagr_pct"], 2),
            "max_dd_%":  round(eb_m["max_drawdown_pct"], 2),
            "sharpe":    round(eb_m["sharpe"], 2),
        })

        # Filtered basket: drop assets with non-positive historical mean funding
        positive_universe = {
            s: d for s, d in funding_by_symbol.items()
            if d["funding_rate"].mean() > 0
        }
        if len(positive_universe) < len(funding_by_symbol):
            fb = backtest_equal_weight_basket(positive_universe, cfg)
            fb_m = compute_carry_metrics(fb)
            rows.append({
                "leverage": f"{leverage:.0f}x",
                "strategy": f"basket[{len(positive_universe)} positive-funding only]",
                "return_%":  round(fb_m["total_return_pct"], 2),
                "cagr_%":    round(fb_m["cagr_pct"], 2),
                "max_dd_%":  round(fb_m["max_drawdown_pct"], 2),
                "sharpe":    round(fb_m["sharpe"], 2),
            })

        # Cross-sectional rotation across all symbols
        cs = backtest_cross_sectional(
            funding_by_symbol, smooth_periods=9, enter_threshold=0.0, config=cfg,
        )
        cs_m = compute_carry_metrics(cs)
        rows.append({
            "leverage": f"{leverage:.0f}x",
            "strategy": f"cross-sectional[{len(funding_by_symbol)}] weekly",
            "return_%":  round(cs_m["total_return_pct"], 2),
            "cagr_%":    round(cs_m["cagr_pct"], 2),
            "max_dd_%":  round(cs_m["max_drawdown_pct"], 2),
            "sharpe":    round(cs_m["sharpe"], 2),
        })

    table = pd.DataFrame(rows)
    print("=" * 92)
    print("FUNDING CARRY RESULTS — realistic costs, varying leverage")
    print("=" * 92)
    print(table.to_string(index=False))

    out_path = DATA_DIR / "funding_results.csv"
    table.to_csv(out_path, index=False)
    print(f"\nResults -> {out_path}")


if __name__ == "__main__":
    main()
