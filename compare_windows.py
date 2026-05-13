"""Run the same parameter sweep on two market windows for comparison.

  - bull  : 2023-01-01 -> 2024-12-31  (BTC trended hard up, ~$16k -> ~$94k)
  - recent: ~Nov 2024 -> May 2026     (already cached; mostly flat/down)

Goal: see if the strategy actually works in a bullish trending regime, which
would confirm the failure on the recent window is regime-driven rather than
fundamental.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bybit_klines import load_or_fetch, load_or_fetch_range
from sweep import run_sweep

DATA_DIR = Path(__file__).parent / "data"


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

    for label, df in [("BULL window (Jan 2023 - Dec 2024)", bull),
                      ("RECENT window (cached)", recent)]:
        bh = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
        print()
        print("=" * 78)
        print(f"{label}")
        print(f"  {len(df)} bars  {df.index[0]} -> {df.index[-1]}")
        print(f"  Buy & hold over this window: {bh:+.2f}%")
        print("=" * 78)
        table = run_sweep(df)
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
