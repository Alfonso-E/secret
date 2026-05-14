"""Test whether expanding the carry universe helps the cross-sectional strategy.

Current live: 7 assets (ETH SOL AVAX LINK ARB DOGE ADA).
Candidate expansion: add mid-cap perps known to carry higher funding rates
due to less arbitrage liquidity — per BIS WP 1087.

Specifically we add:
  ATOM, NEAR, DOT, LTC, OPUSDT, FILUSDT, BCHUSDT, MATICUSDT  (8 new)

Constraints to keep this honest:
  - Same window (Jan 2023 - May 2026) as previous comparisons.
  - Same 5x leverage, same min_switch_advantage hysteresis.
  - Same rebalance cadence.
  - Realistic costs already baked in (slippage, basis drag, funding floor).

If the expanded universe lifts CAGR by >0.5pp without materially worsening
DD, integrate. Otherwise stop.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bybit_funding import load_or_fetch_funding
from funding_backtest import (
    CarryConfig, backtest_cross_sectional, backtest_equal_weight_basket,
    compute_carry_metrics,
)

DATA_DIR = Path(__file__).parent / "data"
LEVERAGE = 5.0
CARRY_CAPITAL = 8_000.0   # the 80% of $10k carry-portion budget

# Current live (7-asset universe; BTC excluded for EMA)
CURRENT = ["ETHUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT", "ARBUSDT", "DOGEUSDT", "ADAUSDT"]

# Candidate additions — mid-cap perps with >2 years of history on Bybit
NEW = ["ATOMUSDT", "NEARUSDT", "DOTUSDT", "LTCUSDT", "OPUSDT", "FILUSDT", "BCHUSDT", "MATICUSDT"]


def _load(symbol: str) -> pd.DataFrame | None:
    path = DATA_DIR / f"funding_{symbol.lower()}.csv"
    try:
        df = load_or_fetch_funding(path, symbol=symbol, category="linear", start="2023-01-01")
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"  [SKIP] {symbol}: {type(e).__name__}: {e}")
        return None


def _run(label: str, universe: dict[str, pd.DataFrame], cfg: CarryConfig) -> dict:
    cs = backtest_cross_sectional(
        universe, smooth_periods=9, enter_threshold=0.0,
        rebalance_every=21, min_switch_advantage=0.0002,
        config=cfg,
    )
    m = compute_carry_metrics(cs)

    eb = backtest_equal_weight_basket(universe, cfg)
    em = compute_carry_metrics(eb)

    return {
        "universe":             label,
        "n_assets":             len(universe),
        "xs_return_%":          round(m["total_return_pct"], 2),
        "xs_cagr_%":            round(m["cagr_pct"], 2),
        "xs_max_dd_%":          round(m["max_drawdown_pct"], 2),
        "xs_sharpe":            round(m["sharpe"], 2),
        "basket_cagr_%":        round(em["cagr_pct"], 2),
        "basket_max_dd_%":      round(em["max_drawdown_pct"], 2),
    }


def main() -> None:
    cfg = CarryConfig(initial_capital=CARRY_CAPITAL, perp_leverage=LEVERAGE)

    print("Loading funding history for all symbols...")
    all_symbols = CURRENT + NEW
    funding: dict[str, pd.DataFrame] = {}
    for sym in all_symbols:
        df = _load(sym)
        if df is not None:
            funding[sym] = df

    print(f"\nLoaded {len(funding)} symbols.")

    print("\nFunding rate stats (annualized raw APR):")
    apr_rows = []
    for sym, df in funding.items():
        mean = df["funding_rate"].mean()
        apr_rows.append({
            "symbol":   sym,
            "bars":     len(df),
            "first":    df.index[0].date(),
            "last":     df.index[-1].date(),
            "apr_%":    round(mean * 3 * 365 * 100, 2),
            "pos_frac": round((df["funding_rate"] > 0).mean(), 3),
        })
    print(pd.DataFrame(apr_rows).sort_values("apr_%", ascending=False).to_string(index=False))

    # Run several universes
    rows = []
    current_only = {s: funding[s] for s in CURRENT if s in funding}
    rows.append(_run(f"current (7 assets)", current_only, cfg))

    new_only = {s: funding[s] for s in NEW if s in funding}
    rows.append(_run(f"new mid-cap only ({len(new_only)})", new_only, cfg))

    combined = {s: funding[s] for s in all_symbols if s in funding}
    rows.append(_run(f"combined ({len(combined)} assets)", combined, cfg))

    # Filtered: combined MINUS negative/marginal funding rates AND truncated data.
    # BCH had -0.43% APR, ATOM 1.81%, MATIC has data only through Sept 2024.
    EXCLUDE = {"BCHUSDT", "ATOMUSDT", "MATICUSDT"}
    filtered = {s: funding[s] for s in all_symbols if s in funding and s not in EXCLUDE}
    rows.append(_run(f"filtered ({len(filtered)} assets)", filtered, cfg))

    print()
    print("=" * 92)
    print("CROSS-SECTIONAL CARRY RESULTS  (Jan 2023 - now, 5x leverage, realistic costs)")
    print("=" * 92)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
