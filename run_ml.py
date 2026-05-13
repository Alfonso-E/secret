"""ML-driven funding carry: train a per-period funding-rate predictor and
size a multi-asset carry basket from the predictions.

Reuses the same 10-asset universe and the same realistic-costs backtester.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bybit_funding import load_or_fetch_funding
from funding_backtest import (
    CarryConfig, backtest_cross_sectional, backtest_equal_weight_basket,
    backtest_ml_weighted, compute_carry_metrics,
)
from funding_ml import (
    MLConfig, build_features, score_predictions, walk_forward_predict,
)

DATA_DIR = Path(__file__).parent / "data"
ALL_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "AVAXUSDT", "LINKUSDT", "ARBUSDT", "DOGEUSDT",
    "ADAUSDT", "INJUSDT",
]
LEVERAGE = 5.0


def main() -> None:
    funding_by_symbol: dict[str, pd.DataFrame] = {}
    for sym in ALL_SYMBOLS:
        path = DATA_DIR / f"funding_{sym.lower()}.csv"
        df = load_or_fetch_funding(path, symbol=sym, category="linear", start="2023-01-01")
        if df.empty:
            continue
        funding_by_symbol[sym] = df

    print(f"Loaded {len(funding_by_symbol)} symbols.")
    print("Building features...")
    X, y, sym_idx = build_features(funding_by_symbol)
    print(f"  Features matrix: {X.shape}, target: {y.shape}")
    print(f"  Time range: {X.index.min()} -> {X.index.max()}")

    ml_cfg = MLConfig()
    print(f"Walk-forward training "
          f"(train={ml_cfg.train_window_periods}p ~6mo, "
          f"test={ml_cfg.test_window_periods}p ~1mo)...")
    preds = walk_forward_predict(X, y, sym_idx, ml_cfg)
    print(f"  Out-of-sample predictions: {len(preds)} rows")

    scores = score_predictions(preds)
    print(f"\nPrediction quality (out-of-sample):")
    for k, v in scores.items():
        print(f"  {k:30s}  {v:.6f}" if isinstance(v, float) else f"  {k:30s}  {v}")

    # Restrict the backtest window to test periods only (where we have predictions).
    test_start = preds.index.min()
    test_end = preds.index.max()
    print(f"\nBacktest window (OOS only): {test_start} -> {test_end}")

    funding_oos = {sym: df.loc[test_start:test_end] for sym, df in funding_by_symbol.items()}

    cfg = CarryConfig(perp_leverage=LEVERAGE)
    rows: list[dict[str, object]] = []

    # Baseline: equal-weight basket on OOS window only
    eb = backtest_equal_weight_basket(funding_oos, cfg)
    eb_m = compute_carry_metrics(eb)
    rows.append({"strategy": f"basket[{len(funding_oos)}] always-on",
                 "return_%": round(eb_m["total_return_pct"], 2),
                 "cagr_%":   round(eb_m["cagr_pct"], 2),
                 "max_dd_%": round(eb_m["max_drawdown_pct"], 2),
                 "sharpe":   round(eb_m["sharpe"], 2)})

    # Filtered (positive-funding) basket on OOS
    pos = {s: d for s, d in funding_oos.items() if d["funding_rate"].mean() > 0}
    if len(pos) < len(funding_oos):
        fb = backtest_equal_weight_basket(pos, cfg)
        fb_m = compute_carry_metrics(fb)
        rows.append({"strategy": f"basket[{len(pos)} positive only] always-on",
                     "return_%": round(fb_m["total_return_pct"], 2),
                     "cagr_%":   round(fb_m["cagr_pct"], 2),
                     "max_dd_%": round(fb_m["max_drawdown_pct"], 2),
                     "sharpe":   round(fb_m["sharpe"], 2)})

    # Naive cross-sectional (uses smoothed historical rate, not ML)
    cs = backtest_cross_sectional(funding_oos, smooth_periods=9, enter_threshold=0.0, config=cfg)
    cs_m = compute_carry_metrics(cs)
    rows.append({"strategy": "cross-sectional[10] weekly (naive)",
                 "return_%": round(cs_m["total_return_pct"], 2),
                 "cagr_%":   round(cs_m["cagr_pct"], 2),
                 "max_dd_%": round(cs_m["max_drawdown_pct"], 2),
                 "sharpe":   round(cs_m["sharpe"], 2)})

    # ML variants
    ml_variants = [
        ("ML magnitude-weighted, weekly, cap40%",
         dict(rebalance_every=21, top_k=None, max_position_share=0.4, min_pred_to_enter=0.0)),
        ("ML top-1 (winner-take-all), weekly",
         dict(rebalance_every=21, top_k=1, max_position_share=1.0, min_pred_to_enter=0.0)),
        ("ML top-3 equal-weight, weekly",
         dict(rebalance_every=21, top_k=3, max_position_share=1.0, min_pred_to_enter=0.0)),
        ("ML top-3 equal-weight, monthly",
         dict(rebalance_every=90, top_k=3, max_position_share=1.0, min_pred_to_enter=0.0)),
        ("ML top-3, monthly, threshold=5e-5",
         dict(rebalance_every=90, top_k=3, max_position_share=1.0, min_pred_to_enter=0.00005)),
        ("ML top-1, monthly, threshold=5e-5",
         dict(rebalance_every=90, top_k=1, max_position_share=1.0, min_pred_to_enter=0.00005)),
    ]
    for label, kw in ml_variants:
        r = backtest_ml_weighted(funding_oos, preds, config=cfg, **kw)
        m = compute_carry_metrics(r)
        rows.append({"strategy": label,
                     "return_%": round(m["total_return_pct"], 2),
                     "cagr_%":   round(m["cagr_pct"], 2),
                     "max_dd_%": round(m["max_drawdown_pct"], 2),
                     "sharpe":   round(m["sharpe"], 2)})

    table = pd.DataFrame(rows)
    print()
    print("=" * 92)
    print(f"ML-DRIVEN CARRY  leverage={LEVERAGE:.0f}x  OOS window only")
    print("=" * 92)
    print(table.to_string(index=False))

    out_path = DATA_DIR / "ml_results.csv"
    table.to_csv(out_path, index=False)
    print(f"\nResults -> {out_path}")


if __name__ == "__main__":
    main()
