"""Test whether HMM regime detection improves the Phase 1 EMA leg.

Plan:
  1. Train a Gaussian HMM on BTC features from a TRAIN window (Jan 2023 - Dec 2023).
  2. Apply it out-of-sample via rolling-window prediction (no look-ahead)
     to two TEST windows:
        - Bull-test:   Jan 2024 - Dec 2024  (mostly bullish; baseline EMA shone)
        - Recent:      2024-12 - 2026-05    (mostly chop/bear; baseline EMA bled)
  3. Compare 4 EMA backtests on each test window:
        A. EMA(50/200) alone — no regime filter
        B. EMA(50/200) + EMA(200) price filter (the v1 filter we shipped)
        C. EMA + 3-state HMM filter (only bull, neutral states allowed)
        D. EMA + 5-state HMM filter (only neutral, bull, euphoria allowed)
  4. Report return, max drawdown, Sharpe, trade count per variant.

If HMM lifts both Sharpe and reduces DD vs baseline filter, we integrate it.
If not, we publish the negative result and stop spending engineering on it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtest import BacktestConfig, compute_metrics, run_backtest
from bybit_klines import load_or_fetch_range
from regime_hmm import (
    build_features, predict_rolling, state_summary, train_hmm,
)
from strategy import StrategyParams, generate_signals

DATA_DIR = Path(__file__).parent / "data"


def _ema_metrics(df, params: StrategyParams, hmm_states: pd.Series | None,
                 allowed_states: set[int] | None, label: str) -> dict:
    signals = generate_signals(df, params)
    if hmm_states is not None and allowed_states is not None:
        signals = signals.join(hmm_states, how="left")
        in_regime = signals["hmm_state"].isin(allowed_states).fillna(False)
        signals["long_entry"] = signals["long_entry"] & in_regime
    result = run_backtest(signals, params, BacktestConfig())
    m = compute_metrics(result)
    return {
        "variant":   label,
        "return_%":  round(m["total_return_pct"], 2),
        "cagr_%":    round(m["cagr_pct"], 2),
        "max_dd_%":  round(m["max_drawdown_pct"], 2),
        "sharpe":    round(m["sharpe"], 2),
        "trades":    int(m["num_trades"]),
        "win_rate_%":round(m["win_rate_pct"], 2),
    }


def main() -> None:
    train = load_or_fetch_range(
        DATA_DIR / "btcusdt_1h_spot_2023.csv",
        symbol="BTCUSDT", interval="60", category="spot",
        start="2023-01-01", end="2023-12-31",
    )
    bull_test = load_or_fetch_range(
        DATA_DIR / "btcusdt_1h_spot_2024.csv",
        symbol="BTCUSDT", interval="60", category="spot",
        start="2024-01-01", end="2024-12-31",
    )
    recent = load_or_fetch_range(
        DATA_DIR / "btcusdt_1h_spot_recent.csv",
        symbol="BTCUSDT", interval="60", category="spot",
        start="2024-12-01", end="2026-05-12",
    )
    print(f"Train: {len(train)} bars  {train.index[0]} -> {train.index[-1]}")
    print(f"Bull test:   {len(bull_test)} bars  {bull_test.index[0]} -> {bull_test.index[-1]}")
    print(f"Recent test: {len(recent)} bars  {recent.index[0]} -> {recent.index[-1]}")

    train_features = build_features(train)
    print(f"\nTraining HMMs on {len(train_features)} feature rows...")

    hmm3 = train_hmm(train_features, n_states=3)
    hmm5 = train_hmm(train_features, n_states=5)

    print("\n--- 3-state HMM on TRAIN data ---")
    train_states3 = predict_rolling(hmm3, train_features, window=7 * 24)
    print(state_summary(hmm3, train_features, train_states3).to_string(index=False))

    print("\n--- 5-state HMM on TRAIN data ---")
    train_states5 = predict_rolling(hmm5, train_features, window=7 * 24)
    print(state_summary(hmm5, train_features, train_states5).to_string(index=False))

    # Apply OOS to each test window
    print("\nGenerating OOS state predictions for test windows (rolling, no look-ahead)...")
    bull_feats = build_features(bull_test)
    recent_feats = build_features(recent)
    bull_s3 = predict_rolling(hmm3, bull_feats, window=7 * 24)
    bull_s5 = predict_rolling(hmm5, bull_feats, window=7 * 24)
    rec_s3 = predict_rolling(hmm3, recent_feats, window=7 * 24)
    rec_s5 = predict_rolling(hmm5, recent_feats, window=7 * 24)

    print("\n--- 3-state HMM on RECENT (OOS) ---")
    print(state_summary(hmm3, recent_feats, rec_s3).to_string(index=False))
    print("\n--- 5-state HMM on RECENT (OOS) ---")
    print(state_summary(hmm5, recent_feats, rec_s5).to_string(index=False))

    # Define which states are "allowed" for EMA entries
    # 3-state: only state 2 (highest mean return)
    # 5-state: states 2, 3, 4 (mid + bull + euphoria)
    allowed3 = {2}
    allowed5 = {2, 3, 4}

    params_no_filter = StrategyParams(ema_fast=50, ema_slow=200, trend_ema=None)
    params_ema_filter = StrategyParams(ema_fast=50, ema_slow=200, trend_ema=200)

    print("\n" + "=" * 92)
    print("BULL TEST  Jan 2024 - Dec 2024  (out-of-sample for HMM)")
    print("=" * 92)
    rows = [
        _ema_metrics(bull_test, params_no_filter, None, None, "A: no filter"),
        _ema_metrics(bull_test, params_ema_filter, None, None, "B: EMA(200) price filter"),
        _ema_metrics(bull_test, params_ema_filter, bull_s3, allowed3, "C: + HMM-3 (top state only)"),
        _ema_metrics(bull_test, params_ema_filter, bull_s5, allowed5, "D: + HMM-5 (top 3 states)"),
    ]
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 92)
    print("RECENT TEST  Dec 2024 - May 2026  (out-of-sample for HMM)")
    print("=" * 92)
    rows = [
        _ema_metrics(recent, params_no_filter, None, None, "A: no filter"),
        _ema_metrics(recent, params_ema_filter, None, None, "B: EMA(200) price filter"),
        _ema_metrics(recent, params_ema_filter, rec_s3, allowed3, "C: + HMM-3 (top state only)"),
        _ema_metrics(recent, params_ema_filter, rec_s5, allowed5, "D: + HMM-5 (top 3 states)"),
    ]
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
