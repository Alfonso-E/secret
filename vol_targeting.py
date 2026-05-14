"""Volatility-targeted position sizing.

For each bar, compute a position-size multiplier based on the realized
volatility of the underlying. Goal: keep the portfolio's expected dollar
volatility roughly constant rather than its notional.

  multiplier(t) = clip(target_annualized_vol / realized_vol(t), min_mult, max_mult)

When the market is calm (realized_vol < target), we scale UP (multiplier > 1).
When the market is volatile (realized_vol > target), we scale DOWN.

We use BTC's price volatility as a systemic crypto-market vol proxy.
Crypto assets are highly correlated; when BTC is volatile, all carry
positions face higher basis risk and liquidation risk. Using BTC as the
single signal keeps the implementation simple and the multiplier
consistent across all assets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HOURS_PER_YEAR = 365 * 24


def realized_vol_annualized(close: pd.Series, lookback_hours: int = 14 * 24) -> pd.Series:
    """Rolling realized volatility from 1h closes, annualized."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(lookback_hours).std() * np.sqrt(HOURS_PER_YEAR)


def compute_vol_multipliers(
    btc_klines: pd.DataFrame,
    target_annualized_vol: float = 0.50,
    lookback_hours: int = 14 * 24,
    min_mult: float = 0.30,
    max_mult: float = 1.50,
) -> pd.Series:
    """Per-bar position multipliers, indexed by the kline timestamp.

    Returns a pd.Series at 1h frequency. Callers that operate at different
    cadences (e.g., 8h funding events) should use `.asof(t)` to look up the
    most recent multiplier.
    """
    vol = realized_vol_annualized(btc_klines["close"], lookback_hours)
    multiplier = (target_annualized_vol / vol).clip(min_mult, max_mult)
    return multiplier.rename("vol_multiplier")


def mult_at(mult_series: pd.Series | None, t: pd.Timestamp,
            default: float = 1.0) -> float:
    """Safe lookup — returns `default` if the series is None or value is NaN."""
    if mult_series is None or mult_series.empty:
        return default
    try:
        v = mult_series.asof(t)
        if pd.isna(v):
            return default
        return float(v)
    except (KeyError, ValueError):
        return default
