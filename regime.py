"""Regime detection: classify each bar as TREND or RANGE based on ADX.

The detector uses a two-threshold hysteresis to avoid flickering near
the boundary:
  ADX >= trend_in   -> switch to TREND   (cross up)
  ADX <= range_in   -> switch to RANGE   (cross down)
  otherwise         -> keep previous regime
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from indicators import adx as adx_indicator

TREND = "TREND"
RANGE = "RANGE"


@dataclass(frozen=True)
class RegimeParams:
    adx_length: int = 14
    trend_in:  float = 25.0   # switch into TREND when ADX crosses up through this
    range_in:  float = 20.0   # switch into RANGE when ADX crosses down through this
    initial:   str   = RANGE  # state before enough data accumulated


def detect_regime(df: pd.DataFrame, params: RegimeParams = RegimeParams()) -> pd.DataFrame:
    """Return df with 'adx' and 'regime' columns added."""
    out = df.copy()
    out["adx"] = adx_indicator(df["high"], df["low"], df["close"], params.adx_length)

    adx_vals = out["adx"].to_numpy()
    n = len(out)
    regime = np.empty(n, dtype=object)
    state = params.initial

    for i in range(n):
        a = adx_vals[i]
        if np.isnan(a):
            regime[i] = state
            continue
        if a >= params.trend_in:
            state = TREND
        elif a <= params.range_in:
            state = RANGE
        regime[i] = state

    out["regime"] = regime
    return out
