"""Technical indicators for the trading bot.

Each function takes pandas Series/DataFrame columns and returns a pandas
Series aligned to the input index. Formulas match TradingView / Wilder's
conventions so values are comparable to charting platforms.
"""

from __future__ import annotations

import pandas as pd


def ema(close: pd.Series, length: int) -> pd.Series:
    """Exponential moving average (TradingView-style, adjust=False)."""
    return close.ewm(span=length, adjust=False).mean()


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index, 0-100 scale."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14
) -> pd.Series:
    """Wilder's Average True Range."""
    return true_range(high, low, close).ewm(alpha=1 / length, adjust=False).mean()


def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14
) -> pd.Series:
    """Wilder's Average Directional Index. Direction-agnostic trend strength, 0-100.

    >25 typically signals a trending market; <20 signals ranging.
    """
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        ((up_move > down_move) & (up_move > 0)) * up_move, index=high.index
    ).clip(lower=0.0)
    minus_dm = pd.Series(
        ((down_move > up_move) & (down_move > 0)) * down_move, index=low.index
    ).clip(lower=0.0)

    tr = true_range(high, low, close)
    atr_n = tr.ewm(alpha=1 / length, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / length, adjust=False).mean() / atr_n
    minus_di = 100 * minus_dm.ewm(alpha=1 / length, adjust=False).mean() / atr_n
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / length, adjust=False).mean()
