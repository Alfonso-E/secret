"""LightGBM-driven funding rate prediction for dynamic carry-trade sizing.

Pipeline:
  1. For each asset, compute per-period features (lags, smoothed, vol, rank).
  2. Pool all (asset, time) rows; target = next period's funding rate.
  3. Walk-forward: train on a rolling 6-month window, predict the next 1 month.
     Strict no-leakage — features at time t use only funding history <= t.
  4. Output: predicted funding rate per (asset, time) for the test windows.

The downstream backtest allocates capital across assets weighted by predicted
funding (positive predictions only), rebalancing weekly.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MLConfig:
    train_window_periods: int = 6 * 30 * 3   # ~6 months on 8h funding
    test_window_periods:  int = 1 * 30 * 3   # ~1 month rolling test windows
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 200
    min_child_samples: int = 50
    feature_fraction: float = 0.85
    random_state: int = 42


def _feature_columns() -> list[str]:
    return [
        "lag1", "lag3", "lag9", "lag21",
        "sma9", "sma21",
        "std9", "std21",
        "max_9", "min_9",
        "trend_short_long",
        "pct_rank_30d",
        "cross_rank",
        "basket_mean",
        "own_minus_basket",
        "hour", "dow",
    ]


def _build_single_asset_features(funding: pd.Series) -> pd.DataFrame:
    """All features use funding rates strictly EARLIER than the feature timestamp.

    Implementation trick: every feature is computed from `funding.shift(1)`
    onwards, so the value at index t never includes funding[t] itself.
    """
    shifted = funding.shift(1)

    feats = pd.DataFrame(index=funding.index)
    feats["lag1"]  = shifted
    feats["lag3"]  = funding.shift(3)
    feats["lag9"]  = funding.shift(9)
    feats["lag21"] = funding.shift(21)
    feats["sma9"]  = shifted.rolling(9).mean()
    feats["sma21"] = shifted.rolling(21).mean()
    feats["std9"]  = shifted.rolling(9).std()
    feats["std21"] = shifted.rolling(21).std()
    feats["max_9"] = shifted.rolling(9).max()
    feats["min_9"] = shifted.rolling(9).min()
    feats["trend_short_long"] = feats["sma9"] - feats["sma21"]
    feats["pct_rank_30d"] = shifted.rolling(30 * 3).rank(pct=True)
    feats["hour"] = feats.index.hour
    feats["dow"]  = feats.index.dayofweek
    return feats


def build_features(
    funding_by_symbol: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.Series, pd.Index]:
    """Stack (symbol, time) rows. Returns (features X, target y, symbol_index).

    Target y is next-period funding rate (already shifted appropriately).
    """
    rates_panel = pd.concat(
        {sym: df["funding_rate"] for sym, df in funding_by_symbol.items()},
        axis=1,
    ).sort_index()

    # Cross-sectional features per time: each asset's rank and basket mean.
    cross_rank   = rates_panel.shift(1).rank(axis=1, pct=True)
    basket_mean  = rates_panel.shift(1).mean(axis=1)

    frames: list[pd.DataFrame] = []
    targets: list[pd.Series] = []
    syms: list[pd.Series] = []
    for sym in funding_by_symbol:
        if sym not in rates_panel.columns:
            continue
        own = rates_panel[sym]
        feats = _build_single_asset_features(own)
        feats["cross_rank"] = cross_rank[sym]
        feats["basket_mean"] = basket_mean
        feats["own_minus_basket"] = feats["lag1"] - feats["basket_mean"]
        feats["symbol"] = sym
        # Target: next period funding (what we'd be paid if positioned at t for period t+1)
        target = own.shift(-1)
        block = feats.assign(_target=target).dropna()
        frames.append(block)
        targets.append(block["_target"])
        syms.append(pd.Series(block["symbol"].values, index=block.index, name="symbol"))

    full = pd.concat(frames).sort_index()
    X = full[_feature_columns()].copy()
    X["symbol_id"] = full["symbol"].astype("category").cat.codes
    y = full["_target"].astype("float64")
    sym_idx = full["symbol"]
    return X, y, sym_idx


def walk_forward_predict(
    X: pd.DataFrame,
    y: pd.Series,
    symbol_idx: pd.Series,
    config: MLConfig = MLConfig(),
) -> pd.DataFrame:
    """Run rolling-window training and return per-row predictions.

    Returns a DataFrame indexed by time with columns ['symbol', 'y_true', 'y_pred'].
    """
    # Time ordering for walk-forward. We use sorted unique timestamps.
    times = X.index.unique().sort_values()
    n_times = len(times)
    if n_times < config.train_window_periods + config.test_window_periods:
        raise ValueError(
            f"Not enough data: need at least {config.train_window_periods + config.test_window_periods} "
            f"unique periods, got {n_times}"
        )

    out_rows: list[pd.DataFrame] = []

    for fold_start in range(config.train_window_periods, n_times, config.test_window_periods):
        train_start_idx = max(0, fold_start - config.train_window_periods)
        train_t0 = times[train_start_idx]
        train_t1 = times[fold_start - 1]   # inclusive
        test_t0  = times[fold_start]
        test_t1  = times[min(fold_start + config.test_window_periods - 1, n_times - 1)]

        train_mask = (X.index >= train_t0) & (X.index <= train_t1)
        test_mask  = (X.index >= test_t0)  & (X.index <= test_t1)

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        X_tr, y_tr = X.loc[train_mask], y.loc[train_mask]
        X_te, y_te = X.loc[test_mask],  y.loc[test_mask]

        model = lgb.LGBMRegressor(
            num_leaves=config.num_leaves,
            learning_rate=config.learning_rate,
            n_estimators=config.n_estimators,
            min_child_samples=config.min_child_samples,
            feature_fraction=config.feature_fraction,
            random_state=config.random_state,
            verbosity=-1,
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        out_rows.append(pd.DataFrame({
            "symbol": symbol_idx.loc[test_mask].values,
            "y_true": y_te.values,
            "y_pred": y_pred,
        }, index=X_te.index))

    return pd.concat(out_rows).sort_index()


def score_predictions(preds: pd.DataFrame) -> dict[str, float]:
    """Aggregate out-of-sample prediction quality."""
    y_true = preds["y_true"].to_numpy()
    y_pred = preds["y_pred"].to_numpy()
    # Pearson correlation
    corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    # Mean absolute error
    mae = float(np.mean(np.abs(y_true - y_pred)))
    # Sign agreement (did we predict the sign correctly?)
    sign_agree = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
    # Directional Sharpe-like: if we used sign(y_pred) as a signal, what's the mean PnL?
    pnl = np.sign(y_pred) * y_true
    pnl_mean = float(pnl.mean())
    return {
        "n_predictions": len(preds),
        "pearson_corr":  corr,
        "mae":           mae,
        "sign_agree":    sign_agree,
        "signal_pnl_per_period": pnl_mean,
    }
