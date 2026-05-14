"""Hidden Markov Model regime detection for the EMA directional overlay.

The idea: instead of always-on EMA entries, gate them on a market regime
inferred from observable features (returns, volatility, volume). If the
HMM thinks we're in a bear/crash state, skip the EMA entry — preserve
capital for when the regime turns favorable.

We use a Gaussian HMM with N states. After training, we reorder the states
by their mean log-return so:
  state 0 = lowest mean return  (crash/bear)
  state N-1 = highest mean return (bull/euphoria)

For backtests we use rolling-window prediction (each state inference uses
only past data) to avoid any look-ahead bias. Look-ahead is the #1 trap
in regime-detection backtests and the reason most published HMM "edges"
disappear out-of-sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


FEATURE_COLUMNS = ["log_return", "vol_z", "volume_z"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract HMM features from OHLCV.

    log_return : per-bar log return (close-to-close)
    vol_z      : z-score of 24-bar rolling std of returns
    volume_z   : z-score of log volume
    """
    feat = pd.DataFrame(index=df.index)
    log_ret = np.log(df["close"] / df["close"].shift(1))
    vol_24 = log_ret.rolling(24).std()
    log_vol = np.log(df["volume"].replace(0, np.nan)).ffill()

    feat["log_return"] = log_ret
    feat["vol_z"] = (vol_24 - vol_24.rolling(7 * 24).mean()) / vol_24.rolling(7 * 24).std()
    feat["volume_z"] = (log_vol - log_vol.rolling(7 * 24).mean()) / log_vol.rolling(7 * 24).std()
    return feat.dropna()


@dataclass(frozen=True)
class HMMModel:
    model:       GaussianHMM
    state_order: np.ndarray   # raw state idx -> ordered idx (0=lowest mean return)
    n_states:    int

    def remap(self, raw_states: np.ndarray) -> np.ndarray:
        return self.state_order[raw_states]


def train_hmm(features: pd.DataFrame, n_states: int = 3, n_iter: int = 200,
              random_state: int = 42) -> HMMModel:
    """Fit a Gaussian HMM on the given feature dataframe."""
    X = features[FEATURE_COLUMNS].to_numpy()
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
        tol=1e-3,
        verbose=False,            # silence per-iter prints
    )
    # hmmlearn's ConvergenceMonitor prints to stdout/stderr when log-likelihood
    # oscillates near the optimum. The delta is tiny (~1e-2) and doesn't affect
    # state assignments — silence both streams to keep cron-run logs clean.
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X)
    # Order states by mean log-return so 0 = bearest, n-1 = bullest
    raw_state_means = model.means_[:, 0]
    sort_idx = np.argsort(raw_state_means)
    state_order = np.empty(n_states, dtype=int)
    for new_idx, raw_idx in enumerate(sort_idx):
        state_order[raw_idx] = new_idx
    return HMMModel(model=model, state_order=state_order, n_states=n_states)


def predict_rolling(hmm: HMMModel, features: pd.DataFrame,
                    window: int = 7 * 24) -> pd.Series:
    """Predict ordered state per row using ONLY past data — no look-ahead.

    At bar i, we run Viterbi on features[max(0, i-window+1) : i+1] and take
    the last state. Cost: O(n * window). Window of one week is plenty for
    HMM smoothing without contaminating future-looking inference.
    """
    X = features[FEATURE_COLUMNS].to_numpy()
    n = len(X)
    out = np.full(n, -1, dtype=int)
    for i in range(window, n):
        sub = X[i - window + 1 : i + 1]
        try:
            path = hmm.model.predict(sub)
            out[i] = path[-1]
        except Exception:
            out[i] = -1
    ordered = np.where(out >= 0, hmm.remap(np.where(out >= 0, out, 0)), -1)
    return pd.Series(ordered, index=features.index, name="hmm_state")


def predict_current_state(hmm: HMMModel, features: pd.DataFrame,
                          window: int = 7 * 24) -> int:
    """Live-bot fast path: predict the state for the MOST RECENT bar only.

    Uses only data up to (and including) the last row in `features`, with a
    sliding window of `window` bars. Returns the ordered state index, or -1
    if there isn't enough history to predict.
    """
    X = features[FEATURE_COLUMNS].to_numpy()
    if len(X) < window:
        return -1
    sub = X[-window:]
    try:
        raw_path = hmm.model.predict(sub)
        return int(hmm.state_order[raw_path[-1]])
    except Exception:
        return -1


def state_summary(hmm: HMMModel, features: pd.DataFrame,
                  states: pd.Series) -> pd.DataFrame:
    """Per-state stats (count, mean return, vol, time fraction).

    Useful to verify the trained states match the bull/bear story
    we tell ourselves before deploying the filter.
    """
    df = features.join(states, how="inner")
    df = df[df["hmm_state"] >= 0]
    grouped = df.groupby("hmm_state")
    rows = []
    for s in sorted(df["hmm_state"].unique()):
        sub = df[df["hmm_state"] == s]
        rows.append({
            "state":         int(s),
            "bars":          len(sub),
            "frac":          len(sub) / len(df),
            "mean_log_ret":  float(sub["log_return"].mean()),
            "mean_vol_z":    float(sub["vol_z"].mean()),
            "mean_volume_z": float(sub["volume_z"].mean()),
            "annualized_return_pct": float(sub["log_return"].mean() * 365 * 24 * 100),
        })
    return pd.DataFrame(rows)
