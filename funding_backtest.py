"""Funding rate carry trade backtester.

Strategy:
  Long $N spot + Short $N perpetual on the same asset. Net price exposure is
  zero; PnL = funding payments received as the short side. Profitable when
  funding rate > 0 on average (longs paying shorts), which is the historical
  norm in crypto.

Backtest model:
  - Per funding interval (typically 8h): equity *= (1 + funding_rate)
    (positive funding_rate means the short side receives, which we are)
  - Entry/exit fees applied as (4 x taker_fee) per round-trip:
    open spot + open perp + close spot + close perp
  - Capital efficiency: we assume the user has 2x the notional in capital
    (one half held in spot, one half as perp margin). Returns are reported
    as a percentage of total deployed capital.

We provide three variants:
  - always_on: enter on day 1, hold until end, single round-trip of fees
  - threshold: enter when smoothed funding > threshold, exit when below.
                Multiple round-trips, more fees but skip negative periods.
  - cross_sectional: at each funding interval, hold the asset with the
                     highest smoothed funding rate; rotate as needed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CarryConfig:
    initial_capital: float = 10_000.0   # total deployed (spot + perp margin)
    taker_fee: float = 0.001            # 0.1% per leg (Bybit spot/perp taker)
    legs_per_roundtrip: int = 4         # open spot + open perp + close spot + close perp
    funding_periods_per_year: int = 3 * 365  # 8h funding ~ 3 per day
    # --- Leverage on the perp short ---
    # Total capital splits as: spot_notional + perp_margin (= perp_notional/leverage)
    # With market-neutral notional N: capital = N + N/L  =>  N = C * L / (L + 1)
    # The funding earned = N * funding_rate; higher L -> larger N -> larger return.
    perp_leverage: float = 1.0          # 1 = no leverage; 3-5 typical for carry, 10+ risky
    # --- Realistic-cost knobs ---
    slippage_per_leg: float = 0.0005    # 0.05% avg slippage on each fill (BTC/ETH; SOL is worse)
    basis_drag_per_roundtrip: float = 0.001  # 0.1% one-off basis drift between spot & perp at exit
    funding_floor: float = -0.003       # cap per-period funding loss; we'd close before this
    funding_ceiling: float = 0.005      # cap per-period funding gain (extreme spikes mean-revert)

    def carry_notional(self, capital: float) -> float:
        """Notional both legs run at, given current capital and leverage."""
        return capital * self.perp_leverage / (self.perp_leverage + 1)


@dataclass
class CarryResult:
    equity: pd.Series             # cumulative capital ($)
    funding_pnl: pd.Series        # per-period cash flow ($)
    entries: int                  # number of (open spot + open perp) events
    config: CarryConfig
    label: str = ""


def _per_leg_cost(config: CarryConfig) -> float:
    """Combined fee + slippage on one fill, as fraction of notional."""
    return config.taker_fee + config.slippage_per_leg


def _apply_open_fees(equity: float, config: CarryConfig) -> float:
    """Pay 2 legs (open spot + open perp) of fees + slippage on the notional."""
    return equity - config.carry_notional(equity) * _per_leg_cost(config) * 2


def _apply_close_fees(equity: float, config: CarryConfig) -> float:
    """Pay 2 legs (close spot + close perp) of fees + slippage + basis drag on close."""
    n = config.carry_notional(equity)
    return equity - n * (_per_leg_cost(config) * 2 + config.basis_drag_per_roundtrip)


def _clipped_funding(rate: float, config: CarryConfig) -> float:
    """Cap per-period funding within plausible bounds — models real-world risk management."""
    if rate < config.funding_floor:
        return config.funding_floor
    if rate > config.funding_ceiling:
        return config.funding_ceiling
    return rate


def backtest_always_on(funding: pd.DataFrame, config: CarryConfig = CarryConfig()) -> CarryResult:
    """Enter on first row, hold until last row, compound clipped funding."""
    rates = funding["funding_rate"].to_numpy()
    n = len(funding)
    equity = np.empty(n, dtype=float)
    pnl = np.empty(n, dtype=float)

    cap = _apply_open_fees(config.initial_capital, config)

    for i in range(n):
        notional = config.carry_notional(cap)
        clipped = _clipped_funding(rates[i], config)
        period_pnl = notional * clipped
        cap += period_pnl
        pnl[i] = period_pnl
        equity[i] = cap

    cap = _apply_close_fees(cap, config)
    equity[-1] = cap

    return CarryResult(
        equity=pd.Series(equity, index=funding.index, name="equity"),
        funding_pnl=pd.Series(pnl, index=funding.index, name="funding_pnl"),
        entries=1, config=config, label="always_on",
    )


def backtest_threshold(
    funding: pd.DataFrame,
    enter_threshold: float = 0.0001,   # enter when smoothed rate >= this (per period)
    exit_threshold:  float = 0.0,      # exit when smoothed rate <= this
    smooth_periods:  int = 9,          # ~3 days of smoothing on 8h interval
    config: CarryConfig = CarryConfig(),
) -> CarryResult:
    """Enter/exit based on smoothed funding rate with hysteresis."""
    rates = funding["funding_rate"].to_numpy()
    smoothed = funding["funding_rate"].rolling(smooth_periods, min_periods=1).mean().to_numpy()
    n = len(funding)
    equity = np.empty(n, dtype=float)
    pnl = np.empty(n, dtype=float)

    cap = config.initial_capital
    in_position = False
    entries = 0

    for i in range(n):
        # Check entry/exit at the start of this period
        if not in_position and smoothed[i] >= enter_threshold:
            cap = _apply_open_fees(cap, config)
            in_position = True
            entries += 1
        elif in_position and smoothed[i] <= exit_threshold:
            cap = _apply_close_fees(cap, config)
            in_position = False

        # Funding accrues if we are in position
        if in_position:
            notional = config.carry_notional(cap)
            clipped = _clipped_funding(rates[i], config)
            period_pnl = notional * clipped
            cap += period_pnl
            pnl[i] = period_pnl
        else:
            pnl[i] = 0.0
        equity[i] = cap

    # Close any open position at the end
    if in_position:
        cap = _apply_close_fees(cap, config)
        equity[-1] = cap

    return CarryResult(
        equity=pd.Series(equity, index=funding.index, name="equity"),
        funding_pnl=pd.Series(pnl, index=funding.index, name="funding_pnl"),
        entries=entries, config=config,
        label=f"threshold[in={enter_threshold:.4f}, out={exit_threshold:.4f}]",
    )


def backtest_cross_sectional(
    funding_by_symbol: dict[str, pd.DataFrame],
    smooth_periods: int = 9,
    enter_threshold: float = 0.0,
    rebalance_every: int = 21,         # ~weekly on 8h funding cadence
    min_switch_advantage: float = 0.0002,  # smoothed rate must beat current by this
    config: CarryConfig = CarryConfig(),
) -> CarryResult:
    """At each rebalance point, hold the asset with the highest smoothed rate.

    Rotation rules:
      - Only consider rotating every `rebalance_every` funding periods.
      - To switch from current asset A to asset B, B's smoothed rate must
        exceed A's by at least `min_switch_advantage` (covers rotation fee).
      - Stay flat if no asset clears `enter_threshold`.
    """
    aligned = pd.concat(
        {sym: df["funding_rate"] for sym, df in funding_by_symbol.items()},
        axis=1,
    ).sort_index().dropna(how="all")
    rates_arr = aligned.fillna(-np.inf).to_numpy()
    smoothed_df = aligned.rolling(smooth_periods, min_periods=1).mean()
    smoothed = smoothed_df.fillna(-np.inf).to_numpy()
    symbols = list(aligned.columns)

    n = len(aligned)
    equity = np.empty(n, dtype=float)
    pnl = np.empty(n, dtype=float)

    cap = config.initial_capital
    current = -1
    entries = 0
    rotation_cost = _per_leg_cost(config) * config.legs_per_roundtrip  # full roundtrip cost on rotation

    for i in range(n):
        # Only consider changing positions at rebalance intervals
        if i % rebalance_every == 0:
            s = smoothed[i]
            current_rate = s[current] if current >= 0 and not np.isinf(s[current]) else -np.inf

            if np.all(s == -np.inf):
                desired = -1
            else:
                best = int(np.argmax(s))
                best_rate = s[best]
                if best_rate < enter_threshold:
                    desired = -1
                elif current == -1:
                    desired = best
                elif best == current:
                    desired = current
                elif best_rate >= current_rate + min_switch_advantage:
                    desired = best
                else:
                    desired = current

            if desired != current:
                if current == -1 and desired != -1:
                    cap = _apply_open_fees(cap, config)
                    entries += 1
                elif current != -1 and desired == -1:
                    cap = _apply_close_fees(cap, config)
                else:
                    cap -= config.carry_notional(cap) * rotation_cost
                    entries += 1
                current = desired

        # Accrue funding for the held symbol every period
        if current >= 0 and not np.isinf(rates_arr[i, current]):
            notional = config.carry_notional(cap)
            clipped = _clipped_funding(rates_arr[i, current], config)
            period_pnl = notional * clipped
            cap += period_pnl
            pnl[i] = period_pnl
        else:
            pnl[i] = 0.0
        equity[i] = cap

    if current >= 0:
        cap = _apply_close_fees(cap, config)
        equity[-1] = cap

    return CarryResult(
        equity=pd.Series(equity, index=aligned.index, name="equity"),
        funding_pnl=pd.Series(pnl, index=aligned.index, name="funding_pnl"),
        entries=entries, config=config,
        label=f"cross_sectional[{'+'.join(symbols)}, every={rebalance_every}]",
    )


def backtest_equal_weight_basket(
    funding_by_symbol: dict[str, pd.DataFrame],
    config: CarryConfig = CarryConfig(),
) -> CarryResult:
    """Split capital equally across symbols, run always-on carry on each, sum equities."""
    sub_config = replace(config, initial_capital=config.initial_capital / len(funding_by_symbol))
    sub_results = {sym: backtest_always_on(df, sub_config) for sym, df in funding_by_symbol.items()}

    eq = pd.concat([r.equity.rename(sym) for sym, r in sub_results.items()], axis=1).ffill().bfill()
    total_equity = eq.sum(axis=1)
    total_pnl = pd.concat([r.funding_pnl.rename(sym) for sym, r in sub_results.items()], axis=1).fillna(0.0).sum(axis=1)

    return CarryResult(
        equity=total_equity.rename("equity"),
        funding_pnl=total_pnl.rename("funding_pnl"),
        entries=len(funding_by_symbol),
        config=config,
        label=f"equal_weight[{'+'.join(funding_by_symbol)}]",
    )


def backtest_ml_weighted(
    funding_by_symbol: dict[str, pd.DataFrame],
    predictions: pd.DataFrame,            # columns: symbol, y_pred (y_true optional)
    rebalance_every: int = 21,            # ~weekly on 8h funding
    min_pred_to_enter: float = 0.0,       # only allocate to assets with predicted rate > this
    max_position_share: float = 0.5,      # cap any single asset at this share of capital
    top_k: int | None = None,             # if set, only allocate to top-K predictions, equal weight
    config: CarryConfig = CarryConfig(),
) -> CarryResult:
    """Allocate capital across assets weighted by ML-predicted funding rate.

    At each rebalance point:
      1. For each asset, look up predicted next-period funding from `predictions`.
      2. Build weights = max(0, prediction) per asset, normalize to sum to 1.
      3. Cap any single weight at `max_position_share`, renormalize.
      4. Compute fee cost vs current allocation; rebalance if cheaper than holding.

    Between rebalance points, funding accrues at each asset's actual rate per its weight.
    """
    aligned = pd.concat(
        {sym: df["funding_rate"] for sym, df in funding_by_symbol.items()},
        axis=1,
    ).sort_index().dropna(how="all")
    rates_arr = aligned.fillna(-np.inf).to_numpy()
    symbols = list(aligned.columns)
    sym_to_col = {s: i for i, s in enumerate(symbols)}

    # Build predicted-rate panel aligned to the funding index.
    pred_panel = (
        predictions.reset_index()
        .pivot_table(index="funding_time", columns="symbol", values="y_pred", aggfunc="first")
        .reindex(aligned.index)
    )
    pred_panel = pred_panel[[s for s in symbols if s in pred_panel.columns]]
    pred_panel = pred_panel.reindex(columns=symbols)
    # If predictions missing for some symbol at some bar, treat as -inf (skip)
    pred_arr = pred_panel.to_numpy().astype(float)
    pred_arr = np.where(np.isnan(pred_arr), -np.inf, pred_arr)

    n = len(aligned)
    n_sym = len(symbols)
    equity = np.empty(n, dtype=float)
    pnl = np.empty(n, dtype=float)

    cap = config.initial_capital
    weights = np.zeros(n_sym, dtype=float)  # share of total notional in each asset (sums to <=1)
    in_market = False
    entries = 0
    rotation_cost = _per_leg_cost(config) * config.legs_per_roundtrip

    for i in range(n):
        if i % rebalance_every == 0 and not np.all(np.isinf(pred_arr[i])):
            preds = pred_arr[i].copy()
            preds[~np.isfinite(preds)] = -np.inf
            eligible = preds > min_pred_to_enter

            if not eligible.any():
                desired = np.zeros(n_sym)
            elif top_k is not None:
                # Pick the top K eligible by predicted rate, equal-weight them.
                k = min(top_k, int(eligible.sum()))
                top_idx = np.argsort(preds)[-k:]
                desired = np.zeros(n_sym)
                desired[top_idx] = 1.0 / k
            else:
                # Weight proportional to predicted rate (positive predictions only).
                w = np.where(eligible, preds, 0.0)
                desired = w / w.sum()
                if max_position_share < 1.0:
                    desired = np.minimum(desired, max_position_share)
                    if desired.sum() > 0:
                        desired = desired / desired.sum()

            notional = config.carry_notional(cap)
            weight_change = np.abs(desired - weights).sum()
            cap -= notional * weight_change * (rotation_cost / 2.0)
            if not in_market and desired.sum() > 0:
                entries += 1
                in_market = True
            elif in_market and desired.sum() == 0:
                in_market = False
            weights = desired

        # Accrue funding for the held weighted basket
        if weights.sum() > 0:
            notional = config.carry_notional(cap)
            period_pnl = 0.0
            for k in range(n_sym):
                if weights[k] > 0 and not np.isinf(rates_arr[i, k]):
                    clipped = _clipped_funding(rates_arr[i, k], config)
                    period_pnl += notional * weights[k] * clipped
            cap += period_pnl
            pnl[i] = period_pnl
        else:
            pnl[i] = 0.0
        equity[i] = cap

    if in_market:
        cap = _apply_close_fees(cap, config)
        equity[-1] = cap

    return CarryResult(
        equity=pd.Series(equity, index=aligned.index, name="equity"),
        funding_pnl=pd.Series(pnl, index=aligned.index, name="funding_pnl"),
        entries=entries, config=config,
        label=f"ml_weighted[{'+'.join(symbols)}, rb={rebalance_every}]",
    )


def compute_carry_metrics(result: CarryResult) -> dict[str, float]:
    eq = result.equity
    cap0 = result.config.initial_capital

    total_return = eq.iloc[-1] / cap0 - 1.0
    n_periods = len(eq)
    periods_per_year = result.config.funding_periods_per_year
    cagr = (eq.iloc[-1] / cap0) ** (periods_per_year / max(n_periods, 1)) - 1.0
    rolling_max = eq.cummax()
    drawdown = eq / rolling_max - 1.0
    max_dd = drawdown.min()

    period_returns = eq.pct_change().dropna()
    sharpe = (period_returns.mean() / period_returns.std()) * np.sqrt(periods_per_year) if period_returns.std() > 0 else 0.0

    positive_periods = (result.funding_pnl > 0).sum()
    negative_periods = (result.funding_pnl < 0).sum()
    zero_periods = (result.funding_pnl == 0).sum()

    return {
        "total_return_pct":   total_return * 100,
        "cagr_pct":           cagr * 100,
        "max_drawdown_pct":   max_dd * 100,
        "sharpe":             sharpe,
        "final_equity":       eq.iloc[-1],
        "n_periods":          n_periods,
        "entries":            result.entries,
        "positive_periods":   int(positive_periods),
        "negative_periods":   int(negative_periods),
        "zero_periods":       int(zero_periods),
    }
