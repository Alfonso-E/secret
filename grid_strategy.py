"""Spot grid bot — cycling buy-low / sell-high across fixed price levels.

Activation:
  When the strategy is started, we anchor the grid to the current price.
  N channels are placed BELOW the center, each with a paired sell level
  one grid-step above its buy price. Each channel is independent and
  cycles continuously: buy fills -> holding -> sell fills -> reset to
  waiting for next buy.

Spot-only (long-only):
  We never short. A channel only holds positive units between a buy and
  its matching sell. Total cash committed across all channels equals the
  grid's allocated capital, partitioned evenly.

Cash accounting (per cycle):
  buy:  cash -= (alloc + alloc * fee)            units = alloc / buy_price
  sell: cash += units * sell_price * (1 - fee)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd


@dataclass(frozen=True)
class GridParams:
    n_levels: int = 10          # buy levels below the anchor (each paired with a sell above)
    spacing_atr_mult: float = 0.5  # grid step = atr_at_anchor * this
    capital_fraction: float = 1.0  # fraction of available cash to commit to the grid


ChannelStatus = Literal["waiting", "holding"]


@dataclass
class GridChannel:
    buy_price:  float
    sell_price: float
    alloc:      float                  # cash committed to this channel when buying
    status:     ChannelStatus = "waiting"
    units:      float = 0.0
    last_buy_time: pd.Timestamp | None = None


@dataclass
class GridState:
    anchor_price: float
    anchor_time:  pd.Timestamp
    spacing:      float
    channels:     list[GridChannel]
    realized_pnl: float = 0.0
    cycles:       int   = 0            # full buy+sell completions across all channels
    cash_committed: float = 0.0        # total cash currently parked in channels


def build_grid(
    anchor_price: float,
    anchor_time: pd.Timestamp,
    atr_at_anchor: float,
    available_cash: float,
    params: GridParams,
) -> GridState:
    """Set up N channels below `anchor_price`, evenly spaced by ATR-scaled steps."""
    spacing = max(atr_at_anchor * params.spacing_atr_mult, 1e-9)
    total_alloc = available_cash * params.capital_fraction
    per_channel = total_alloc / params.n_levels

    channels: list[GridChannel] = []
    for i in range(1, params.n_levels + 1):
        buy_price  = anchor_price - i * spacing
        sell_price = anchor_price - (i - 1) * spacing
        channels.append(GridChannel(buy_price=buy_price, sell_price=sell_price, alloc=per_channel))

    return GridState(
        anchor_price=anchor_price,
        anchor_time=anchor_time,
        spacing=spacing,
        channels=channels,
    )


@dataclass
class GridFill:
    time:   pd.Timestamp
    price:  float
    side:   Literal["buy", "sell"]
    units:  float
    cash_delta: float        # change to cash account, fees included
    pnl:    float = 0.0      # only set for sell fills (per-channel realized)


def step_grid(
    state: GridState,
    bar_time:  pd.Timestamp,
    bar_high:  float,
    bar_low:   float,
    fee_rate:  float,
) -> tuple[list[GridFill], float]:
    """Walk the bar's high/low through the grid; fire fills where price crosses levels.

    Returns (fills, cash_delta). Caller applies cash_delta to its cash balance.
    """
    fills: list[GridFill] = []
    total_cash_delta = 0.0

    for ch in state.channels:
        if ch.status == "waiting" and bar_low <= ch.buy_price:
            # Buy at the level price (limit order semantics).
            fee = ch.alloc * fee_rate
            units = ch.alloc / ch.buy_price
            cash_delta = -(ch.alloc + fee)
            total_cash_delta += cash_delta
            ch.units = units
            ch.status = "holding"
            ch.last_buy_time = bar_time
            state.cash_committed += ch.alloc
            fills.append(GridFill(
                time=bar_time, price=ch.buy_price, side="buy",
                units=units, cash_delta=cash_delta,
            ))
        if ch.status == "holding" and bar_high >= ch.sell_price:
            proceeds = ch.units * ch.sell_price
            fee = proceeds * fee_rate
            cash_delta = proceeds - fee
            cycle_pnl = (ch.sell_price - ch.buy_price) * ch.units - (ch.alloc * fee_rate) - fee
            total_cash_delta += cash_delta
            state.realized_pnl += cycle_pnl
            state.cash_committed -= ch.alloc
            state.cycles += 1
            fills.append(GridFill(
                time=bar_time, price=ch.sell_price, side="sell",
                units=ch.units, cash_delta=cash_delta, pnl=cycle_pnl,
            ))
            ch.units = 0.0
            ch.status = "waiting"
            ch.last_buy_time = None

    return fills, total_cash_delta


def liquidate_grid(
    state: GridState,
    bar_time: pd.Timestamp,
    exit_price: float,
    fee_rate: float,
) -> tuple[list[GridFill], float]:
    """Close all open channels at `exit_price` (called on regime change)."""
    fills: list[GridFill] = []
    total_cash_delta = 0.0
    for ch in state.channels:
        if ch.status == "holding":
            proceeds = ch.units * exit_price
            fee = proceeds * fee_rate
            cash_delta = proceeds - fee
            forced_pnl = (exit_price - ch.buy_price) * ch.units - (ch.alloc * fee_rate) - fee
            total_cash_delta += cash_delta
            state.realized_pnl += forced_pnl
            state.cash_committed -= ch.alloc
            fills.append(GridFill(
                time=bar_time, price=exit_price, side="sell",
                units=ch.units, cash_delta=cash_delta, pnl=forced_pnl,
            ))
            ch.units = 0.0
            ch.status = "waiting"
            ch.last_buy_time = None
    return fills, total_cash_delta


def grid_mark_to_market(state: GridState, mark_price: float) -> float:
    """Unrealized value of open positions at mark_price."""
    total = 0.0
    for ch in state.channels:
        if ch.status == "holding":
            total += ch.units * mark_price
    return total
