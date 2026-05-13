"""Translate strategy intent into exchange-ready (qty, price) pairs.

For the carry trade:
  Given $C of capital and leverage L on the perp short, the market-neutral
  notional N satisfies:  N = C * L / (L + 1)
  (capital is split between spot collateral and perp margin).

  Spot leg:  long  N / spot_price  units
  Perp leg:  short N / perp_price  units

For the EMA directional overlay:
  Long-only spot. Position size = capital / spot_price.
"""

from __future__ import annotations

from dataclasses import dataclass

from bybit_symbols import SymbolInfo


@dataclass
class CarryLegSpec:
    spot_info:    SymbolInfo
    perp_info:    SymbolInfo
    spot_price:   float
    perp_price:   float
    spot_qty:     float    # base coin to BUY on spot
    perp_qty:     float    # base coin to SHORT on perp
    notional_usd: float
    fits_exchange: bool
    rejection_reason: str | None = None


def carry_leg_size(
    *,
    spot_info:  SymbolInfo,
    perp_info:  SymbolInfo,
    spot_price: float,
    perp_price: float,
    capital_usd: float,
    leverage:   float,
) -> CarryLegSpec:
    """Compute the spot+perp quantities for a single asset's carry slice.

    Returns a CarryLegSpec; if the trade would violate exchange filters
    (lot size, min notional), `fits_exchange=False` and the reason is reported.
    The caller decides whether to skip this asset or reduce position elsewhere.
    """
    if leverage <= 0:
        raise ValueError("leverage must be > 0")
    notional = capital_usd * leverage / (leverage + 1)
    spot_qty_raw = notional / spot_price
    perp_qty_raw = notional / perp_price

    from bybit_symbols import format_qty, validate_order
    spot_qty = float(format_qty(spot_info, spot_qty_raw))
    perp_qty = float(format_qty(perp_info, perp_qty_raw))

    spot_err = validate_order(spot_info, spot_qty, spot_price)
    perp_err = validate_order(perp_info, perp_qty, perp_price)
    fits = (spot_err is None) and (perp_err is None)
    reason = None
    if spot_err: reason = f"spot: {spot_err}"
    elif perp_err: reason = f"perp: {perp_err}"

    return CarryLegSpec(
        spot_info=spot_info, perp_info=perp_info,
        spot_price=spot_price, perp_price=perp_price,
        spot_qty=spot_qty, perp_qty=perp_qty,
        notional_usd=notional, fits_exchange=fits, rejection_reason=reason,
    )


@dataclass
class EmaPositionSpec:
    info:         SymbolInfo
    price:        float
    qty:          float
    notional_usd: float
    fits_exchange: bool
    rejection_reason: str | None = None


def ema_position_size(
    *,
    info:        SymbolInfo,
    price:       float,
    capital_usd: float,
) -> EmaPositionSpec:
    """Compute spot qty to buy for the directional EMA strategy."""
    from bybit_symbols import format_qty, validate_order
    raw = capital_usd / price
    qty = float(format_qty(info, raw))
    err = validate_order(info, qty, price)
    return EmaPositionSpec(
        info=info, price=price, qty=qty, notional_usd=qty * price,
        fits_exchange=(err is None), rejection_reason=err,
    )
