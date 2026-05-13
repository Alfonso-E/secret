"""Pre-trade safety guards and runtime kill-switches.

The bot calls `SafetyGuards.check(...)` before every order. If any check
fails, the order is REJECTED at the application layer — it never even
reaches the dry-run printer.

Guards are intentionally conservative. You can loosen them by passing
different thresholds at construction, but the defaults are designed so the
bot is unable to do anything catastrophic without an explicit override.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


class SafetyError(RuntimeError):
    """Raised when a safety guard rejects an action."""


@dataclass
class SafetyLimits:
    # Account-level
    min_equity_usd: float = 100.0          # refuse to operate if equity below this
    max_total_notional_usd: float = 50_000.0 # absolute cap on combined positions
    max_single_position_usd: float = 15_000.0 # cap on any one asset

    # Daily P&L
    max_daily_loss_pct: float = 5.0          # auto-flatten if equity drops > 5% from session high

    # Leverage cap (defensive — strategy uses 5x, this is the absolute ceiling)
    max_leverage: float = 7.0

    # Data freshness
    max_data_age_seconds: int = 600          # refuse if last kline / funding > 10 min old

    # Behavioral guards
    require_unique_clord: bool = True        # prevent accidental duplicate orders


@dataclass
class SessionState:
    """Mutable session-level state that the guards inspect."""
    session_started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_high_equity: float = 0.0
    current_equity:      float = 0.0
    total_notional_usd:  float = 0.0          # currently deployed
    per_asset_notional:  dict[str, float] = field(default_factory=dict)
    orders_this_session: set[str] = field(default_factory=set)

    def update_equity(self, equity: float) -> None:
        self.current_equity = equity
        if equity > self.session_high_equity:
            self.session_high_equity = equity

    def drawdown_pct(self) -> float:
        if self.session_high_equity <= 0:
            return 0.0
        return (self.current_equity / self.session_high_equity - 1.0) * 100.0


class SafetyGuards:
    def __init__(self, limits: SafetyLimits, state: SessionState):
        self.limits = limits
        self.state = state

    # --- Pre-trade checks --------------------------------------------------

    def check_can_trade(self) -> None:
        """Account-level sanity. Call once before each rebalance cycle."""
        eq = self.state.current_equity
        if eq < self.limits.min_equity_usd:
            raise SafetyError(
                f"Equity ${eq:,.2f} < min_equity_usd ${self.limits.min_equity_usd:,.2f} — refusing to trade"
            )
        dd = self.state.drawdown_pct()
        if dd <= -self.limits.max_daily_loss_pct:
            raise SafetyError(
                f"Drawdown {dd:.2f}% from session high exceeds limit {-self.limits.max_daily_loss_pct:.2f}% — flatten and halt"
            )

    def check_order(
        self,
        *,
        symbol:        str,
        notional_usd:  float,
        client_order_id: str | None = None,
    ) -> None:
        """Order-level guard. Call before EVERY place_order."""
        if notional_usd <= 0:
            raise SafetyError(f"notional_usd must be > 0 (got {notional_usd})")

        existing = self.state.per_asset_notional.get(symbol, 0.0)
        if existing + notional_usd > self.limits.max_single_position_usd:
            raise SafetyError(
                f"{symbol}: would exceed max_single_position_usd "
                f"(existing ${existing:,.0f} + new ${notional_usd:,.0f} > ${self.limits.max_single_position_usd:,.0f})"
            )

        if self.state.total_notional_usd + notional_usd > self.limits.max_total_notional_usd:
            raise SafetyError(
                f"would exceed max_total_notional_usd "
                f"(${self.state.total_notional_usd:,.0f} + ${notional_usd:,.0f} > ${self.limits.max_total_notional_usd:,.0f})"
            )

        if self.limits.require_unique_clord and client_order_id is not None:
            if client_order_id in self.state.orders_this_session:
                raise SafetyError(f"duplicate client_order_id {client_order_id!r}")
            self.state.orders_this_session.add(client_order_id)

    def check_data_age(self, age_seconds: float, kind: str) -> None:
        if age_seconds > self.limits.max_data_age_seconds:
            raise SafetyError(
                f"{kind} data is {age_seconds:.0f}s old (limit {self.limits.max_data_age_seconds}s) — refusing to act"
            )

    # --- Post-trade bookkeeping -------------------------------------------

    def register_position_change(self, symbol: str, notional_delta_usd: float) -> None:
        cur = self.state.per_asset_notional.get(symbol, 0.0)
        new = max(0.0, cur + notional_delta_usd)
        self.state.per_asset_notional[symbol] = new
        self.state.total_notional_usd = sum(self.state.per_asset_notional.values())


def should_flatten(state: SessionState, limits: SafetyLimits) -> tuple[bool, str | None]:
    """Quick yes/no for the scheduler's dead-man check."""
    if state.current_equity < limits.min_equity_usd:
        return True, f"equity ${state.current_equity:,.0f} < ${limits.min_equity_usd:,.0f}"
    dd = state.drawdown_pct()
    if dd <= -limits.max_daily_loss_pct:
        return True, f"drawdown {dd:.2f}% < -{limits.max_daily_loss_pct:.2f}%"
    return False, None
