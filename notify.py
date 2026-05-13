"""Discord notifications for trade events, errors, and a daily check-in.

Reads `DISCORD_WEBHOOK_URL` from the environment. If it's unset, every notify
call becomes a silent no-op so the bot runs identically with or without
alerting configured.

Design rules:
  - This module NEVER raises. A Discord outage must not interrupt the trade loop.
  - Notifications fire only for things a human would want to know about:
      live orders, errors, halts, the daily summary.
  - Dry-run runs stay quiet by default (the bot passes notify_enabled=False).

Test from the CLI:
    set DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
    python notify.py "hello from the bot"
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import requests

from logger import log

WEBHOOK_ENV = "DISCORD_WEBHOOK_URL"

_COLOR_INFO    = 0x3498DB
_COLOR_SUCCESS = 0x2ECC71
_COLOR_WARNING = 0xF39C12
_COLOR_ERROR   = 0xE74C3C
_COLOR_DAILY   = 0x9B59B6


def _webhook_url() -> str | None:
    url = os.environ.get(WEBHOOK_ENV, "").strip()
    return url if url else None


def is_configured() -> bool:
    return _webhook_url() is not None


def _post_embed(
    title: str,
    description: str,
    color: int,
    fields: list[dict[str, Any]] | None = None,
    footer: str = "bitget-bot",
) -> None:
    url = _webhook_url()
    if not url:
        return
    embed: dict[str, Any] = {
        "title":       title[:256],
        "description": description[:4000],
        "color":       color,
        "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "footer":      {"text": footer},
    }
    if fields:
        embed["fields"] = fields[:25]
    try:
        resp = requests.post(url, json={"embeds": [embed]}, timeout=5)
        if resp.status_code >= 400:
            log.warning(f"Discord notify HTTP {resp.status_code}: {resp.text[:200]!r}")
    except Exception as e:
        log.warning(f"Discord notify failed: {type(e).__name__}: {e}")


# ---------- Public API ----------

def notify_trade(
    *,
    action:       str,                    # "open_carry" | "close_carry" | "open_ema" | "close_ema"
    symbol:       str,
    side:         str,                    # "Buy" | "Sell"
    qty:          str | float,
    notional_usd: float,
    reason:       str,
    extra:        dict[str, str] | None = None,
) -> None:
    emoji = {
        "open_carry":  "🟢",
        "close_carry": "🔴",
        "open_ema":    "📈",
        "close_ema":   "📉",
    }.get(action, "📌")
    title = f"{emoji}  {action.upper().replace('_', ' ')} — {symbol}"
    desc = f"**{side} {qty}** {symbol}  ·  notional **${notional_usd:,.2f}**"
    fields: list[dict[str, Any]] = [{"name": "Reason", "value": reason, "inline": False}]
    if extra:
        for k, v in extra.items():
            fields.append({"name": k, "value": str(v), "inline": True})
    color = _COLOR_SUCCESS if action.startswith("open") else _COLOR_INFO
    _post_embed(title, desc, color, fields)


def notify_error(message: str, cycle: int | None = None) -> None:
    title = f"⚠️  Cycle {cycle} crashed" if cycle is not None else "⚠️  Bot error"
    _post_embed(title, f"```{message[:1900]}```", _COLOR_ERROR)


def notify_halt(reason: str) -> None:
    _post_embed("🛑  Safety halt — bot stopped trading", reason[:1900], _COLOR_WARNING)


def notify_daily_summary(
    *,
    equity_usd:   float,
    carry_target: str | None,
    ema_holding:  bool,
    extra_lines:  list[str] | None = None,
) -> None:
    title = "📊  Daily check-in"
    desc = f"Account equity: **${equity_usd:,.2f}**"
    fields = [
        {"name": "Carry", "value": carry_target or "flat", "inline": True},
        {"name": "EMA",   "value": "long" if ema_holding else "flat", "inline": True},
    ]
    if extra_lines:
        fields.append({"name": "Notes", "value": "\n".join(extra_lines), "inline": False})
    _post_embed(title, desc, _COLOR_DAILY, fields)


def notify_test(message: str = "hello from the bot — Discord webhook is working") -> None:
    _post_embed("✅  Test notification", message, _COLOR_INFO)


# ---------- CLI ----------

if __name__ == "__main__":
    if not is_configured():
        sys.stderr.write(
            "DISCORD_WEBHOOK_URL not set. "
            "Set the env var (or put it in .env) and re-run.\n"
        )
        raise SystemExit(2)
    msg = " ".join(sys.argv[1:]) or "hello from the bot — Discord webhook is working"
    notify_test(msg)
    print(f"Sent: {msg}")
