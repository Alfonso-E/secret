"""Persistent bot state across cron runs.

A cron-driven bot has no in-memory state — each `live_bot.py` invocation
starts fresh. That broke our session-drawdown safety guard, which compared
current equity against the "session high" set at process start (always
equal to current equity in a fresh process).

This module persists a tiny JSON file at `logs/state.json` that survives
between runs (via the GitHub Actions cache in cloud deployments, or just
the filesystem on a VPS). Tracks:
  - first-run timestamp + total runs
  - peak equity ever observed + the timestamp of that peak
  - recent halt reasons (so we don't oscillate enter/exit on the same trigger)

The file is small (<1 KB), atomically written, and resilient to missing
or corrupted reads — a fresh state is always a valid fallback.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from logger import LOG_DIR, log

STATE_FILE = LOG_DIR / "state.json"
STATE_SCHEMA_VERSION = 1


@dataclass
class BotState:
    schema_version:    int = STATE_SCHEMA_VERSION
    first_run_utc:     str = ""
    last_run_utc:      str = ""
    total_runs:        int = 0
    peak_equity_usd:   float = 0.0
    peak_equity_utc:   str = ""
    last_halt_reason:  str = ""
    last_halt_utc:     str = ""

    def drawdown_from_peak(self, current_equity_usd: float) -> float:
        """Negative number, e.g. -0.07 = -7% drawdown from all-time peak."""
        if self.peak_equity_usd <= 0:
            return 0.0
        return (current_equity_usd / self.peak_equity_usd) - 1.0


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_state() -> BotState:
    """Read state from disk; return a fresh BotState if anything is wrong."""
    if not STATE_FILE.exists():
        log.info("  No prior state file — starting fresh.")
        return BotState(first_run_utc=_utc_now())
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if data.get("schema_version") != STATE_SCHEMA_VERSION:
            log.warning(f"  State schema mismatch (file={data.get('schema_version')}, "
                        f"expected={STATE_SCHEMA_VERSION}). Starting fresh.")
            return BotState(first_run_utc=_utc_now())
        return BotState(**{k: v for k, v in data.items() if k in BotState.__annotations__})
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        log.warning(f"  Could not parse state file ({e}). Starting fresh.")
        return BotState(first_run_utc=_utc_now())


def save_state(state: BotState) -> None:
    """Atomic write: write to a temp file then rename, so a crash mid-write
    can't leave a corrupted state.json."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")
    os.replace(tmp, STATE_FILE)


def update_for_cycle(state: BotState, current_equity_usd: float) -> BotState:
    """Bookkeep one cycle: bump run count, update peak if exceeded."""
    state.total_runs += 1
    state.last_run_utc = _utc_now()
    if not state.first_run_utc:
        state.first_run_utc = state.last_run_utc
    if current_equity_usd > state.peak_equity_usd:
        state.peak_equity_usd = current_equity_usd
        state.peak_equity_utc = state.last_run_utc
    return state


def record_halt(state: BotState, reason: str) -> BotState:
    state.last_halt_reason = reason[:200]
    state.last_halt_utc = _utc_now()
    return state
