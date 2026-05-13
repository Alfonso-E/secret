"""Configuration loader — reads credentials and runtime knobs from environment.

Loading priority (highest first):
  1. Real environment variables (os.environ).
  2. A `.env` file in this directory.

Bitget is the primary venue. The legacy BybitConfig is kept for compatibility
with the older backtest scripts that reference cached Bybit data, but new live
code should use BitgetConfig.

A secret loaded into a config is `repr`-masked so it never leaks into logs or
stack traces by accident.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# --- Bitget ----------------------------------------------------------------

BitgetEnv = Literal["demo", "live"]

# Bitget uses a single production base URL for both demo and live. Demo is
# enabled by adding the header `paptrading: 1` to authenticated requests.
BITGET_BASE_URL = "https://api.bitget.com"


def _load_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _masked(value: str) -> str:
    if not value:
        return "<empty>"
    if len(value) <= 8:
        return "***"
    return f"{value[:3]}...{value[-3:]}"


@dataclass
class BitgetConfig:
    api_key:    str
    api_secret: str = field(repr=False)
    passphrase: str = field(repr=False)
    env:        BitgetEnv = "demo"
    recv_window_ms: int = 5000   # unused by Bitget but kept for parity with old code

    @property
    def base_url(self) -> str:
        return BITGET_BASE_URL

    @property
    def is_demo(self) -> bool:
        return self.env == "demo"

    def __repr__(self) -> str:
        return (
            f"BitgetConfig(env={self.env!r}, base_url={self.base_url!r}, "
            f"api_key={_masked(self.api_key)!r}, "
            f"api_secret=<hidden>, passphrase=<hidden>)"
        )


def load_bitget_config(env_path: Path | None = None) -> BitgetConfig:
    """Read BITGET_* env vars (process env wins over .env file)."""
    if env_path is None:
        env_path = Path(__file__).parent / ".env"
    dotenv_vars = _load_dotenv(env_path)

    def get(name: str, default: str = "") -> str:
        return os.environ.get(name) or dotenv_vars.get(name, default)

    env_str = get("BITGET_ENV", "demo").lower()
    if env_str not in ("demo", "live"):
        raise ValueError(f"BITGET_ENV must be 'demo' or 'live', got {env_str!r}")

    key = get("BITGET_API_KEY")
    secret = get("BITGET_API_SECRET")
    passphrase = get("BITGET_API_PASSPHRASE")
    if (not key or not secret or not passphrase
            or key == "your_api_key_here"):
        raise RuntimeError(
            "Bitget API credentials missing. Copy .env.example to .env and fill in "
            "BITGET_API_KEY, BITGET_API_SECRET, and BITGET_API_PASSPHRASE."
        )

    return BitgetConfig(api_key=key, api_secret=secret, passphrase=passphrase, env=env_str)  # type: ignore[arg-type]


# --- Legacy Bybit (kept for old backtest scripts) --------------------------

BybitEnv = Literal["demo", "testnet", "mainnet"]
_BYBIT_BASE_URLS: dict[str, str] = {
    "mainnet": "https://api.bybit.com",
    "testnet": "https://api-testnet.bybit.com",
    "demo":    "https://api-demo.bybit.com",
}


@dataclass
class BybitConfig:
    """LEGACY — kept only so old scripts importing this don't crash. Don't use for live."""
    api_key:    str
    api_secret: str = field(repr=False)
    env:        BybitEnv = "demo"
    recv_window_ms: int = 5000

    @property
    def base_url(self) -> str:
        return _BYBIT_BASE_URLS[self.env]

    def __repr__(self) -> str:
        return (
            f"BybitConfig(env={self.env!r}, base_url={self.base_url!r}, "
            f"api_key={_masked(self.api_key)!r}, api_secret=<hidden>)"
        )
