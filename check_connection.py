"""Smoke test for the Bitget authenticated client.

Reads credentials from environment / .env (NEVER from hard-coded values), then:
  1. Pings /api/v2/public/time (public, no auth) — verifies network + clock.
  2. Calls wallet balance endpoints (authenticated) — verifies key + secret + passphrase.
  3. Lists USDT-perp positions and pending orders.

Run:  python check_connection.py
Expected:  resolved env, masked key, server time, wallet snapshot, positions.
"""

from __future__ import annotations

import sys
import time

from bitget_account import get_open_orders, get_positions, get_wallet_balance
from bitget_client import BitgetAPIError, BitgetClient
from config import load_bitget_config


def main() -> int:
    try:
        cfg = load_bitget_config()
    except RuntimeError as e:
        print(f"[FAIL] config load: {e}", file=sys.stderr)
        return 2

    print(f"Loaded: {cfg!r}")
    client = BitgetClient(cfg)

    # 1. Public time
    print()
    print("=== Server time check (public) ===")
    try:
        server_ms = client.server_time_ms()
        local_ms  = int(time.time() * 1000)
        skew = local_ms - server_ms
        print(f"  Server: {server_ms} ms  |  Local: {local_ms} ms  |  Skew: {skew:+d} ms")
        if abs(skew) > 4500:
            print("  WARNING: clock skew > 4500ms — your machine clock is significantly off.")
    except Exception as e:
        print(f"[FAIL] Could not reach Bitget: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    # 2. Wallet
    print()
    print("=== Wallet balance (authenticated) ===")
    try:
        snap = get_wallet_balance(client)
        print(f"  Account:            {snap.account_type}")
        print(f"  Total equity (USD): ${snap.total_equity_usd:,.2f}")
        print(f"  Available (USD):    ${snap.total_available_usd:,.2f}")
        print(f"  Margin in use:      ${snap.total_margin_usd:,.2f}")
        if snap.coins:
            print(f"  Holdings:")
            for c in snap.coins:
                print(f"    {c.coin:30s}  balance={c.wallet_balance:>14.6f}  usd=${c.usd_value:,.2f}")
        else:
            print("  No coin balances yet (expected for a fresh demo account before funding).")
    except BitgetAPIError as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        if e.code in ("40001", "40002", "40003", "40006"):
            print("  Hint: invalid key/sig/passphrase. Re-check .env entries.", file=sys.stderr)
        if e.code == "40037":
            print("  Hint: 'paptrading' header mismatch. If your key is a DEMO key, "
                  "BITGET_ENV must be 'demo'; if LIVE, set BITGET_ENV=live.", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[FAIL] Wallet fetch: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    # 3. Positions
    print()
    print("=== Open positions (USDT-perp) ===")
    try:
        pos = get_positions(client)
        print("  None." if pos.empty else pos.to_string(index=False))
    except BitgetAPIError as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        return 1

    # 4. Open orders
    print()
    print("=== Open orders (USDT-perp) ===")
    try:
        oo = get_open_orders(client)
        print("  None." if oo.empty else oo.to_string(index=False))
    except BitgetAPIError as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        return 1

    print()
    print("[OK] All checks passed. The client is ready for live order placement next turn.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
