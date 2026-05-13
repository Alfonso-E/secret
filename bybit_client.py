"""Authenticated client for Bybit v5 REST API.

Implements the v5 HMAC-SHA256 scheme:
  signature_payload = timestamp + api_key + recv_window + (queryString OR jsonBody)
  X-BAPI-SIGN = hex( hmac_sha256(api_secret, payload) )

Headers sent on every authenticated request:
  X-BAPI-API-KEY, X-BAPI-TIMESTAMP, X-BAPI-RECV-WINDOW, X-BAPI-SIGN

Clock skew: Bybit accepts timestamps in [server_time - recv_window, server_time + 1000).
We auto-sync against /v5/market/time on the first authenticated call and apply
a local offset thereafter; you do NOT need NTP to be configured.

Safety:
  - Secrets are never logged.
  - Network errors and non-zero retCodes raise clear exceptions.
  - This module knows nothing about strategies — it just signs and sends.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any
from urllib.parse import urlencode

import requests

from config import BybitConfig


class BybitAPIError(RuntimeError):
    """Raised when Bybit returns a non-zero retCode."""

    def __init__(self, ret_code: int, ret_msg: str, endpoint: str):
        super().__init__(f"Bybit API error retCode={ret_code} retMsg={ret_msg!r} on {endpoint}")
        self.ret_code = ret_code
        self.ret_msg = ret_msg
        self.endpoint = endpoint


class BybitClient:
    """Thin authenticated REST client for Bybit V5.

    Use `.get(path, params)` for GET requests and `.post(path, body)` for POST.
    Server-time sync is performed lazily on the first authenticated call.
    """

    def __init__(self, config: BybitConfig, session: requests.Session | None = None):
        self.config = config
        self.session = session or requests.Session()
        self._time_offset_ms: int | None = None

    # --- Public helpers --------------------------------------------------

    def get(self, path: str, params: dict[str, Any] | None = None, *, authenticated: bool = True) -> dict[str, Any]:
        params = params or {}
        query = urlencode(sorted(params.items())) if params else ""
        url = f"{self.config.base_url}{path}{'?' + query if query else ''}"
        headers = self._auth_headers(query) if authenticated else {}
        resp = self.session.get(url, headers=headers, timeout=15)
        return self._parse_response(resp, path)

    def post(self, path: str, body: dict[str, Any] | None = None, *, authenticated: bool = True) -> dict[str, Any]:
        body_str = json.dumps(body, separators=(",", ":")) if body else ""
        url = f"{self.config.base_url}{path}"
        headers = self._auth_headers(body_str) if authenticated else {}
        headers["Content-Type"] = "application/json"
        resp = self.session.post(url, data=body_str, headers=headers, timeout=15)
        return self._parse_response(resp, path)

    def server_time_ms(self) -> int:
        """Public endpoint, no auth."""
        data = self.get("/v5/market/time", authenticated=False)
        return int(data["result"]["timeNano"]) // 1_000_000

    # --- Internals -------------------------------------------------------

    def _sync_time(self) -> None:
        local_ms = int(time.time() * 1000)
        server_ms = self.server_time_ms()
        self._time_offset_ms = server_ms - local_ms

    def _timestamp_ms(self) -> int:
        if self._time_offset_ms is None:
            self._sync_time()
        return int(time.time() * 1000) + (self._time_offset_ms or 0)

    def _sign(self, timestamp_ms: int, payload: str) -> str:
        message = f"{timestamp_ms}{self.config.api_key}{self.config.recv_window_ms}{payload}"
        return hmac.new(
            self.config.api_secret.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()

    def _auth_headers(self, payload: str) -> dict[str, str]:
        ts = self._timestamp_ms()
        return {
            "X-BAPI-API-KEY":     self.config.api_key,
            "X-BAPI-TIMESTAMP":   str(ts),
            "X-BAPI-RECV-WINDOW": str(self.config.recv_window_ms),
            "X-BAPI-SIGN":        self._sign(ts, payload),
        }

    def _parse_response(self, resp: requests.Response, path: str) -> dict[str, Any]:
        resp.raise_for_status()
        data = resp.json()
        ret_code = data.get("retCode")
        if ret_code != 0:
            raise BybitAPIError(ret_code, data.get("retMsg", ""), path)
        return data
