"""Authenticated client for Bitget V2 REST API.

Signing scheme (per Bitget docs):
  payload = timestamp + METHOD + path + ('?' + queryString if any) + body
  ACCESS-SIGN = base64( hmac_sha256(api_secret, payload) )

Required headers on authenticated requests:
  ACCESS-KEY, ACCESS-SIGN, ACCESS-PASSPHRASE, ACCESS-TIMESTAMP, locale

Demo trading is enabled by adding `paptrading: 1` to every authenticated
request — Bitget uses the same base URL for live and demo, distinguished by
this header. Demo keys only validate when the header is present.

Clock skew: like Bybit, Bitget rejects timestamps that drift too far from
their server clock. We auto-sync once on first authenticated call and apply
a local offset thereafter.

Success indicator: Bitget returns `code` as the STRING "00000" on success
(NOT an int 0 — distinct from Bybit). Anything else is an error.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any
from urllib.parse import urlencode

import requests

from config import BitgetConfig


class BitgetAPIError(RuntimeError):
    """Raised when Bitget returns a non-success code."""

    def __init__(self, code: str, msg: str, endpoint: str):
        super().__init__(f"Bitget API error code={code} msg={msg!r} on {endpoint}")
        self.code = code
        self.msg = msg
        self.endpoint = endpoint


_SUCCESS_CODE = "00000"


class BitgetClient:
    """Thin authenticated REST client for Bitget V2.

    Use `.get(path, params)` for GET requests and `.post(path, body)` for POST.
    Pass `authenticated=False` to skip signing for public endpoints.
    """

    def __init__(self, config: BitgetConfig, session: requests.Session | None = None):
        self.config = config
        self.session = session or requests.Session()
        self._time_offset_ms: int | None = None

    # --- Public helpers --------------------------------------------------

    def get(self, path: str, params: dict[str, Any] | None = None, *, authenticated: bool = True) -> dict[str, Any]:
        params = params or {}
        query = urlencode(sorted((k, str(v)) for k, v in params.items())) if params else ""
        url = f"{self.config.base_url}{path}{'?' + query if query else ''}"
        headers = self._auth_headers("GET", path, query, "") if authenticated else {}
        if authenticated and self.config.is_demo:
            headers["paptrading"] = "1"
        resp = self.session.get(url, headers=headers, timeout=15)
        return self._parse_response(resp, path)

    def post(self, path: str, body: dict[str, Any] | None = None, *, authenticated: bool = True) -> dict[str, Any]:
        body_str = json.dumps(body, separators=(",", ":")) if body else ""
        url = f"{self.config.base_url}{path}"
        headers = self._auth_headers("POST", path, "", body_str) if authenticated else {}
        if authenticated and self.config.is_demo:
            headers["paptrading"] = "1"
        headers["Content-Type"] = "application/json"
        resp = self.session.post(url, data=body_str, headers=headers, timeout=15)
        return self._parse_response(resp, path)

    def server_time_ms(self) -> int:
        """Public endpoint, no auth."""
        data = self.get("/api/v2/public/time", authenticated=False)
        return int(data["data"]["serverTime"])

    # --- Internals -------------------------------------------------------

    def _sync_time(self) -> None:
        local_ms = int(time.time() * 1000)
        server_ms = self.server_time_ms()
        self._time_offset_ms = server_ms - local_ms

    def _timestamp_ms(self) -> int:
        if self._time_offset_ms is None:
            self._sync_time()
        return int(time.time() * 1000) + (self._time_offset_ms or 0)

    def _sign(self, timestamp_ms: int, method: str, path: str, query: str, body: str) -> str:
        payload = f"{timestamp_ms}{method.upper()}{path}"
        if query:
            payload += f"?{query}"
        payload += body
        digest = hmac.new(
            self.config.api_secret.encode(),
            payload.encode(),
            hashlib.sha256,
        ).digest()
        return base64.b64encode(digest).decode()

    def _auth_headers(self, method: str, path: str, query: str, body: str) -> dict[str, str]:
        ts = self._timestamp_ms()
        return {
            "ACCESS-KEY":        self.config.api_key,
            "ACCESS-SIGN":       self._sign(ts, method, path, query, body),
            "ACCESS-PASSPHRASE": self.config.passphrase,
            "ACCESS-TIMESTAMP":  str(ts),
            "locale":            "en-US",
        }

    def _parse_response(self, resp: requests.Response, path: str) -> dict[str, Any]:
        try:
            data = resp.json()
        except ValueError:
            resp.raise_for_status()
            raise RuntimeError(f"Non-JSON response from {path}: {resp.text[:300]!r}")

        code = str(data.get("code", ""))
        if code != _SUCCESS_CODE:
            raise BitgetAPIError(code, data.get("msg", ""), path)
        if not resp.ok:
            # Bitget returned 4xx/5xx but the JSON didn't indicate an error code.
            resp.raise_for_status()
        return data
