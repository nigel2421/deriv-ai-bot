"""
Deriv Developers API (v2) authentication helpers.

New OAuth / PAT apps use:
  - REST base: https://api.derivws.com
  - Headers: Authorization: Bearer <token>, Deriv-App-ID: <app_id>
  - Authenticated WebSocket: POST .../options/accounts/{id}/otp → connect to returned URL

This is distinct from the legacy `wss://ws.binaryws.com/websockets/v3?app_id=NUMBER`
flow that accepts only numeric app IDs.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import secrets
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import httpx

logger = logging.getLogger(__name__)

DEFAULT_API_BASE = "https://api.derivws.com"
DEFAULT_AUTH_URL = "https://auth.deriv.com/oauth2/auth"
DEFAULT_TOKEN_URL = "https://auth.deriv.com/oauth2/token"


def is_legacy_app_id(app_id: str) -> bool:
    """Legacy WebSocket app_id is numeric (e.g. 1089)."""
    return str(app_id).strip().isdigit()


def api_headers(app_id: str, bearer_token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {bearer_token}",
        "Deriv-App-ID": str(app_id),
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


async def list_options_accounts(
    app_id: str,
    bearer_token: str,
    *,
    api_base: str = DEFAULT_API_BASE,
    timeout: float = 30.0,
) -> List[Dict[str, Any]]:
    """GET /trading/v1/options/accounts"""
    url = f"{api_base.rstrip('/')}/trading/v1/options/accounts"
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(url, headers=api_headers(app_id, bearer_token))
        if r.status_code >= 400:
            logger.error(
                "list_options_accounts HTTP %s: %s", r.status_code, r.text[:500]
            )
            r.raise_for_status()
        payload = r.json()
    # Response shapes: {data: [...]} or [...]
    if isinstance(payload, list):
        return payload
    data = payload.get("data")
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "accounts" in data:
        return list(data["accounts"])
    logger.warning("Unexpected accounts payload keys: %s", list(payload.keys()))
    return []


def pick_account(
    accounts: List[Dict[str, Any]],
    *,
    mode: str = "demo",
    preferred_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if not accounts:
        return None
    if preferred_id:
        for a in accounts:
            aid = str(a.get("account_id") or a.get("loginid") or a.get("id") or "")
            if aid == preferred_id:
                return a

    want_demo = (mode or "demo").lower() != "real"

    def _is_demo(a: Dict[str, Any]) -> bool:
        aid = str(a.get("account_id") or a.get("loginid") or "")
        group = str(a.get("group") or a.get("account_type") or "").lower()
        if aid.startswith(("VRT", "VRTC", "VRW")):
            return True
        if "demo" in group:
            return True
        return False

    filtered = [a for a in accounts if _is_demo(a) == want_demo]
    pool = filtered or accounts
    return pool[0]


async def fetch_otp_websocket_url(
    app_id: str,
    bearer_token: str,
    account_id: str,
    *,
    api_base: str = DEFAULT_API_BASE,
    timeout: float = 30.0,
) -> str:
    """POST /trading/v1/options/accounts/{accountId}/otp → websocket URL."""
    url = (
        f"{api_base.rstrip('/')}/trading/v1/options/accounts/{account_id}/otp"
    )
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, headers=api_headers(app_id, bearer_token))
        if r.status_code >= 400:
            logger.error("fetch_otp HTTP %s: %s", r.status_code, r.text[:500])
            r.raise_for_status()
        payload = r.json()

    # {data: {url: "wss://..."}} or {url: "..."}
    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, dict) and data.get("url"):
        return str(data["url"])
    if isinstance(payload, dict) and payload.get("url"):
        return str(payload["url"])
    raise RuntimeError(f"OTP response missing url: {payload}")


async def resolve_authenticated_ws_url(
    app_id: str,
    bearer_token: str,
    *,
    mode: str = "demo",
    account_id: Optional[str] = None,
    api_base: str = DEFAULT_API_BASE,
) -> Tuple[str, Dict[str, Any]]:
    """
    Full v2 auth path: list accounts → pick one → OTP URL.
    Returns (websocket_url, account_dict).
    """
    accounts = await list_options_accounts(app_id, bearer_token, api_base=api_base)
    if not accounts:
        raise RuntimeError(
            "No options trading accounts returned. "
            "Ensure your OAuth/PAT app has access and the account has Options enabled."
        )
    account = pick_account(accounts, mode=mode, preferred_id=account_id)
    if not account:
        raise RuntimeError("Could not select a trading account from API response")
    aid = str(
        account.get("account_id")
        or account.get("loginid")
        or account.get("id")
        or ""
    )
    if not aid:
        raise RuntimeError(f"Account missing id: {account}")
    ws_url = await fetch_otp_websocket_url(
        app_id, bearer_token, aid, api_base=api_base
    )
    # Normalize https → wss if server returns https
    if ws_url.startswith("https://"):
        ws_url = "wss://" + ws_url[len("https://") :]
    elif ws_url.startswith("http://"):
        ws_url = "ws://" + ws_url[len("http://") :]
    logger.info(
        "V2 OTP WebSocket URL obtained for account %s (mode=%s)", aid, mode
    )
    return ws_url, account


# ---- OAuth2 PKCE helpers (web login) ---------------------------------


def generate_pkce_pair() -> Tuple[str, str]:
    """Returns (code_verifier, code_challenge S256)."""
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def build_oauth_authorize_url(
    client_id: str,
    redirect_uri: str,
    *,
    code_challenge: str,
    state: str,
    auth_url: str = DEFAULT_AUTH_URL,
    scope: str = "trade",
) -> str:
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
        "scope": scope,
    }
    return f"{auth_url}?{urlencode(params)}"


async def exchange_oauth_code(
    client_id: str,
    code: str,
    redirect_uri: str,
    code_verifier: str,
    *,
    token_url: str = DEFAULT_TOKEN_URL,
    client_secret: Optional[str] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Exchange authorization code for access_token (PKCE)."""
    data = {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "code": code,
        "redirect_uri": redirect_uri,
        "code_verifier": code_verifier,
    }
    if client_secret:
        data["client_secret"] = client_secret
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(
            token_url,
            data=data,
            headers={"Accept": "application/json"},
        )
        if r.status_code >= 400:
            logger.error("OAuth token exchange HTTP %s: %s", r.status_code, r.text[:500])
            r.raise_for_status()
        return r.json()
