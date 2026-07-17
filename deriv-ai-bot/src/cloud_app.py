"""
Cloud Run HTTP entrypoint + OAuth login for new Deriv developers apps.

- Starts trading bot on startup (background)
- /health /status /ready
- /oauth/login  → Deriv OAuth authorize (PKCE)
- /oauth/callback → exchange code, store token, restart bot
"""
from __future__ import annotations

import os
import secrets
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from starlette.routing import Route

from config.settings import (
    DERIV_API_MODE,
    DERIV_APP_ID,
    DERIV_OAUTH_AUTH_URL,
    DERIV_OAUTH_CLIENT_ID,
    DERIV_OAUTH_CLIENT_SECRET,
    DERIV_OAUTH_REDIRECT_URI,
    DERIV_OAUTH_TOKEN_URL,
)
from src.api.deriv_v2_auth import (
    build_oauth_authorize_url,
    exchange_oauth_code,
    generate_pkce_pair,
    is_legacy_app_id,
)
from src.api.token_store import clear_token, load_access_token, save_token_payload
from src.bot_runtime import runtime, start_bot, stop_bot
from src.utils.logger import setup_logger

logger = setup_logger()

# In-memory PKCE state (single instance Cloud Run)
_oauth_sessions: Dict[str, Dict[str, str]] = {}


@asynccontextmanager
async def lifespan(app: Starlette):
    mode = os.getenv("MODE", "demo")
    cycle = int(os.getenv("TRADE_CYCLE_SECONDS", "60"))
    logger.info(
        "Cloud app startup mode=%s api_mode=%s app_id=%s…",
        mode,
        DERIV_API_MODE,
        str(DERIV_APP_ID)[:12],
    )
    await start_bot(mode, cycle_seconds=cycle)
    try:
        yield
    finally:
        logger.info("Cloud app shutdown — stopping bot")
        await stop_bot()


async def health(_: Request) -> PlainTextResponse:
    st = runtime.status
    if st in {"running", "starting"}:
        return PlainTextResponse("ok", status_code=200)
    if st == "error":
        return PlainTextResponse(
            f"error:{runtime.last_error or 'unknown'}", status_code=200
        )
    return PlainTextResponse(st or "unknown", status_code=200)


async def ready(_: Request) -> JSONResponse:
    body = runtime.public_status()
    code = 200 if runtime.status == "running" else 503
    return JSONResponse(body, status_code=code)


async def status(_: Request) -> JSONResponse:
    body = runtime.public_status()
    body["api_mode"] = DERIV_API_MODE
    body["app_id_prefix"] = str(DERIV_APP_ID)[:8]
    body["legacy_app_id"] = is_legacy_app_id(str(DERIV_APP_ID))
    body["has_stored_oauth_token"] = bool(load_access_token())
    return JSONResponse(body)


def _public_base(request: Request) -> str:
    # Prefer configured redirect origin; else request base
    if DERIV_OAUTH_REDIRECT_URI:
        p = urlparse(DERIV_OAUTH_REDIRECT_URI)
        if p.scheme and p.netloc:
            return f"{p.scheme}://{p.netloc}"
    return str(request.base_url).rstrip("/")


async def oauth_login(request: Request) -> RedirectResponse:
    """Start OAuth2 Authorization Code + PKCE flow."""
    client_id = DERIV_OAUTH_CLIENT_ID or DERIV_APP_ID
    redirect_uri = DERIV_OAUTH_REDIRECT_URI
    if not redirect_uri:
        redirect_uri = f"{_public_base(request)}/oauth/callback"

    verifier, challenge = generate_pkce_pair()
    state = secrets.token_urlsafe(24)
    _oauth_sessions[state] = {
        "verifier": verifier,
        "redirect_uri": redirect_uri,
    }
    url = build_oauth_authorize_url(
        str(client_id),
        redirect_uri,
        code_challenge=challenge,
        state=state,
        auth_url=DERIV_OAUTH_AUTH_URL,
    )
    logger.info("OAuth login redirect (client_id=%s…)", str(client_id)[:8])
    return RedirectResponse(url, status_code=302)


async def oauth_callback(request: Request):
    """Handle OAuth redirect, exchange code, store token, restart bot."""
    params = request.query_params
    err = params.get("error")
    if err:
        return HTMLResponse(
            f"<h1>OAuth error</h1><pre>{err}: {params.get('error_description')}</pre>"
            f"<p><a href='/'>Home</a></p>",
            status_code=400,
        )
    code = params.get("code")
    state = params.get("state")
    if not code or not state or state not in _oauth_sessions:
        return HTMLResponse(
            "<h1>Invalid OAuth callback</h1><p>Missing code/state. "
            "<a href='/oauth/login'>Try login again</a></p>",
            status_code=400,
        )
    sess = _oauth_sessions.pop(state)
    try:
        token_payload = await exchange_oauth_code(
            str(DERIV_OAUTH_CLIENT_ID or DERIV_APP_ID),
            code,
            sess["redirect_uri"],
            sess["verifier"],
            token_url=DERIV_OAUTH_TOKEN_URL,
            client_secret=DERIV_OAUTH_CLIENT_SECRET,
        )
        if not token_payload.get("access_token"):
            raise RuntimeError(f"No access_token in response: {token_payload.keys()}")
        save_token_payload(token_payload)
        # Restart bot with new token
        await stop_bot()
        mode = os.getenv("MODE", "demo")
        cycle = int(os.getenv("TRADE_CYCLE_SECONDS", "60"))
        await start_bot(mode, cycle_seconds=cycle)
        st = runtime.public_status()
        return HTMLResponse(
            f"""<!doctype html><html><body style="font-family:system-ui;background:#0b1220;color:#e8eefc;padding:2rem">
            <h1>Login complete</h1>
            <p>Bot status: <b>{st.get('status')}</b></p>
            <p>Error: {st.get('last_error') or 'none'}</p>
            <p><a href="/" style="color:#7eb6ff">Open dashboard</a> ·
               <a href="/status" style="color:#7eb6ff">/status</a></p>
            </body></html>"""
        )
    except Exception as e:
        logger.exception("OAuth callback failed: %s", e)
        return HTMLResponse(
            f"<h1>Token exchange failed</h1><pre>{e}</pre>"
            f"<p><a href='/oauth/login'>Retry</a></p>",
            status_code=500,
        )


async def oauth_logout(_: Request) -> HTMLResponse:
    clear_token()
    await stop_bot()
    return HTMLResponse(
        "<h1>Logged out</h1><p>Cleared stored OAuth token.</p>"
        "<p><a href='/oauth/login'>Login again</a></p>"
    )


async def root(_: Request) -> HTMLResponse:
    s = runtime.public_status()
    risk = s.get("risk") or {}
    status_cls = "ok" if s.get("status") == "running" else "bad"
    err_html = (
        f"<p class='bad'>Error: {s.get('last_error')}</p>" if s.get("last_error") else ""
    )
    strats = risk.get("strategies") or {}
    if not strats and runtime.orchestrator is not None:
        try:
            strats = runtime.orchestrator.strategy_engine.snapshots()
        except Exception:
            strats = {}
    strat_lines = []
    for sym, snap in list(strats.items())[:12]:
        mg = (snap or {}).get("martingale") or {}
        strat_lines.append(
            f"<li><code>{sym}</code> · {(snap or {}).get('type')} · "
            f"next stake { (snap or {}).get('next_stake') } · "
            f"OVER@{(snap or {}).get('over_barrier', 6)} "
            f"UNDER@{(snap or {}).get('under_barrier', 4)} · "
            f"streak {mg.get('loss_streak', 0)}</li>"
        )
    strat_html = (
        "<ul>" + "".join(strat_lines) + "</ul>"
        if strat_lines
        else "<p class='muted'>Strategies load after first cycle.</p>"
    )
    symbols = ", ".join(s.get("symbols") or [])
    oauth_hint = ""
    if not is_legacy_app_id(str(DERIV_APP_ID)) and s.get("status") == "error":
        oauth_hint = (
            "<p class='muted'>Auth error? Complete "
            "<a href='/oauth/login'>OAuth login</a> "
            "or set a PAT in <code>DERIV_API_TOKEN</code>.</p>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Deriv AI Bot</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 760px; margin: 2rem auto; padding: 0 1rem;
           background: #0b1220; color: #e8eefc; }}
    .card {{ background: #151e33; border-radius: 12px; padding: 1.25rem 1.5rem; margin-bottom: 1rem; }}
    h1 {{ margin-top: 0; }}
    .ok {{ color: #3ddc97; }}
    .bad {{ color: #ff6b6b; }}
    .muted {{ color: #9bb0d3; }}
    a {{ color: #7eb6ff; }}
    code {{ background: #0b1220; padding: 0.1rem 0.35rem; border-radius: 4px; }}
    ul {{ margin: 0.4rem 0 0; padding-left: 1.2rem; }}
    li {{ margin: 0.25rem 0; }}
  </style>
  <meta http-equiv="refresh" content="30"/>
</head>
<body>
  <div class="card">
    <h1>Deriv AI Bot</h1>
    <p class="muted">Live status (auto-refresh 30s) · API mode: <code>{DERIV_API_MODE}</code></p>
    <p>Status:
      <strong class="{status_cls}">{s.get('status')}</strong>
      · Mode: <code>{s.get('mode')}</code>
    </p>
    <p>Balance: <strong>{risk.get('balance')}</strong> {risk.get('currency') or ''}
       · Open: {risk.get('open_trades')}
       · Daily PnL: {risk.get('daily_pnl')}
    </p>
    <p>Trading enabled: {risk.get('telegram_trading')}
       · Execute trades: {risk.get('execute_trades')}
       · Risk paused: {risk.get('paused')}
    </p>
    <p class="muted">Markets: <code>{symbols}</code></p>
    <p class="muted">Started: {s.get('started_at') or '—'}
       · Last cycle: {s.get('last_cycle_at') or '—'}
    </p>
    <p class="muted">Buffers: {s.get('buffer_sizes')}</p>
    {err_html}
    {oauth_hint}
  </div>
  <div class="card">
    <h2 style="margin-top:0;font-size:1.1rem">Strategy (digits OVER 6 / UNDER 4 + martingale)</h2>
    <p class="muted">Win sets: OVER@6 → digits 7–9 · UNDER@4 → digits 0–3.
       Risk: max daily loss / consecutive losses / stake % / max open trades.</p>
    {strat_html}
  </div>
  <div class="card">
    <p>
      <a href="/oauth/login">OAuth Login</a> ·
      <a href="/oauth/logout">Logout</a> ·
      <a href="/status">/status</a> ·
      <a href="/health">/health</a>
    </p>
    <p class="muted">How to verify trades: balance drops after buy · open &gt; 0 while live ·
      daily PnL moves · Cloud Run logs show Proposal ok / Buy ok · Telegram trade alerts.</p>
  </div>
</body>
</html>"""
    return HTMLResponse(html)


routes = [
    Route("/", root),
    Route("/health", health),
    Route("/ready", ready),
    Route("/status", status),
    Route("/oauth/login", oauth_login),
    Route("/oauth/callback", oauth_callback),
    Route("/oauth/logout", oauth_logout),
]

app = Starlette(
    debug=os.getenv("DEBUG", "").lower() in {"1", "true"},
    routes=routes,
    lifespan=lifespan,
)
