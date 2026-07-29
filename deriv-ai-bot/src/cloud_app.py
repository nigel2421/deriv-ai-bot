"""
Cloud Run HTTP entrypoint + OAuth login for new Deriv developers apps.

- Starts trading bot on startup (background)
- /health /status /ready
- /oauth/login  → Deriv OAuth authorize (PKCE)
- /oauth/callback → exchange code, store token, restart bot
"""
from __future__ import annotations

import asyncio
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
    cycle = int(os.getenv("TRADE_CYCLE_SECONDS", "45"))
    logger.info(
        "Cloud app startup mode=%s api_mode=%s app_id=%s…",
        mode,
        DERIV_API_MODE,
        str(DERIV_APP_ID)[:12],
    )
    # Retry start so transient OTP/auth blips don't leave service dead
    for attempt in range(1, 4):
        try:
            await start_bot(mode, cycle_seconds=cycle)
            if runtime.status in {"running", "starting"}:
                break
            logger.warning(
                "Bot start attempt %s status=%s err=%s",
                attempt,
                runtime.status,
                runtime.last_error,
            )
        except Exception as e:
            logger.exception("Bot start attempt %s failed: %s", attempt, e)
        if attempt < 3:
            await asyncio.sleep(5 * attempt)
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


async def control_resume(request: Request):
    """
    Clear risk cooldown + enable trading immediately (no wait for pause timer).
    GET or POST /control/resume
    """
    orch = runtime.orchestrator
    if orch is None:
        return HTMLResponse(
            "<h1>Bot not ready</h1><p>Orchestrator not started yet. "
            "<a href='/'>Home</a></p>",
            status_code=503,
        )
    st = orch.force_resume(source="cloud_run:/control/resume")
    # Also restart trading loop if process is in error state
    if runtime.status == "error":
        mode = os.getenv("MODE", "demo")
        cycle = int(os.getenv("TRADE_CYCLE_SECONDS", "45"))
        try:
            await stop_bot()
            await start_bot(mode, cycle_seconds=cycle)
        except Exception as e:
            logger.exception("Resume restart failed: %s", e)
    want_json = "application/json" in (request.headers.get("accept") or "")
    if want_json or request.query_params.get("format") == "json":
        return JSONResponse(
            {
                "ok": True,
                "action": "resume",
                "status": runtime.public_status(),
                "risk": {
                    "paused": st.get("paused"),
                    "consecutive_losses": st.get("consecutive_losses"),
                    "telegram_trading": st.get("telegram_trading"),
                },
            }
        )
    return HTMLResponse(
        f"""<!doctype html><html><body style="font-family:system-ui;background:#0b1220;color:#e8eefc;padding:2rem">
        <h1>▶️ Trading resumed</h1>
        <p>Risk cooldown cleared · loss streak reset · trading switch ON.</p>
        <p>Risk paused: <b>{st.get('paused')}</b> ·
           Consecutive losses: <b>{st.get('consecutive_losses')}</b> ·
           Telegram trading: <b>{st.get('telegram_trading')}</b></p>
        <p><a href="/" style="color:#7eb6ff">Dashboard</a> ·
           <a href="/status" style="color:#7eb6ff">/status</a></p>
        <meta http-equiv="refresh" content="3;url=/"/>
        </body></html>"""
    )


async def control_pause(request: Request):
    """Pause new trades from the dashboard. GET or POST /control/pause"""
    orch = runtime.orchestrator
    if orch is None:
        return HTMLResponse(
            "<h1>Bot not ready</h1><p><a href='/'>Home</a></p>", status_code=503
        )
    mins = int(request.query_params.get("minutes") or 60)
    st = orch.force_pause(source="cloud_run:/control/pause", minutes=mins)
    want_json = "application/json" in (request.headers.get("accept") or "")
    if want_json or request.query_params.get("format") == "json":
        return JSONResponse({"ok": True, "action": "pause", "risk": st})
    return HTMLResponse(
        f"""<!doctype html><html><body style="font-family:system-ui;background:#0b1220;color:#e8eefc;padding:2rem">
        <h1>⏸ Trading paused</h1>
        <p>No new trades for ~{mins} minutes (or until Resume).</p>
        <p><a href="/control/resume" style="color:#3ddc97">Resume now</a> ·
           <a href="/" style="color:#7eb6ff">Dashboard</a></p>
        </body></html>"""
    )


async def control_restart(request: Request):
    """Full bot reconnect (WS + loop) without waiting for cooldown."""
    mode = os.getenv("MODE", "demo")
    cycle = int(os.getenv("TRADE_CYCLE_SECONDS", "45"))
    try:
        await stop_bot()
        await start_bot(mode, cycle_seconds=cycle)
        if runtime.orchestrator:
            runtime.orchestrator.force_resume(source="cloud_run:/control/restart")
    except Exception as e:
        logger.exception("Restart failed: %s", e)
        return HTMLResponse(f"<h1>Restart failed</h1><pre>{e}</pre>", status_code=500)
    st = runtime.public_status()
    return HTMLResponse(
        f"""<!doctype html><html><body style="font-family:system-ui;background:#0b1220;color:#e8eefc;padding:2rem">
        <h1>🔄 Bot restarted</h1>
        <p>Status: <b class="ok">{st.get('status')}</b></p>
        <p><a href="/" style="color:#7eb6ff">Dashboard</a></p>
        <meta http-equiv="refresh" content="3;url=/"/>
        </body></html>"""
    )


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


def _conf_badge(level: str) -> str:
    cls = {"LOW": "badge-low", "MEDIUM": "badge-med", "HIGH": "badge-high"}.get(level, "")
    return f"<span class='badge {cls}'>{level}</span>"


def _fmt_trade_rows(trades: list, *, open_mode: bool = False) -> str:
    if not trades:
        return (
            "<tr><td colspan='9' class='muted' style='text-align:center;padding:1rem'>"
            "No trades yet — wait for a cycle or check confidence / cooldowns."
            "</td></tr>"
        )
    rows = []
    for t in trades[:20]:
        st = str(t.get("status") or ("open" if open_mode else "?"))
        st_cls = (
            "ok"
            if st in {"win", "open"}
            else (
                "bad"
                if st in {"loss", "failed", "buy_failed"}
                else ("muted" if st in {"skipped_low_payout", "push"} else "muted")
            )
        )
        profit = t.get("profit")
        profit_s = "\u2014" if profit is None else f"{float(profit):+.2f}"
        p_cls = "ok" if profit is not None and float(profit) > 0 else (
            "bad" if profit is not None and float(profit) < 0 else "muted"
        )
        conf = t.get("confidence")
        conf_s = f"{float(conf):.0%}" if conf is not None else "\u2014"
        level = str(t.get("confidence_level") or "")
        level_badge = _conf_badge(level) if level else ""
        support = t.get("historical_support")
        support_s = f"<br/><span class='muted' style='font-size:0.7rem'>{support} trades</span>" if support is not None else ""
        ev = t.get("ev")
        ev_s = f"EV {float(ev):+.3f}" if ev is not None else ""
        ev_cls = "ok" if ev is not None and float(ev) > 0.15 else ("warn" if ev is not None and float(ev) > 0 else "bad")
        barrier = t.get("barrier")
        bar_s = "\u2014" if barrier is None else str(barrier)
        dur = t.get("duration")
        du = t.get("duration_unit") or ""
        dur_s = f"{dur}{du}" if dur is not None else (t.get("horizon") or "\u2014")
        fam = t.get("family") or "\u2014"
        ts = t.get("closed_at") or t.get("opened_at") or t.get("ts") or "\u2014"
        if isinstance(ts, str) and "T" in ts:
            ts = ts.replace("T", " ")[:19]
        mor = t.get("mor_score")
        mor_s = f"MOR {mor:.0f}" if mor is not None else ""
        rows.append(
            f"<tr>"
            f"<td class='{st_cls}'><b>{st.upper()}</b></td>"
            f"<td><code>{t.get('symbol') or '\u2014'}</code></td>"
            f"<td>{t.get('contract_type') or '\u2014'}</td>"
            f"<td>{bar_s}</td>"
            f"<td>{t.get('stake') if t.get('stake') is not None else '\u2014'}</td>"
            f"<td class='{p_cls}'>{profit_s}</td>"
            f"<td>{conf_s} {level_badge}{support_s}<br/>"
            f"<span class='muted' style='font-size:0.75rem'>{fam} \u00b7 {dur_s}</span></td>"
            f"<td class='{ev_cls}' style='font-size:0.8rem'>{ev_s}<br/>"
            f"<span class='muted'>{mor_s}</span></td>"
            f"<td class='muted' style='font-size:0.8rem'>{ts}<br/>"
            f"<span class='muted'>#{t.get('contract_id') or '\u2014'}</span></td>"
            f"</tr>"
        )
    return "".join(rows)


async def root(_: Request) -> HTMLResponse:
    s = runtime.public_status()
    risk = s.get("risk") or {}
    status_cls = "ok" if s.get("status") == "running" else "bad"
    err_html = (
        f"<p class='bad'>Error: {s.get('last_error')}</p>" if s.get("last_error") else ""
    )
    strats = s.get("strategies") or {}
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
            f"mode {(snap or {}).get('barrier_mode', 'adaptive')} · "
            f"streak {mg.get('loss_streak', 0)}</li>"
        )
    strat_html = (
        "<ul>" + "".join(strat_lines) + "</ul>"
        if strat_lines
        else "<p class='muted'>Strategies load after first cycle.</p>"
    )
    symbols = ", ".join(s.get("symbols") or [])
    open_rows = _fmt_trade_rows(s.get("open_trade_details") or [], open_mode=True)
    recent_rows = _fmt_trade_rows(s.get("recent_trades") or [])
    anti = s.get("anti_spiral") or {}
    bans = anti.get("setup_bans") or {}
    ban_html = (
        ", ".join(f"<code>{k}</code> ({v}m)" for k, v in list(bans.items())[:8])
        if bans
        else "none"
    )
    oauth_hint = ""
    if not is_legacy_app_id(str(DERIV_APP_ID)) and s.get("status") == "error":
        oauth_hint = (
            "<p class='muted'>Auth error? Complete "
            "<a href='/oauth/login'>OAuth login</a> "
            "or set a PAT in <code>DERIV_API_TOKEN</code>.</p>"
        )
    pnl = risk.get("daily_pnl")
    pnl_cls = "ok" if pnl is not None and float(pnl) >= 0 else "bad"
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Deriv AI Bot</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 1.5rem auto; padding: 0 1rem;
           background: #0b1220; color: #e8eefc; }}
    .card {{ background: #151e33; border-radius: 12px; padding: 1.25rem 1.5rem; margin-bottom: 1rem;
             box-shadow: 0 4px 24px rgba(0,0,0,.25); }}
    h1 {{ margin-top: 0; font-size: 1.45rem; }}
    h2 {{ margin: 0 0 0.75rem; font-size: 1.05rem; }}
    .ok {{ color: #3ddc97; }}
    .bad {{ color: #ff6b6b; }}
    .warn {{ color: #f5c842; }}
    .muted {{ color: #9bb0d3; }}
    a {{ color: #7eb6ff; }}
    code {{ background: #0b1220; padding: 0.1rem 0.35rem; border-radius: 4px; font-size: 0.9em; }}
    ul {{ margin: 0.4rem 0 0; padding-left: 1.2rem; }}
    li {{ margin: 0.25rem 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0.75rem; }}
    .grid3 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 0.75rem; }}
    .stat {{ background: #0b1220; border-radius: 8px; padding: 0.75rem; }}
    .stat .label {{ font-size: 0.75rem; color: #9bb0d3; }}
    .stat .val {{ font-size: 1.15rem; font-weight: 650; margin-top: 0.2rem; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
    th {{ text-align: left; color: #9bb0d3; font-weight: 600; padding: 0.45rem 0.35rem;
         border-bottom: 1px solid #243049; }}
    td {{ padding: 0.5rem 0.35rem; border-bottom: 1px solid #1a2438; vertical-align: top; }}
    tr:hover td {{ background: rgba(126,182,255,.06); }}
    .btnrow {{ display:flex; flex-wrap:wrap; gap:0.65rem; align-items:center; }}
    .btn {{ color:#fff !important; padding:0.55rem 1rem; border-radius:8px; text-decoration:none; font-weight:600; }}
    .btn-go {{ background:#1f6f4a; }}
    .btn-stop {{ background:#6b2d2d; }}
    .btn-blue {{ background:#2a3f6b; }}
    .badge {{ display:inline-block; padding:0.15rem 0.5rem; border-radius:999px; font-size:0.7rem; font-weight:700; }}
    .badge-low {{ background:#6b2d2d; color:#ffb0b0; }}
    .badge-med {{ background:#5a4200; color:#f5c842; }}
    .badge-high {{ background:#1f6f4a; color:#a0ffcb; }}
    .badge-block {{ background:#6b2d2d; color:#ff8080; }}
    .badge-warn {{ background:#5a4200; color:#f5c842; }}
    .badge-watch {{ background:#2a3f6b; color:#7eb6ff; }}
    .badge-healthy {{ background:#1f6f4a; color:#a0ffcb; }}
    .panel-pair {{ display:grid; grid-template-columns:1fr 1fr; gap:0.75rem; }}
    @media(max-width:700px) {{ .panel-pair {{ grid-template-columns:1fr; }} }}
  </style>
  <meta http-equiv="refresh" content="15"/>
</head>
<body>
  <div class="card">
    <h1>Deriv AI Bot</h1>
    <p class="muted">Auto-refresh 15s · API <code>{DERIV_API_MODE}</code>
       · stake <code>{s.get('stake_mode') or 'flat'}</code>
       · minutes <code>{'on' if s.get('enable_minute') else 'off'} ({s.get('minute_duration') or 2}m)</code>
    </p>
    <div class="grid">
      <div class="stat"><div class="label">Status</div>
        <div class="val {status_cls}">{s.get('status')}</div></div>
      <div class="stat"><div class="label">Balance</div>
        <div class="val">{risk.get('balance')} {risk.get('currency') or ''}</div></div>
      <div class="stat"><div class="label">Daily PnL</div>
        <div class="val {pnl_cls}">{risk.get('daily_pnl')}</div></div>
      <div class="stat"><div class="label">Open</div>
        <div class="val">{risk.get('open_trades')}</div></div>
      <div class="stat"><div class="label">Trades today</div>
        <div class="val">{risk.get('trades_today')}</div></div>
      <div class="stat"><div class="label">Risk paused</div>
        <div class="val {'bad' if risk.get('paused') else 'ok'}">{risk.get('paused')}
        {(' · ' + str(risk.get('pause_remaining_min')) + 'm left') if risk.get('paused') and risk.get('pause_remaining_min') is not None else ''}
        </div>
        <div class="muted" style="font-size:0.75rem">{risk.get('pause_reason') or ''} · auto-resumes: {risk.get('auto_resume_count', 0)}</div>
      </div>
    </div>
    <p style="margin-top:1rem" class="muted">Markets: <code>{symbols}</code></p>
    <p class="muted">Started: {s.get('started_at') or '—'} · Last cycle: {s.get('last_cycle_at') or '—'}</p>
    <p class="muted">Setup bans: {ban_html}</p>
    {err_html}
    {oauth_hint}
  </div>

  <div class="card">
    <h2>Open trades</h2>
    <div style="overflow-x:auto">
    <table>
      <thead><tr>
        <th>Status</th><th>Symbol</th><th>Type</th><th>Barrier</th>
        <th>Stake</th><th>PnL</th><th>Conf / family</th><th>Time / id</th>
      </tr></thead>
      <tbody>{open_rows}</tbody>
    </table>
    </div>
  </div>

  <div class="card">
    <h2>Recent trades</h2>
    <p class="muted">Last 20 placed / closed (win, loss, open, failed).</p>
    <div style="overflow-x:auto">
    <table>
      <thead><tr>
        <th>Status</th><th>Symbol</th><th>Type</th><th>Barrier</th>
        <th>Stake</th><th>PnL</th><th>Conf / family</th><th>Time / id</th>
      </tr></thead>
      <tbody>{recent_rows}</tbody>
    </table>
    </div>
  </div>

  <div class="card">
    <h2>&#128202; Probability Engine &amp; HPP</h2>
    <p class="muted">Confidence level = LOW (&lt;30 trades) / MEDIUM (30-99) / HIGH (&ge;100). Pattern decay: Watch -10 / Warning -15 / Block &lt;-20 + clarity &lt;75%.</p>
    {_fmt_probability_panel(s)}
  </div>

  <div class="card panel-pair" style="display:grid;grid-template-columns:1fr 1fr;gap:0.75rem">
    <div>
      <h2>&#8651; Transition Matrix</h2>
      <p class="muted">Rise/Fall direction persistence. &gt;58% = persistent market.</p>
      {_fmt_transition_panel(s)}
    </div>
    <div>
      <h2>&#128203; Correlation Filter</h2>
      <p class="muted">Within R_* and 1HZ* groups, only highest-EV passes.</p>
      {_fmt_correlation_panel(s)}
    </div>
  </div>

  <div class="card">
    <h2>&#128200; Market Opportunity Ranking</h2>
    <p class="muted">Score 0-100 (normalized). Velocity = current vs 24h-ago avg. MOR90+ WR validates scoring.</p>
    {_fmt_mor_panel(s)}
  </div>

  <div class="card panel-pair" style="display:grid;grid-template-columns:1fr 1fr;gap:0.75rem">
    <div>
      <h2>&#127919; Calibration</h2>
      <p class="muted">Phase 1: display + alert only. Phase 2 auto-deflation after &gt;1000 trades, error &gt;15%, 3 consecutive audits.</p>
      {_fmt_calibration_panel(s)}
    </div>
    <div>
      <h2>&#129302; AI Auditor</h2>
      <p class="muted">Persistent cumulative closes across restarts. Every 100: standard audit. Every 1000: deep audit.</p>
      {_fmt_auditor_panel(s)}
    </div>
  </div>

  <div class="card">
    <h2>&#129504; DeepSeek AI Advisor</h2>
    <p class="muted">Per-market deep analysis triggers every 100 closed trades per symbol. Reads full GCS trade history for accuracy.</p>
    {_fmt_deepseek_panel(s)}
  </div>

  <div class="card">
    <h2>Controls</h2>
    <div class="btnrow">
      <a class="btn btn-go" href="/control/resume">▶️ Resume</a>
      <a class="btn btn-stop" href="/control/pause">⏸ Pause</a>
      <a class="btn btn-blue" href="/control/restart">🔄 Restart</a>
    </div>
    <p class="muted" style="margin-top:0.75rem">Resume clears risk cooldown + anti-spiral bans.</p>
  </div>

  <div class="card">
    <h2>Strategy markets</h2>
    <p class="muted">Ticks: digits + short CALL/PUT · Minutes: candle EMA/RSI CALL/PUT · conf &ge; 80%</p>
    {strat_html}
  </div>

  <div class="card">
    <p>
      <a href="/status">JSON /status</a> &middot;
      <a href="/health">/health</a> &middot;
      <a href="/oauth/login">OAuth</a>
    </p>
  </div>
</body>
</html>"""
    return HTMLResponse(html)


def _fmt_deepseek_panel(s: dict) -> str:
    """DeepSeek per-market AI advisor panel."""
    ds = s.get("deepseek") or {}
    if not ds.get("enabled"):
        return "<p class='muted'>DeepSeek disabled. Set <code>DEEPSEEK_ENABLED=true</code> and <code>DEEPSEEK_API_KEY</code> in Cloud Run secrets.</p>"

    sym_status = ds.get("symbol_status") or {}
    latest = ds.get("latest_report")
    model = ds.get("model", "?")
    every = ds.get("analyze_every", 100)

    sym_rows = []
    for sym, info in sorted(sym_status.items()):
        until = info.get("closes_until_next", every)
        done = every - until
        pct = min(100, int(done / max(every, 1) * 100))
        health = info.get("health") or ""
        health_cls = {"HEALTHY": "ok", "WATCH": "warn", "STRUGGLING": "warn", "BAN": "bad"}.get(health, "muted")
        health_s = f"<span class='{health_cls}'>{health}</span>" if health else "<span class='muted'>pending</span>"
        last = (info.get("last_analysis") or "")[:16].replace("T", " ")
        sym_rows.append(
            f"<tr><td><code>{sym}</code></td>"
            f"<td style='min-width:120px'>"
            f"<div style='background:#0b1220;border-radius:4px;height:8px;overflow:hidden'>"
            f"<div style='background:#7eb6ff;width:{pct}%;height:100%'></div></div>"
            f"<span class='muted' style='font-size:0.72rem'>{done}/{every}</span></td>"
            f"<td>{health_s}</td>"
            f"<td class='muted' style='font-size:0.8rem'>{last or '—'}</td></tr>"
        )

    table_html = ""
    if sym_rows:
        table_html = (
            "<table><thead><tr><th>Symbol</th><th>Progress to next</th>"
            "<th>Health</th><th>Last run</th></tr></thead>"
            f"<tbody>{''.join(sym_rows)}</tbody></table>"
        )

    rec_html = ""
    if latest:
        rec = latest.get("recommendation") or {}
        sym_name = latest.get("symbol", "?")
        gen = str(latest.get("generated_at", ""))[:19].replace("T", " ")
        n = latest.get("trades_analyzed", 0)
        health = rec.get("health", "?")
        health_cls = {"HEALTHY": "ok", "WATCH": "warn", "STRUGGLING": "warn", "BAN": "bad"}.get(health, "muted")
        summary = rec.get("summary", "")
        hints = rec.get("learning_hints") or []
        bans = rec.get("ban_setups") or []
        boosts = rec.get("boost_setups") or []
        conf_rec = rec.get("confidence_recommendation") or {}
        ct_recs = rec.get("contract_recommendations") or []

        ct_rows = "".join(
            f"<tr><td><code>{r.get('contract_type')}</code></td>"
            f"<td><span class='{'ok' if r.get('action') in ('KEEP','BOOST') else 'bad'}'>{r.get('action')}</span></td>"
            f"<td class='muted' style='font-size:0.82rem'>{r.get('reason','')[:80]}</td></tr>"
            for r in ct_recs[:8]
        )

        rec_html = f"""
        <hr style='border-color:#243049;margin:1rem 0'/>
        <p style='margin:0 0 0.5rem'><b>Latest: {sym_name}</b> &middot;
        <span class='{health_cls}'>{health}</span> &middot;
        {n} trades &middot; <span class='muted'>{gen}</span></p>
        <p class='muted' style='font-size:0.9rem'>{summary}</p>
        {'<table><thead><tr><th>Contract</th><th>Action</th><th>Reason</th></tr></thead><tbody>' + ct_rows + '</tbody></table>' if ct_rows else ''}
        <div class='grid3' style='margin-top:0.75rem'>
          <div><b class='ok'>&#128161; Hints</b><ul>{''.join(f'<li class=muted style=font-size:0.85rem>{h}</li>' for h in hints[:5]) or '<li class=muted>none</li>'}</ul></div>
          <div><b class='bad'>&#9940; Bans</b><ul>{''.join(f'<li class=bad style=font-size:0.85rem><code>{b}</code></li>' for b in bans[:6]) or '<li class=muted>none</li>'}</ul></div>
          <div><b class='ok'>&#128640; Boosts</b><ul>{''.join(f'<li class=ok style=font-size:0.85rem><code>{b}</code></li>' for b in boosts[:6]) or '<li class=muted>none</li>'}</ul></div>
        </div>
        {'<p class=muted style=font-size:0.85rem>Confidence: ' + conf_rec.get('action','') + ' → ' + str(conf_rec.get('suggested_threshold','')) + ' — ' + conf_rec.get('reason','')[:100] + '</p>' if conf_rec else ''}
        """

    return (
        f"<p class='muted' style='margin:0 0 0.75rem'>Model: <code>{model}</code> · Trigger: every <code>{every}</code> closes per symbol · Analyzed: <code>{ds.get('total_symbols_analyzed', 0)}</code> markets</p>"
        + table_html
        + rec_html
    )


def _fmt_probability_panel(s: dict) -> str:
    """Rec #1 & #2: Probability Engine with confidence levels and HPP per contract."""
    learning = s.get("learning") or {}
    top = learning.get("top") or []
    if not top:
        return "<p class='muted'>No HPP data yet — waiting for first trades.</p>"
    rows = []
    for entry in top[:8]:
        key = entry.get("key", "")
        level = entry.get("confidence_level", "?")
        support = entry.get("historical_support", 0)
        decay = entry.get("decay_status", "")
        wins = entry.get("wins", 0)
        losses = entry.get("losses", 0)
        total = (wins or 0) + (losses or 0)
        wr = round(wins / total * 100, 1) if total > 0 else 0
        pnl = entry.get("pnl", 0)
        level_cls = {"LOW": "badge-low", "MEDIUM": "badge-med", "HIGH": "badge-high"}.get(level, "")
        decay_cls = "badge-block" if "Block" in str(decay) else ("badge-warn" if "Warning" in str(decay) else ("badge-watch" if "Watch" in str(decay) else "badge-healthy"))
        rows.append(
            f"<tr><td><code>{key}</code></td>"
            f"<td><span class='badge {level_cls}'>{level}</span></td>"
            f"<td>{support} trades</td>"
            f"<td>{'ok' if wr >= 55 else 'bad' and 'bad'}"  # unused cls
            f"<span class='{'ok' if wr >= 55 else 'bad'}'>{wr}%</span></td>"
            f"<td class='{'ok' if pnl >= 0 else 'bad'}'>{pnl:+.2f}</td>"
            f"<td><span class='badge {decay_cls}' style='font-size:0.65rem'>{decay[:20] if decay else '—'}</span></td>"
            f"</tr>"
        )
    return (
        "<table><thead><tr><th>Setup</th><th>Confidence Level</th><th>Support</th>"
        "<th>Win Rate</th><th>PnL</th><th>Decay</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _fmt_transition_panel(s: dict) -> str:
    """Rec #3: Transition Matrix."""
    tm = s.get("transition_matrix") or {}
    if not tm:
        return "<p class='muted'>No rise/fall trades yet — transition matrix populates after CALL/PUT settlements.</p>"
    rows = []
    for sym, d in sorted(tm.items()):
        total = d.get("total", 0)
        if total == 0:
            continue
        persist = d.get("persistence_pct", 50)
        p_cls = "ok" if persist > 58 else ("bad" if persist < 45 else "muted")
        arrow = "\u2191" if persist > 58 else ("\u2193" if persist < 45 else "\u2192")
        suf = "insufficient" if not d.get("sufficient_data") else ""
        rows.append(
            f"<tr><td><code>{sym}</code></td>"
            f"<td>{d.get('UP_UP_pct', 0):.1f}%</td>"
            f"<td>{d.get('UP_DOWN_pct', 0):.1f}%</td>"
            f"<td>{d.get('DOWN_UP_pct', 0):.1f}%</td>"
            f"<td>{d.get('DOWN_DOWN_pct', 0):.1f}%</td>"
            f"<td class='{p_cls}'>{persist:.1f}% {arrow}</td>"
            f"<td class='muted' style='font-size:0.75rem'>{total} trades {suf}</td>"
            f"</tr>"
        )
    if not rows:
        return "<p class='muted'>No transitions recorded yet.</p>"
    return (
        "<table><thead><tr><th>Symbol</th><th>UP&#8594;UP</th><th>UP&#8594;DOWN</th>"
        "<th>DOWN&#8594;UP</th><th>DOWN&#8594;DOWN</th><th>Persistence</th><th>N</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _fmt_mor_panel(s: dict) -> str:
    """Rec #4 & #5: Market Opportunity Ranking with velocity and outcome validation."""
    mor = s.get("mor") or {}
    ranked = (mor.get("ranked") or [])
    if not ranked:
        return "<p class='muted'>MOR data populates once signals are generated each cycle.</p>"
    rows = []
    for i, r in enumerate(ranked[:10], 1):
        vel = r.get("velocity", 0)
        vel_s = f"{vel:+.1f}" if vel is not None else "\u2014"
        vel_cls = "ok" if vel and vel > 10 else ("bad" if vel and vel < -10 else "muted")
        arrow = r.get("velocity_arrow", "\u2192")
        hi_wr = r.get("high_mor_wr")
        hi_wr_s = f"{hi_wr:.1f}%" if hi_wr is not None else "\u2014"
        rows.append(
            f"<tr><td class='muted'>{i}</td>"
            f"<td><code>{r.get('symbol')}</code></td>"
            f"<td><b>{r.get('score', 0):.1f}</b></td>"
            f"<td class='muted'>{r.get('yesterday') or '\u2014'}</td>"
            f"<td class='{vel_cls}'>{vel_s} {arrow}</td>"
            f"<td>{hi_wr_s}</td>"
            f"<td class='muted'>{r.get('total_outcomes', 0)}</td>"
            f"</tr>"
        )
    bucket_html = ""
    buckets = (mor.get("bucket_analysis") or {})
    if buckets:
        brows = []
        for b in ["90+", "80-89", "70-79", "<70"]:
            d = buckets.get(b, {})
            wr = d.get("win_rate")
            n = d.get("n", 0)
            wr_s = f"{wr}%" if wr is not None else "\u2014 (need more trades)"
            wr_cls = "ok" if wr and wr >= 60 else ("bad" if wr and wr < 50 else "muted")
            brows.append(
                f"<tr><td>MOR {b}</td>"
                f"<td class='{wr_cls}'>{wr_s}</td>"
                f"<td class='muted'>{n} trades</td></tr>"
            )
        bucket_html = (
            "<br/><p style='margin:0.75rem 0 0.4rem'><b>MOR Validation</b></p>"
            f"<table><thead><tr><th>Bucket</th><th>Win Rate</th><th>N</th></tr></thead>"
            f"<tbody>{''.join(brows)}</tbody></table>"
        )
    return (
        "<table><thead><tr><th>#</th><th>Symbol</th><th>Score</th><th>Yesterday</th>"
        "<th>Velocity</th><th>MOR90+ WR</th><th>Trades</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        f"{bucket_html}"
    )


def _fmt_calibration_panel(s: dict) -> str:
    """Rec #8: Calibration tracking."""
    cal = s.get("calibration") or {}
    rows_data = cal.get("rows") or []
    overall = cal.get("overall_error")
    cum = cal.get("cumulative_trades", 0)
    auto = cal.get("auto_deflation_enabled", False)
    overall_cls = "bad" if overall and overall > 0.15 else ("warn" if overall and overall > 0.08 else "ok")
    overall_s = f"{overall*100:.1f}%" if overall is not None else "insufficient data"
    deflation_s = "<span class='ok'>Phase 2 ACTIVE</span>" if auto else "<span class='muted'>Phase 1 (display only)</span>"
    rows = []
    for r in rows_data:
        status = r.get("status", "")
        sc = r.get("status_code", "")
        st_cls = (
            "ok" if sc == "good"
            else ("bad" if sc in ("overconfident", "severely_overconfident", "underconfident")
                  else ("warn" if sc == "watch" else "muted"))
        )
        pred = r.get("predicted_avg")
        actual = r.get("actual_wr")
        err = r.get("error")
        rows.append(
            f"<tr><td>{r.get('bucket')}</td>"
            f"<td>{f'{pred:.1f}%' if pred is not None else '\u2014'}</td>"
            f"<td>{f'{actual:.1f}%' if actual is not None else '\u2014'}</td>"
            f"<td>{f'{err:+.1f}%' if err is not None else '\u2014'}</td>"
            f"<td>{r.get('n', 0)}</td>"
            f"<td class='{st_cls}'>{status}</td>"
            f"</tr>"
        )
    return (
        f"<p style='margin:0 0 0.5rem'>"
        f"Overall Error: <span class='{overall_cls}'><b>{overall_s}</b></span> · "
        f"{cum} trades · {deflation_s}</p>"
        "<table><thead><tr><th>Bucket</th><th>Predicted</th><th>Actual WR</th>"
        "<th>Error</th><th>N</th><th>Status</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _fmt_auditor_panel(s: dict) -> str:
    """Rec #10: AI Auditor report."""
    report = s.get("ai_auditor")
    if not report:
        return "<p class='muted'>First audit runs after 100 closed trades.</p>"
    atype = report.get("type", "minor").upper()
    generated = str(report.get("generated_at", ""))[:19]
    total_trades = report.get("trades_analyzed", 0)
    wr = report.get("overall_win_rate", 0)
    wr_cls = "ok" if wr >= 55 else ("bad" if wr < 45 else "muted")
    helping = report.get("helping") or []
    hurting = report.get("hurting") or []
    recs = report.get("recommendations") or []
    help_html = "".join(
        f"<li><code>{h['feature']}</code> <span class='ok'>{h['contribution']}</span></li>"
        for h in helping[:5]
    ) or "<li class='muted'>None identified yet</li>"
    hurt_html = "".join(
        f"<li><code>{h['feature']}</code> <span class='bad'>{h['contribution']}</span></li>"
        for h in hurting[:5]
    ) or "<li class='muted'>None identified yet</li>"
    rec_html = "".join(
        f"<li class='muted' style='font-size:0.85rem'>{r}</li>" for r in recs[:5]
    )
    return (
        f"<p style='margin:0 0 0.5rem' class='muted'>{atype} audit · {generated} · "
        f"{total_trades} trades · WR <span class='{wr_cls}'>{wr}%</span></p>"
        f"<div class='panel-pair'>"
        f"<div><b class='ok'>\U0001f7e2 Helping</b><ul>{help_html}</ul></div>"
        f"<div><b class='bad'>\U0001f534 Hurting</b><ul>{hurt_html}</ul></div>"
        f"</div>"
        f"<br/><b>&#128161; Recommendations</b><ul>{rec_html}</ul>"
    )


def _fmt_correlation_panel(s: dict) -> str:
    """Rec #7: Correlation filter status."""
    corr = s.get("correlation") or {}
    groups = corr.get("groups") or []
    if not groups:
        return "<p class='muted'>No correlated signals this cycle (or no signals).</p>"
    rows = []
    for g in groups:
        blocked = g.get("blocked") or []
        rows.append(
            f"<tr><td><code>{g.get('group')}</code></td>"
            f"<td>{g.get('signals', 0)}</td>"
            f"<td class='ok'>{g.get('selected', '\u2014')}</td>"
            f"<td class='bad' style='font-size:0.8rem'>{', '.join(blocked) if blocked else '\u2014'}</td>"
            f"</tr>"
        )
    return (
        "<table><thead><tr><th>Correlation Group</th><th>Signals</th>"
        "<th>Selected</th><th>Blocked</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


routes = [
    Route("/", root),
    Route("/health", health),
    Route("/ready", ready),
    Route("/status", status),
    Route("/control/resume", control_resume, methods=["GET", "POST"]),
    Route("/control/pause", control_pause, methods=["GET", "POST"]),
    Route("/control/restart", control_restart, methods=["GET", "POST"]),
    Route("/oauth/login", oauth_login),
    Route("/oauth/callback", oauth_callback),
    Route("/oauth/logout", oauth_logout),
]

app = Starlette(
    debug=os.getenv("DEBUG", "").lower() in {"1", "true"},
    routes=routes,
    lifespan=lifespan,
)
