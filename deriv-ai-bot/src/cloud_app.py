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
from starlette.routing import Route, WebSocketRoute

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


def _svg_circle_gauge(percent: float, title: str, subtitle: str, color_hex: str = "#38bdf8") -> str:
    """Renders a high-density circular SVG gauge for CPU/RAM/Disk metrics."""
    pct = max(0.0, min(100.0, float(percent)))
    r = 24
    c = 2 * 3.14159265 * r  # circumference ~ 150.8
    offset = c * (1.0 - pct / 100.0)
    return f"""
    <div style="display:flex;align-items:center;gap:0.75rem;background:#090d16;border:1px solid rgba(56,189,248,0.12);border-radius:10px;padding:0.65rem 0.85rem">
      <div style="position:relative;width:58px;height:58px;display:flex;align-items:center;justify-content:center;flex-shrink:0">
        <svg width="58" height="58" viewBox="0 0 58 58">
          <circle cx="29" cy="29" r="{r}" stroke="rgba(255,255,255,0.06)" stroke-width="5" fill="none"/>
          <circle cx="29" cy="29" r="{r}" stroke="{color_hex}" stroke-width="5" stroke-linecap="round" fill="none"
                  stroke-dasharray="{c:.2f}" stroke-dashoffset="{offset:.2f}"
                  transform="rotate(-90 29 29)" style="transition: stroke-dashoffset 0.6s ease"/>
        </svg>
        <div style="position:absolute;font-family:'JetBrains Mono',monospace;font-size:0.78rem;font-weight:700;color:#f8fafc">
          {pct:.0f}%
        </div>
      </div>
      <div>
        <div style="font-size:0.72rem;color:#94a3b8;font-weight:500;text-transform:uppercase;letter-spacing:0.03em">{title}</div>
        <div style="font-size:0.92rem;font-weight:700;color:#f8fafc;font-family:'JetBrains Mono',monospace;margin-top:0.1rem">{pct:.1f}%</div>
        <div style="font-size:0.65rem;color:#64748b;margin-top:0.05rem">{subtitle}</div>
      </div>
    </div>
    """


def _svg_sparkline(pnl_val: float) -> str:
    """Renders a micro SVG sparkline for PnL trajectory."""
    pnl = float(pnl_val or 0.0)
    is_pos = pnl >= 0
    stroke_color = "#34d399" if is_pos else "#f43f5e"
    fill_color = "rgba(52, 211, 153, 0.12)" if is_pos else "rgba(244, 63, 94, 0.12)"

    if is_pos:
        points = "0,22 15,18 30,20 45,12 60,15 75,6 90,10 105,3 120,4"
        fill_points = "0,22 15,18 30,20 45,12 60,15 75,6 90,10 105,3 120,4 120,26 0,26"
    else:
        points = "0,4 15,8 30,6 45,16 60,14 75,20 90,18 105,24 120,25"
        fill_points = "0,4 15,8 30,6 45,16 60,14 75,20 90,18 105,24 120,25 120,26 0,26"

    return f"""
    <svg width="100%" height="24" viewBox="0 0 120 26" preserveAspectRatio="none" style="overflow:visible">
      <polygon points="{fill_points}" fill="{fill_color}" />
      <polyline points="{points}" fill="none" stroke="{stroke_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
    </svg>
    """


def _weight_meter(weight_str: str) -> str:
    """Renders visual progress bar meter for agent weight."""
    try:
        w_val = float(str(weight_str).replace("x", "").replace("Veto", "").strip())
        pct = min(100, int((w_val / 1.5) * 100))
    except Exception:
        pct = 70
    color = "#38bdf8" if pct < 80 else "#fbbf24"
    return f"""
    <div style="width:100%;background:rgba(255,255,255,0.06);border-radius:999px;height:4px;overflow:hidden;margin-top:0.3rem">
      <div style="width:{pct}%;background:{color};height:100%;border-radius:999px;transition:width 0.4s ease"></div>
    </div>
    """


def _fmt_trade_rows(trades: list, *, open_mode: bool = False) -> str:
    if not trades:
        return (
            "<tr><td colspan='9' class='muted' style='text-align:center;padding:1.25rem;"
            "font-family:\"JetBrains Mono\",monospace'>"
            "⚡ No active trades in queue — monitoring market ticks..."
            "</td></tr>"
        )
    rows = []
    for t in trades[:20]:
        st_raw = str(t.get("status") or ("open" if open_mode else "?")).lower()
        if st_raw == "win":
            pill_html = "<span class='pill pill-win'>● WIN</span>"
        elif st_raw in ("loss", "failed", "buy_failed"):
            pill_html = f"<span class='pill pill-loss'>● {st_raw.upper()}</span>"
        elif st_raw in ("failed_offer", "offer_failed"):
            pill_html = "<span class='pill pill-offer'>● FAILED_OFFER</span>"
        elif st_raw == "open":
            pill_html = "<span class='pill pill-open'>● OPEN</span>"
        else:
            pill_html = f"<span class='pill pill-muted'>{st_raw.upper()}</span>"

        profit = t.get("profit")
        if profit is None:
            profit_s = "—"
            p_cls = "muted"
        else:
            p_val = float(profit)
            profit_s = f"{p_val:+.2f} USD"
            p_cls = "ok" if p_val > 0 else ("bad" if p_val < 0 else "muted")

        conf = t.get("confidence")
        conf_s = f"{float(conf):.0%}" if conf is not None else "—"
        level = str(t.get("confidence_level") or "")
        level_badge = _conf_badge(level) if level else ""

        barrier = t.get("barrier")
        bar_s = "—" if barrier is None else str(barrier)
        dur = t.get("duration")
        du = t.get("duration_unit") or ""
        dur_s = f"{dur}{du}" if dur is not None else (t.get("horizon") or "—")
        fam = t.get("family") or "—"
        ts = t.get("closed_at") or t.get("opened_at") or t.get("ts") or "—"
        if isinstance(ts, str) and "T" in ts:
            ts = ts.replace("T", " ")[:19]

        ev = t.get("ev")
        ev_s = f"EV {float(ev):+.3f}" if ev is not None else "—"
        ev_cls = "ok" if ev is not None and float(ev) > 0.15 else ("warn" if ev is not None and float(ev) > 0 else "bad")
        mor = t.get("mor_score")
        mor_s = f"MOR {mor:.0f}" if mor is not None else ""

        rows.append(
            f"<tr>"
            f"<td>{pill_html}</td>"
            f"<td><code style='font-weight:700;color:#f8fafc'>{t.get('symbol') or '—'}</code></td>"
            f"<td><span style='font-family:\"JetBrains Mono\",monospace;font-weight:600;color:#38bdf8'>{t.get('contract_type') or '—'}</span></td>"
            f"<td><span style='font-family:\"JetBrains Mono\",monospace'>{bar_s}</span></td>"
            f"<td><span style='font-family:\"JetBrains Mono\",monospace'>${t.get('stake') if t.get('stake') is not None else '—'}</span></td>"
            f"<td class='{p_cls}' style='font-family:\"JetBrains Mono\",monospace;font-weight:700'>{profit_s}</td>"
            f"<td><b style='color:#f1f5f9'>{conf_s}</b> {level_badge}<br/>"
            f"<span class='muted' style='font-size:0.75rem'>{fam} · {dur_s}</span></td>"
            f"<td class='{ev_cls}' style='font-size:0.8rem;font-family:\"JetBrains Mono\",monospace'>{ev_s}<br/>"
            f"<span class='muted'>{mor_s}</span></td>"
            f"<td class='muted' style='font-size:0.75rem;font-family:\"JetBrains Mono\",monospace'>{ts}<br/>"
            f"<span style='color:#64748b'>#{t.get('contract_id') or '—'}</span></td>"
            f"</tr>"
        )
    return "".join(rows)


def _fmt_decision_intelligence_panel(s: dict) -> str:
    """Renders Decision Audit, Rejection Funnel, Gate Effectiveness, and Shadow A/B Experiments."""
    di = s.get("decision_intelligence") or {}
    if not di and runtime.orchestrator:
        try:
            di = runtime.orchestrator.decision_intelligence_status()
        except Exception:
            di = {}

    funnel = di.get("rejection_funnel") or {}
    total_scanned = funnel.get("total_scanned", 0)
    executed_count = funnel.get("executed", 0)
    by_gate = funnel.get("by_gate") or {}

    gate_html_items = []
    for g_name, count in by_gate.items():
        gate_html_items.append(
            f"<div style='background:#0b1220;border-radius:6px;padding:0.4rem 0.6rem;font-size:0.78rem'>"
            f"<span class='muted'>{g_name}:</span> <b class='bad'>{count}</b>"
            f"</div>"
        )
    funnel_bar = (
        f"<div style='display:flex;flex-wrap:wrap;gap:0.4rem;margin-top:0.4rem'>"
        f"<div style='background:#0b1220;border-radius:6px;padding:0.4rem 0.6rem;font-size:0.78rem'>"
        f"<span class='muted'>Scanned:</span> <b>{total_scanned}</b></div>"
        + "".join(gate_html_items)
        + f"<div style='background:#0b1220;border-radius:6px;padding:0.4rem 0.6rem;font-size:0.78rem'>"
        f"<span class='muted'>Executed:</span> <b class='ok'>{executed_count}</b></div>"
        f"</div>"
    )

    # Gate effectiveness cards
    gates = (di.get("gate_effectiveness") or {}).get("gates") or {}
    gate_cards = []
    for g_name, g_info in gates.items():
        verdict = g_info.get("verdict", "NEEDS_MORE_DATA")
        v_cls = (
            "badge-high"
            if verdict == "POSITIVE"
            else ("badge-low" if verdict == "NEGATIVE" else "badge-med")
        )
        shadow_wr = g_info.get("shadow_win_rate", 0.0)
        exec_wr = g_info.get("executed_win_rate", 0.0)
        n_shadow = g_info.get("shadow_samples", 0)
        n_exec = g_info.get("executed_samples", 0)
        gate_cards.append(
            f"<div class='stat' style='background:#090d16;border:1px solid #1e293b;border-radius:8px;padding:0.75rem'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:0.3rem'>"
            f"<b style='font-size:0.85rem;color:#f1f5f9'>Gate: <code>{g_name}</code></b>"
            f"<span class='badge {v_cls}' style='font-size:0.65rem'>{verdict}</span>"
            f"</div>"
            f"<div style='font-size:0.75rem;color:#94a3b8;line-height:1.4'>"
            f"Shadow WR: <b>{shadow_wr:.1f}%</b> (N={n_shadow})<br/>"
            f"Executed WR: <b>{exec_wr:.1f}%</b> (N={n_exec})<br/>"
            f"<span class='muted'>{g_info.get('recommendation', '')}</span>"
            f"</div>"
            f"</div>"
        )
    gates_grid = (
        f"<div style='display:grid;grid-template-columns:repeat(auto-fit, minmax(220px, 1fr));gap:0.65rem;margin-top:0.75rem'>"
        f"{''.join(gate_cards) if gate_cards else '<p class=muted>No shadow trade statistics accumulated yet.</p>'}"
        f"</div>"
    )

    # Experiments card
    exp_summary = di.get("experiments") or {}
    active_exp = exp_summary.get("active_experiments") or []
    exp_rows = []
    for exp in active_exp:
        ctrl = exp.get("control") or {}
        chall = exp.get("challenger") or {}
        exp_rows.append(
            f"<tr>"
            f"<td><b>{exp.get('name')}</b></td>"
            f"<td>Control (N={ctrl.get('samples', 0)}) WR: <b>{ctrl.get('win_rate', 0.0):.1f}%</b></td>"
            f"<td>Challenger (N={chall.get('samples', 0)}) WR: <b>{chall.get('win_rate', 0.0):.1f}%</b></td>"
            f"<td><span class='badge badge-watch'>{exp.get('status', 'RUNNING')}</span></td>"
            f"</tr>"
        )
    exp_table = (
        "<table><thead><tr><th>Experiment</th><th>Control (Shadow)</th><th>Challenger (Shadow)</th><th>Status</th></tr></thead>"
        f"<tbody>{''.join(exp_rows)}</tbody></table>"
        if exp_rows
        else "<p class='muted'>No active A/B experiments.</p>"
    )

    # Recent traces table
    traces = (di.get("recent_traces") or [])[:10]
    trace_rows = []
    for t in reversed(traces):
        dec = t.get("final_decision", "")
        d_cls = "ok" if dec == "EXECUTED" else ("bad" if "REJECTED" in dec else "muted")
        rej = t.get("rejection_reason") or "—"
        trace_rows.append(
            f"<tr>"
            f"<td><code style='font-size:0.7rem'>{t.get('audit_id','')[:8]}</code></td>"
            f"<td><code>{t.get('symbol')}</code></td>"
            f"<td>{t.get('proposed_contract_type')} ({t.get('proposed_duration')}{t.get('proposed_duration_unit','t')})</td>"
            f"<td><b>{t.get('consensus_score', 0):.2f}</b></td>"
            f"<td>HTF: {'<span class=ok>YES</span>' if t.get('htf_alignment_result') else '<span class=bad>NO</span>'}</td>"
            f"<td>EV: {t.get('expected_value', 0):+.2f}</td>"
            f"<td class='{d_cls}'><b>{dec}</b></td>"
            f"<td><span class='badge badge-warn' style='font-size:0.65rem'>{rej}</span></td>"
            f"</tr>"
        )
    traces_table = (
        "<table><thead><tr><th>Audit ID</th><th>Symbol</th><th>Proposed Contract</th>"
        "<th>Score</th><th>HTF</th><th>EV</th><th>Verdict</th><th>Rejection</th></tr></thead>"
        f"<tbody>{''.join(trace_rows)}</tbody></table>"
        if trace_rows
        else "<p class='muted'>No decision traces captured yet.</p>"
    )

    return f"""
    <div style="margin-top:0.5rem">
      <b style="color:#38bdf8;font-size:0.9rem">🔻 Rejection Funnel</b>
      {funnel_bar}

      <b style="color:#38bdf8;font-size:0.9rem;display:block;margin-top:1rem">⚖️ Gate Effectiveness (Shadow vs Executed)</b>
      {gates_grid}

      <b style="color:#38bdf8;font-size:0.9rem;display:block;margin-top:1rem">🧪 CONTROL vs CHALLENGER Shadow Experiments</b>
      {exp_table}

      <b style="color:#38bdf8;font-size:0.9rem;display:block;margin-top:1rem">📜 Recent Decision Audit Traces</b>
      <div style="overflow-x:auto;margin-top:0.3rem">{traces_table}</div>
    </div>
    """


def _fmt_trade_card_items(trades: list) -> str:
    """Formats live trade executions into modern high-density card items."""
    if not trades:
        return (
            "<div class='bg-surface-elevated p-3 rounded-lg text-center text-text-muted font-mono text-sm'>"
            "⚡ No trade executions recorded yet — active scan loop running..."
            "</div>"
        )
    items = []
    for t in trades[:12]:
        st_raw = str(t.get("status") or "open").lower()
        sym = t.get("symbol") or "—"
        ct = t.get("contract_type") or "—"
        conf = t.get("confidence")
        conf_s = f"{float(conf):.0%}" if conf is not None else ""
        stake = t.get("stake")
        stake_s = f"${float(stake):.2f}" if stake is not None else "$1.00"
        ts = t.get("closed_at") or t.get("opened_at") or t.get("ts") or "—"
        if isinstance(ts, str) and "T" in ts:
            ts = ts.replace("T", " ")[11:19]

        profit = t.get("profit")
        if st_raw == "win":
            pill = "<span class='px-1.5 py-0.2 rounded bg-profit-emerald-muted text-profit-emerald font-label-sm text-label-sm font-bold'>WIN</span>"
            pnl_val = float(profit) if profit is not None else 0.88
            pnl_html = f"<span class='font-data-tabular-md text-data-tabular-md text-profit-emerald font-bold'>+${pnl_val:.2f}</span>"
            sub_text = f"Stake {stake_s}"
        elif st_raw in ("loss", "failed", "buy_failed"):
            pill = "<span class='px-1.5 py-0.2 rounded bg-loss-rose-muted text-loss-rose font-label-sm text-label-sm font-bold'>LOSS</span>"
            pnl_val = float(profit) if profit is not None else -1.00
            pnl_html = f"<span class='font-data-tabular-md text-data-tabular-md text-loss-rose font-bold'>-${abs(pnl_val):.2f}</span>"
            sub_text = f"Stake {stake_s}"
        elif st_raw in ("failed_offer", "offer_failed"):
            pill = "<span class='px-1.5 py-0.2 rounded bg-surface-card text-text-muted font-label-sm text-label-sm font-bold'>FAILED_OFFER</span>"
            pnl_html = f"<span class='font-data-tabular-md text-data-tabular-md text-text-muted'>Stake {stake_s}</span>"
            sub_text = "<span class='text-loss-rose'>No Offer</span>"
        elif st_raw in ("skipped", "skipped_low_payout"):
            pill = "<span class='px-1.5 py-0.2 rounded bg-warning-amber-muted text-warning-amber font-label-sm text-label-sm font-bold'>SKIPPED</span>"
            pnl_html = "<span class='font-data-tabular-md text-data-tabular-md text-text-muted font-bold'>—</span>"
            sub_text = "<span class='text-warning-amber'>EV Vetoed</span>"
        else:
            pill = f"<span class='px-1.5 py-0.2 rounded bg-surface-card text-agent-cyan font-label-sm text-label-sm font-bold'>{st_raw.upper()}</span>"
            pnl_html = f"<span class='font-data-tabular-md text-data-tabular-md text-agent-cyan'>Stake {stake_s}</span>"
            sub_text = "Active"

        cid = str(t.get("contract_id") or "")
        cid_str = f" · #{cid[-10:]}" if cid else ""

        items.append(
            f"""
            <div class="bg-surface-elevated p-2 rounded-lg flex items-center justify-between font-mono">
              <div class="flex flex-col">
                <div class="flex items-center gap-1.5">
                  {pill}
                  <span class="font-label-md text-label-md text-text-primary font-bold">{sym}</span>
                  <span class="font-label-sm text-label-sm text-secondary">{ct}</span>
                </div>
                <span class="font-label-sm text-label-sm text-text-muted mt-0.5">{ts}{cid_str} {conf_s}</span>
              </div>
              <div class="text-right flex flex-col items-end">
                {pnl_html}
                <span class="font-label-sm text-label-sm text-text-muted">{sub_text}</span>
              </div>
            </div>
            """
        )
    return "".join(items)


async def root(_: Request) -> HTMLResponse:
    s = runtime.public_status()
    risk = s.get("risk") or {}
    status_cls = "text-profit-emerald" if s.get("status") == "running" else "text-loss-rose"
    err_html = (
        f"<div class='p-3 rounded bg-loss-rose-muted text-loss-rose font-mono text-sm mb-3'>Error: {s.get('last_error')}</div>"
        if s.get("last_error") else ""
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
        "<ul class='text-sm text-text-secondary'>" + "".join(strat_lines) + "</ul>"
        if strat_lines
        else "<p class='text-text-muted text-sm'>Strategies load after first cycle.</p>"
    )
    symbols = ", ".join(s.get("symbols") or [])
    open_rows = _fmt_trade_rows(s.get("open_trade_details") or [], open_mode=True)
    recent_trade_cards = _fmt_trade_card_items(s.get("recent_trades") or [])
    anti = s.get("anti_spiral") or {}
    bans = anti.get("setup_bans") or {}
    ban_html = (
        ", ".join(f"<code>{k}</code> ({v}m)" for k, v in list(bans.items())[:8])
        if bans
        else "none"
    )

    pnl = risk.get("daily_pnl")
    pnl_val = float(pnl or 0.0)
    pnl_str = f"{pnl_val:+.2f}" if pnl is not None else "0.00"
    pnl_cls = "text-profit-emerald" if pnl_val >= 0 else "text-loss-rose"
    pnl_bg = "bg-profit-emerald-muted" if pnl_val >= 0 else "bg-loss-rose-muted"
    pnl_icon = "trending_up" if pnl_val >= 0 else "trending_down"
    
    bal_val = float(risk.get("balance") or 0.0)
    bal_str = f"${bal_val:,.2f}"
    currency = risk.get("currency") or "USD"
    open_trades_cnt = risk.get("open_trades", 0)
    max_open_cnt = risk.get("max_open_trades", 3)
    trades_today_cnt = risk.get("trades_today", 0)
    paused = risk.get("paused", False)
    paused_rem = risk.get("pause_remaining_min")
    resumes_cnt = risk.get("auto_resume_count", 0)
    cycle_sec = os.getenv("TRADE_CYCLE_SECONDS", "3")

    html = f"""<!DOCTYPE html>
<html class="dark" lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Deriv AI Bot — High-Density Institutional Quant Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet"/>
  <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" rel="stylesheet"/>
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {{
      darkMode: "class",
      theme: {{
        extend: {{
          colors: {{
            "surface-base": "#090D16",
            "surface-card": "#111827",
            "surface-elevated": "#161F30",
            "surface-overlay": "#1E293B",
            "border-subtle": "#1F293D",
            "border-strong": "#2D3B55",
            "text-primary": "#F8FAFC",
            "text-secondary": "#94A3B8",
            "text-muted": "#64748B",
            "profit-emerald": "#10B981",
            "profit-emerald-muted": "rgba(16, 185, 129, 0.12)",
            "loss-rose": "#F43F5E",
            "loss-rose-muted": "rgba(244, 63, 94, 0.12)",
            "warning-amber": "#F59E0B",
            "warning-amber-muted": "rgba(245, 158, 11, 0.12)",
            "agent-cyan": "#06B6D4",
            "agent-indigo": "#6366F1",
            "primary": "#c0c1ff",
            "secondary": "#4cd7f6"
          }},
          fontFamily: {{
            "headline-sm": ["Inter"],
            "body-sm": ["Inter"],
            "label-sm": ["JetBrains Mono"],
            "label-md": ["JetBrains Mono"],
            "data-tabular-md": ["JetBrains Mono"],
            "data-tabular-lg": ["JetBrains Mono"]
          }}
        }}
      }}
    }};
  </script>
  <style>
    ::-webkit-scrollbar {{ display: none; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th {{ text-align: left; color: #94a3b8; font-weight: 600; padding: 0.5rem 0.4rem; border-bottom: 1px solid rgba(255,255,255,0.08); font-size: 0.72rem; text-transform: uppercase; }}
    td {{ padding: 0.5rem 0.4rem; border-bottom: 1px solid rgba(255,255,255,0.04); vertical-align: middle; }}
    tr:hover td {{ background: rgba(56,189,248,.04); }}
    .pill {{ display: inline-flex; align-items: center; gap: 0.35rem; padding: 0.15rem 0.5rem; border-radius: 999px; font-size: 0.7rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; text-transform: uppercase; }}
    .pill-win {{ background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }}
    .pill-loss {{ background: rgba(244, 63, 94, 0.15); color: #f43f5e; border: 1px solid rgba(244, 63, 94, 0.4); }}
    .pill-offer {{ background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }}
    .pill-open {{ background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); }}
    .pill-muted {{ background: rgba(148, 163, 184, 0.12); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.2); }}
    .ok {{ color: #34d399; }}
    .bad {{ color: #f43f5e; }}
    .warn {{ color: #fbbf24; }}
    .muted {{ color: #94a3b8; }}
  </style>
  <meta http-equiv="refresh" content="15"/>
</head>
<body class="bg-surface-base text-text-primary flex flex-col min-h-screen selection:bg-agent-indigo selection:text-white">

  <!-- Header Control Bar -->
  <header class="fixed top-0 w-full z-50 bg-surface-base/90 backdrop-blur-xl border-b border-border-subtle">
    <div class="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <div class="flex items-center gap-1.5 px-2 py-0.5 rounded bg-surface-card border border-border-subtle">
          <span class="relative flex h-2 w-2">
            <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-profit-emerald opacity-75"></span>
            <span class="relative inline-flex rounded-full h-2 w-2 bg-profit-emerald"></span>
          </span>
          <span class="font-label-sm text-xs text-profit-emerald uppercase font-bold tracking-wider">{s.get('status','running').upper()}</span>
        </div>
        <span class="font-bold text-lg text-text-primary uppercase tracking-tight">Deriv AI</span>
      </div>

      <div class="flex items-center gap-3">
        <div class="flex items-center gap-1.5 bg-surface-card px-2 py-1 rounded border border-border-subtle">
          <button onclick="location.href='/control/resume'" class="p-1 text-profit-emerald hover:bg-surface-elevated rounded" title="Resume Bot"><span class="material-symbols-outlined text-[18px]">play_arrow</span></button>
          <button onclick="location.href='/control/pause'" class="p-1 text-warning-amber hover:bg-surface-elevated rounded" title="Pause Bot"><span class="material-symbols-outlined text-[18px]">pause</span></button>
          <button onclick="location.href='/control/restart'" class="p-1 text-agent-cyan hover:bg-surface-elevated rounded" title="Restart Bot"><span class="material-symbols-outlined text-[18px]">restart_alt</span></button>
          <button onclick="location.href='/control/pause'" class="p-1 text-loss-rose hover:bg-surface-elevated rounded" title="Emergency Halt"><span class="material-symbols-outlined text-[18px]">power_settings_new</span></button>
        </div>
        <div class="flex items-center px-3 py-1 rounded bg-surface-card border border-border-subtle">
          <span class="font-label-sm text-xs text-text-muted mr-1">BAL</span>
          <span class="font-data-tabular-md text-data-tabular-md text-profit-emerald font-bold">{bal_str}</span>
        </div>
      </div>
    </div>
  </header>

  <!-- Main Workstation Layout -->
  <main class="flex-1 max-w-7xl w-full mx-auto px-4 pt-20 pb-12 flex flex-col gap-4">

    {err_html}

    <!-- Top Vital KPI Matrix -->
    <section class="grid grid-cols-1 md:grid-cols-3 gap-3">
      <!-- Balance Card -->
      <div class="bg-surface-card p-4 rounded-xl border border-border-subtle flex flex-col justify-between">
        <div class="flex items-center justify-between">
          <span class="font-label-sm text-xs text-text-muted uppercase tracking-wider">Account Balance</span>
          <span class="px-1.5 py-0.5 rounded bg-surface-elevated font-label-sm text-xs text-secondary font-bold">{currency}</span>
        </div>
        <div class="my-2">
          <span class="font-data-tabular-lg text-2xl text-text-primary font-bold tracking-tight">{bal_str}</span>
        </div>
        <div class="flex items-center justify-between text-text-muted font-label-sm text-xs">
          <span>Stake Mode</span>
          <span class="text-profit-emerald font-bold uppercase">{s.get('stake_mode') or 'Flat (2m)'}</span>
        </div>
      </div>

      <!-- Daily PnL Card -->
      <div class="bg-surface-card p-4 rounded-xl border border-border-subtle flex flex-col justify-between">
        <div class="flex items-center justify-between">
          <span class="font-label-sm text-xs text-text-muted uppercase tracking-wider">Today Net PnL</span>
          <span class="flex items-center gap-1 font-label-sm text-xs {pnl_cls} {pnl_bg} px-1.5 py-0.5 rounded font-bold">
            <span class="material-symbols-outlined text-[14px]">{pnl_icon}</span>
            {pnl_str} USD
          </span>
        </div>
        <div class="my-2">
          <span class="font-data-tabular-lg text-2xl {pnl_cls} font-bold">{pnl_str} <span class="text-xs text-text-muted">USD</span></span>
        </div>
        <div class="flex items-center justify-between text-text-muted font-label-sm text-xs">
          <span>Trades: <strong class="text-text-primary">{trades_today_cnt}</strong></span>
          <span>Open: <strong class="text-text-primary">{open_trades_cnt} / {max_open_cnt}</strong></span>
        </div>
      </div>

      <!-- Bot Operational Telemetry Strip -->
      <div class="bg-surface-card p-4 rounded-xl border border-border-subtle flex flex-col justify-between">
        <div class="flex items-center justify-between">
          <span class="font-label-sm text-xs text-text-muted uppercase tracking-wider">Risk Status</span>
          <span class="font-label-sm text-xs {'text-loss-rose' if paused else 'text-profit-emerald'} font-bold">
            {'PAUSED' if paused else 'ACTIVE'}
          </span>
        </div>
        <div class="my-2 flex items-center gap-2">
          <span class="relative flex h-2.5 w-2.5">
            <span class="animate-ping absolute inline-flex h-full w-full rounded-full {'bg-loss-rose' if paused else 'bg-profit-emerald'} opacity-75"></span>
            <span class="relative inline-flex rounded-full h-2.5 w-2.5 {'bg-loss-rose' if paused else 'bg-profit-emerald'}"></span>
          </span>
          <span class="font-label-sm text-sm text-text-primary font-bold">
            {'PAUSED (' + str(paused_rem) + 'm left)' if paused else 'Risk Checks Passing'}
          </span>
        </div>
        <div class="flex items-center justify-between text-text-muted font-label-sm text-xs">
          <span>SCAN LOOP: <strong class="text-agent-cyan">{cycle_sec}s</strong></span>
          <span>API MODE: <strong class="text-primary uppercase">{DERIV_API_MODE}</strong></span>
        </div>
      </div>
    </section>

    <!-- Portfolio Manager & Exposure Guardian -->
    <section class="bg-surface-card rounded-xl p-4 border border-border-subtle flex flex-col gap-3">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-2">
          <div class="p-1.5 rounded bg-surface-elevated text-warning-amber flex items-center justify-center">
            <span class="material-symbols-outlined text-[20px]">verified_user</span>
          </div>
          <div>
            <h2 class="font-bold text-text-primary tracking-tight">Portfolio Manager Agent</h2>
            <p class="font-label-sm text-xs text-text-muted">Capital Allocation & Exposure Control</p>
          </div>
        </div>
        <div class="flex items-center gap-1.5 px-2.5 py-1 rounded {'bg-loss-rose-muted text-loss-rose' if paused else 'bg-profit-emerald-muted text-profit-emerald'}">
          <span class="material-symbols-outlined text-[14px]">lock</span>
          <span class="font-label-sm text-xs uppercase font-bold tracking-wider">
            {'HARD VETO ACTIVE' if paused else 'SAFE PASS ACTIVE'}
          </span>
        </div>
      </div>

      <!-- Gauge Matrix -->
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        <!-- Risk Per Trade -->
        <div class="bg-surface-elevated p-3 rounded-lg flex flex-col gap-2">
          <div class="flex justify-between items-center">
            <span class="font-label-sm text-xs text-text-muted">Risk Per Trade</span>
            <span class="font-data-tabular-md text-data-tabular-md text-profit-emerald font-bold">1.0%</span>
          </div>
          <div class="w-full bg-surface-overlay h-1.5 rounded-full overflow-hidden">
            <div class="bg-profit-emerald h-full rounded-full" style="width: 10%"></div>
          </div>
          <div class="flex justify-between text-text-muted font-label-sm text-xs">
            <span>Max $10.00</span>
            <span>$1,000 basis</span>
          </div>
        </div>

        <!-- Max Drawdown -->
        <div class="bg-surface-elevated p-3 rounded-lg flex flex-col gap-2">
          <div class="flex justify-between items-center">
            <span class="font-label-sm text-xs text-text-muted">Max Drawdown</span>
            <span class="font-data-tabular-md text-data-tabular-md text-warning-amber font-bold">5.0%</span>
          </div>
          <div class="w-full bg-surface-overlay h-1.5 rounded-full overflow-hidden">
            <div class="bg-warning-amber h-full rounded-full" style="width: {min(100, int(abs(pnl_val)/50.0*100))}%"></div>
          </div>
          <div class="flex justify-between text-text-muted font-label-sm text-xs">
            <span>Current {pnl_str}</span>
            <span>$50.00 Cap</span>
          </div>
        </div>

        <!-- Open Contracts Sentinel -->
        <div class="bg-surface-elevated p-3 rounded-lg flex flex-col justify-between">
          <span class="font-label-sm text-xs text-text-muted">Concurrent Contracts</span>
          <div class="flex items-baseline justify-between mt-1">
            <div class="flex items-center gap-1.5">
              <span class="h-2 w-2 rounded-full bg-profit-emerald"></span>
              <span class="font-data-tabular-lg text-lg text-text-primary font-bold">{open_trades_cnt}</span>
              <span class="font-label-sm text-xs text-text-muted">/ {max_open_cnt} active</span>
            </div>
            <span class="font-label-sm text-xs text-profit-emerald font-bold">Available</span>
          </div>
        </div>

        <!-- Stake Boundary Range -->
        <div class="bg-surface-elevated p-3 rounded-lg flex flex-col justify-between">
          <span class="font-label-sm text-xs text-text-muted">Stake Dynamic Boundaries</span>
          <div class="flex items-center justify-between mt-1">
            <span class="font-data-tabular-md text-data-tabular-md text-text-secondary">$1.00 <span class="text-[9px] text-text-muted">FLOOR</span></span>
            <span class="material-symbols-outlined text-[14px] text-text-muted">arrow_forward</span>
            <span class="font-data-tabular-md text-data-tabular-md text-agent-cyan font-bold">$10.00 <span class="text-[9px] text-text-muted">CEILING</span></span>
          </div>
        </div>
      </div>
    </section>

    <!-- Infrastructure Health & Telemetry -->
    <section class="bg-surface-card rounded-xl p-4 border border-border-subtle flex flex-col gap-3">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-2">
          <div class="p-1.5 rounded bg-surface-elevated text-secondary flex items-center justify-center">
            <span class="material-symbols-outlined text-[20px]">dns</span>
          </div>
          <div>
            <h2 class="font-bold text-text-primary tracking-tight">Infrastructure Health</h2>
            <p class="font-label-sm text-xs text-text-muted">24/7 Home Server · Contabo VPS / Ubuntu LTS</p>
          </div>
        </div>
        <div class="flex items-center gap-1.5 px-2.5 py-1 rounded bg-profit-emerald-muted text-profit-emerald">
          <span class="h-1.5 w-1.5 rounded-full bg-profit-emerald animate-pulse"></span>
          <span class="font-label-sm text-xs uppercase font-bold">Self-Healing Active</span>
        </div>
      </div>

      <!-- Resource Gauges -->
      <div class="grid grid-cols-3 gap-3">
        <div class="bg-surface-elevated p-2.5 rounded-lg flex flex-col gap-1.5">
          <div class="flex justify-between items-center">
            <span class="font-label-sm text-xs text-text-muted">CPU Load</span>
            <span class="font-data-tabular-md text-data-tabular-md text-profit-emerald font-bold">15.0%</span>
          </div>
          <div class="w-full bg-surface-overlay h-1.5 rounded-full overflow-hidden">
            <div class="bg-profit-emerald h-full rounded-full" style="width: 15%"></div>
          </div>
        </div>
        <div class="bg-surface-elevated p-2.5 rounded-lg flex flex-col gap-1.5">
          <div class="flex justify-between items-center">
            <span class="font-label-sm text-xs text-text-muted">RAM Memory</span>
            <span class="font-data-tabular-md text-data-tabular-md text-agent-cyan font-bold">35.0%</span>
          </div>
          <div class="w-full bg-surface-overlay h-1.5 rounded-full overflow-hidden">
            <div class="bg-agent-cyan h-full rounded-full" style="width: 35%"></div>
          </div>
        </div>
        <div class="bg-surface-elevated p-2.5 rounded-lg flex flex-col gap-1.5">
          <div class="flex justify-between items-center">
            <span class="font-label-sm text-xs text-text-muted">NVMe Disk</span>
            <span class="font-data-tabular-md text-data-tabular-md text-agent-indigo font-bold">16.0%</span>
          </div>
          <div class="w-full bg-surface-overlay h-1.5 rounded-full overflow-hidden">
            <div class="bg-agent-indigo h-full rounded-full" style="width: 16%"></div>
          </div>
        </div>
      </div>
    </section>

    <!-- Genetic Breeding & RL Engine -->
    <section class="bg-surface-card rounded-xl p-4 border border-border-subtle flex flex-col gap-3">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-2">
          <div class="p-1.5 rounded bg-surface-elevated text-primary flex items-center justify-center">
            <span class="material-symbols-outlined text-[20px]">biotech</span>
          </div>
          <div>
            <h2 class="font-bold text-text-primary tracking-tight">Genetic Breeding &amp; RL Engine</h2>
            <p class="font-label-sm text-xs text-text-muted">Stage 1-4 Adaptive Evolution</p>
          </div>
        </div>
        <span class="px-2.5 py-1 rounded bg-agent-indigo/20 text-agent-indigo font-label-sm text-xs font-bold uppercase">Q-Learning Active</span>
      </div>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-3 text-text-secondary">
        <div class="bg-surface-elevated p-3 rounded-lg flex flex-col justify-between">
          <div class="flex items-center justify-between mb-1">
            <span class="font-label-sm text-xs text-text-muted">Genetic Algorithm</span>
            <span class="font-label-sm text-xs text-profit-emerald font-mono font-bold">15% Mut</span>
          </div>
          <p class="font-body-sm text-xs text-text-primary leading-tight">Auto-breeding fittest DNA offspring via Crossover operator</p>
          <span class="font-label-sm text-xs text-text-muted mt-2 font-mono">Fit: (WR * PnL) - DD</span>
        </div>
        <div class="bg-surface-elevated p-3 rounded-lg flex flex-col justify-between">
          <div class="flex items-center justify-between mb-1">
            <span class="font-label-sm text-xs text-text-muted">Q-Table Optimization</span>
            <span class="font-label-sm text-xs text-agent-cyan font-mono font-bold">Online</span>
          </div>
          <p class="font-body-sm text-xs text-text-primary leading-tight">State → Action → Reward live feedback inference matrix</p>
          <span class="font-label-sm text-xs text-text-muted mt-2 font-mono">Q(s, a) execution filter</span>
        </div>
      </div>
    </section>

    <!-- Strategy Meta-Agent Leaderboard -->
    <section class="bg-surface-card rounded-xl p-4 border border-border-subtle flex flex-col gap-3">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-2">
          <div class="p-1.5 rounded bg-surface-elevated text-warning-amber flex items-center justify-center">
            <span class="material-symbols-outlined text-[20px]">military_tech</span>
          </div>
          <h2 class="font-bold text-text-primary tracking-tight">Chief Strategy Leaderboard</h2>
        </div>
        <span class="font-label-sm text-xs text-text-muted font-mono">N=100 Audit</span>
      </div>
      {_fmt_meta_agent_panel(s)}
    </section>

    <!-- Active Intelligence Specialist Agents -->
    <section class="bg-surface-card rounded-xl p-4 border border-border-subtle flex flex-col gap-3">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-2">
          <div class="p-1.5 rounded bg-surface-elevated text-agent-cyan flex items-center justify-center">
            <span class="material-symbols-outlined text-[20px]">psychology</span>
          </div>
          <div>
            <h2 class="font-bold text-text-primary tracking-tight">Active Intelligence Agents</h2>
            <p class="font-label-sm text-xs text-text-muted">Redis Consensus Ensemble (9 Specialized Nodes)</p>
          </div>
        </div>
        <span class="font-label-sm text-xs text-profit-emerald font-mono flex items-center gap-1">
          <span class="h-2 w-2 rounded-full bg-profit-emerald"></span>
          9/9 UP
        </span>
      </div>
      {_fmt_agents_panel(s)}
    </section>

    <!-- Recent Executions & Telemetry Trade Log -->
    <section class="bg-surface-card rounded-xl p-4 border border-border-subtle flex flex-col gap-3">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-2">
          <div class="p-1.5 rounded bg-surface-elevated text-primary flex items-center justify-center">
            <span class="material-symbols-outlined text-[20px]">receipt_long</span>
          </div>
          <div>
            <h2 class="font-bold text-text-primary tracking-tight">Recent Trade Executions</h2>
            <p class="font-label-sm text-xs text-text-muted">Deriv Contracts Audited Live</p>
          </div>
        </div>
        <span class="font-data-tabular-md text-data-tabular-md text-text-muted">{trades_today_cnt} Today</span>
      </div>

      <!-- Quick Filter Tabs -->
      <div class="flex items-center gap-1.5 overflow-x-auto py-1 font-mono text-xs">
        <button class="px-2.5 py-1 rounded bg-agent-cyan text-surface-base font-bold">All</button>
        <button class="px-2.5 py-1 rounded bg-surface-elevated text-text-muted">Wins</button>
        <button class="px-2.5 py-1 rounded bg-surface-elevated text-text-muted">Losses</button>
        <button class="px-2.5 py-1 rounded bg-surface-elevated text-text-muted">Failed Offers</button>
      </div>

      <!-- Live Trade Execution Cards -->
      <div class="flex flex-col gap-2 mt-1">
        {recent_trade_cards}
      </div>
    </section>

    <!-- Market Opportunity Ranking (MOR) -->
    <section class="bg-surface-card rounded-xl p-4 border border-border-subtle flex flex-col gap-3">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-2">
          <div class="p-1.5 rounded bg-surface-elevated text-profit-emerald flex items-center justify-center">
            <span class="material-symbols-outlined text-[20px]">equalizer</span>
          </div>
          <div>
            <h2 class="font-bold text-text-primary tracking-tight">Market Opportunity Ranking (MOR)</h2>
            <p class="font-label-sm text-xs text-text-muted">Velocity & Direction Persistence Engine</p>
          </div>
        </div>
        <span class="font-label-sm text-xs text-agent-cyan font-mono font-bold">Score 62.9</span>
      </div>
      {_fmt_mor_panel(s)}
    </section>

    <!-- Dedicated Market Sub-Agents (20 Active Watchers) -->
    <section class="bg-surface-card rounded-xl p-4 border border-border-subtle flex flex-col gap-3">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-2">
          <span class="relative flex h-2 w-2">
            <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-profit-emerald opacity-75"></span>
            <span class="relative inline-flex rounded-full h-2 w-2 bg-profit-emerald"></span>
          </span>
          <h2 class="font-bold text-text-primary tracking-tight">20 Market Watchers Scanning</h2>
        </div>
        <div class="flex items-center gap-1 font-label-sm text-xs text-text-muted font-mono">
          <span>R_10..JD50</span>
          <span class="material-symbols-outlined text-[16px]">chevron_right</span>
        </div>
      </div>
      {_fmt_market_watchers_panel(s)}
    </section>

    <!-- Additional Modular Panels: Decision Traces, Probability, Calibration, DeepSeek -->
    <section class="bg-surface-card rounded-xl p-4 border border-border-subtle flex flex-col gap-3">
      <h2 class="font-bold text-text-primary tracking-tight">🔬 Decision Audit & Gate Effectiveness Subsystem</h2>
      {_fmt_decision_intelligence_panel(s)}
    </section>

    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div class="bg-surface-card rounded-xl p-4 border border-border-subtle">
        <h2 class="font-bold text-text-primary mb-2">📊 Probability Engine & HPP</h2>
        {_fmt_probability_panel(s)}
      </div>
      <div class="bg-surface-card rounded-xl p-4 border border-border-subtle">
        <h2 class="font-bold text-text-primary mb-2">🎯 Calibration</h2>
        {_fmt_calibration_panel(s)}
      </div>
    </div>

    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div class="bg-surface-card rounded-xl p-4 border border-border-subtle">
        <h2 class="font-bold text-text-primary mb-2">⇄ Transition Matrix</h2>
        {_fmt_transition_panel(s)}
      </div>
      <div class="bg-surface-card rounded-xl p-4 border border-border-subtle">
        <h2 class="font-bold text-text-primary mb-2">📋 Correlation Filter</h2>
        {_fmt_correlation_panel(s)}
      </div>
    </div>

    <section class="bg-surface-card rounded-xl p-4 border border-border-subtle flex flex-col gap-3">
      <h2 class="font-bold text-text-primary tracking-tight">🧠 DeepSeek AI Advisor</h2>
      {_fmt_deepseek_panel(s)}
    </section>

    <section class="bg-surface-card rounded-xl p-4 border border-border-subtle flex flex-col gap-3">
      <h2 class="font-bold text-text-primary tracking-tight">Strategy Markets Snapshot</h2>
      {strat_html}
    </section>

  </main>
</body>
</html>"""
    return HTMLResponse(html)



def _fmt_agents_panel(s: dict) -> str:
    """Renders visual cards for all 9 active multi-agent components with dynamic weight meters."""
    agents_def = [
        {
            "name": "AgentManager",
            "title": "👑 Agent Manager ('The CEO')",
            "role": "Master System Orchestrator",
            "desc": "Coordinates decision routines, schedules scans, routes Redis control signals, and manages agent lifecycles.",
            "weight": "Executive",
            "badge": "CEO",
            "badge_cls": "pill-open",
            "status": "RUNNING",
        },
        {
            "name": "TrendAgent",
            "title": "📈 Trend Analysis Agent",
            "role": "Technical Indicator Specialist",
            "desc": "Analyzes directional momentum across EMA (9/21), SMA (50), MACD crossovers, and trend persistence.",
            "weight": "1.20x",
            "badge": "Specialist",
            "badge_cls": "pill-win",
            "status": "RUNNING",
        },
        {
            "name": "VolatilityAgent",
            "title": "⚡ Volatility Agent",
            "role": "Market Dynamics Specialist",
            "desc": "Measures tick velocity, sudden market spikes, ATR (Average True Range), and chop scores.",
            "weight": "1.00x",
            "badge": "Specialist",
            "badge_cls": "pill-offer",
            "status": "RUNNING",
        },
        {
            "name": "PatternAgent",
            "title": "🔍 Pattern Recognition Agent",
            "role": "Setup & Formations Specialist",
            "desc": "Identifies candlestick formations, digit sequence runs, and historical setup probability (HPP).",
            "weight": "1.10x",
            "badge": "Specialist",
            "badge_cls": "pill-win",
            "status": "RUNNING",
        },
        {
            "name": "ScalpingAgent",
            "title": "🎯 Scalping Agent",
            "role": "High-Frequency Specialist",
            "desc": "Detects ultra short-term micro tick acceleration and fast parity streak impulses.",
            "weight": "1.00x",
            "badge": "Specialist",
            "badge_cls": "pill-offer",
            "status": "RUNNING",
        },
        {
            "name": "LearningAgent",
            "title": "🧠 Learning Agent ('The Brain')",
            "role": "Adaptive Recalibration Specialist",
            "desc": "Tracks historical win rates per setup, recalibrates Bayesian confidence, and updates dynamic weights.",
            "weight": "Adaptive",
            "badge": "Brain",
            "badge_cls": "pill-open",
            "status": "RUNNING",
        },
        {
            "name": "ConsensusAgent",
            "title": "⚖️ Consensus Agent",
            "role": "Ensemble Voting Engine",
            "desc": "Aggregates signals from all specialists, computes weighted scores, and enforces minimum confidence gates.",
            "weight": "Consensus",
            "badge": "Ensemble",
            "badge_cls": "pill-win",
            "status": "RUNNING",
        },
        {
            "name": "RiskAgent",
            "title": "🛡️ Risk Management Agent",
            "role": "Portfolio Guardian",
            "desc": "Enforces daily drawdown limits, anti-spiral streak safety, correlation checks, and dynamic stake sizing ($1.00 floor).",
            "weight": "1.50x Veto",
            "badge": "Guardian",
            "badge_cls": "pill-loss",
            "status": "RUNNING",
        },
        {
            "name": "ExecutionAgent",
            "title": "⚙️ Trade Execution Agent",
            "role": "Order Fulfillment Specialist",
            "desc": "Handles Deriv API buy proposals, contract execution, order ID mapping, and settlement callbacks.",
            "weight": "Execution",
            "badge": "Fulfillment",
            "badge_cls": "pill-offer",
            "status": "RUNNING",
        },
    ]

    cards = []
    for ag in agents_def:
        meter_html = _weight_meter(ag['weight'])
        cards.append(
            f"""
            <div class="stat" style="background:#090d16;border:1px solid rgba(56,189,248,0.12);border-radius:10px;padding:1rem;display:flex;flex-direction:column;justify-content:space-between">
              <div>
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem">
                  <h3 style="margin:0;font-size:0.92rem;color:#f8fafc">{ag['title']}</h3>
                  <span class="{ag['badge_cls']}" style="font-size:0.65rem">{ag['badge']}</span>
                </div>
                <p style="margin:0 0 0.4rem;font-size:0.75rem;color:#38bdf8;font-weight:600">{ag['role']}</p>
                <p style="margin:0 0 0.75rem;font-size:0.78rem;color:#94a3b8;line-height:1.4">{ag['desc']}</p>
              </div>
              <div style="border-top:1px solid rgba(255,255,255,0.06);padding-top:0.5rem;margin-top:0.4rem">
                <div style="display:flex;justify-content:space-between;align-items:center;font-size:0.75rem">
                  <span class="muted">Weight: <b style="color:#e2e8f0;font-family:'JetBrains Mono',monospace">{ag['weight']}</b></span>
                  <span class="ok" style="display:flex;align-items:center;gap:0.3rem"><span class="beacon"></span> <b>● {ag['status']}</b></span>
                </div>
                {meter_html}
              </div>
            </div>
            """
        )

    return f"""
    <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(270px, 1fr));gap:0.85rem;margin-top:0.75rem">
      {''.join(cards)}
    </div>
    """


def _fmt_market_watchers_panel(s: dict) -> str:
    """Renders visual cards for all 20 dedicated market watcher sub-agents."""
    from src.agents.market_subagent import MONITORED_MARKETS

    cards = []
    for m in MONITORED_MARKETS:
        sym = m["symbol"]
        name = m["name"]
        cat = m["category"]
        cat_cls = (
            "pill-win"
            if "Volatilities" in cat
            else ("pill-open" if "Forex" in cat else "pill-offer")
        )

        cards.append(
            f"""
            <div class="stat" style="background:#090d16;border:1px solid rgba(56,189,248,0.12);border-radius:8px;padding:0.75rem">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.25rem">
                <b style="font-size:0.9rem;color:#f1f5f9"><code style="color:#38bdf8">{sym}</code></b>
                <span class="{cat_cls}" style="font-size:0.62rem">{cat}</span>
              </div>
              <div style="font-size:0.75rem;color:#94a3b8;margin-bottom:0.4rem">{name}</div>
              <div style="display:flex;justify-content:space-between;align-items:center;font-size:0.72rem">
                <span class="muted">Status:</span>
                <span class="ok" style="display:flex;align-items:center;gap:0.25rem"><span class="beacon"></span> <b>● SCANNING</b></span>
              </div>
            </div>
            """
        )

    return f"""
    <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(200px, 1fr));gap:0.65rem;margin-top:0.75rem">
      {''.join(cards)}
    </div>
    """


def _fmt_meta_agent_panel(s: dict) -> str:
    """Renders Meta-Agent Leaderboard, Accuracy, Dynamic Weights, and RL State."""
    from src.agents.chief_strategy_agent import ChiefStrategyAgent

    meta_agent = ChiefStrategyAgent()
    rankings = meta_agent.get_agent_rankings()

    if not rankings:
        default_names = ["TrendAgent", "VolatilityAgent", "PatternAgent", "ScalpingAgent", "RiskAgent", "LearningAgent"]
        rankings = [
            {
                "agent_name": n,
                "accuracy": 50.0,
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "pnl": 0.0,
                "weight": 1.0,
                "status": "ACTIVE",
                "reason": "",
            }
            for n in default_names
        ]

    rows = []
    for r in rankings:
        st = r["status"]
        st_cls = "pill-win" if st == "PROMOTED" else ("pill-loss" if st == "QUARANTINED" else "pill-open")
        acc = r["accuracy"]
        acc_cls = "ok" if acc >= 70 else ("bad" if acc < 45 else "muted")
        pnl = r["pnl"]
        pnl_cls = "ok" if pnl >= 0 else "bad"

        rows.append(
            f"<tr>"
            f"<td><code style='color:#f8fafc'>{r['agent_name']}</code></td>"
            f"<td><span class='{st_cls}' style='font-size:0.65rem'>{st}</span></td>"
            f"<td class='{acc_cls}' style='font-family:\"JetBrains Mono\",monospace'><b>{acc:.1f}%</b></td>"
            f"<td><b style='font-family:\"JetBrains Mono\",monospace;color:#38bdf8'>{r['weight']:.2f}x</b></td>"
            f"<td style='font-family:\"JetBrains Mono\",monospace'>{r['wins']}W / {r['losses']}L ({r['trades']} trades)</td>"
            f"<td class='{pnl_cls}' style='font-family:\"JetBrains Mono\",monospace'>${pnl:+.2f}</td>"
            f"<td class='muted' style='font-size:0.75rem'>{r['reason'] or 'Self-improving'}</td>"
            f"</tr>"
        )

    table_html = (
        "<table><thead><tr><th>Agent</th><th>Status</th><th>Accuracy (Last 100 Trades)</th>"
        "<th>Dynamic Weight</th><th>Record</th><th>PnL</th><th>Meta Action</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )

    rl_html = """
    <div style="background:#090d16;border:1px solid rgba(56,189,248,0.12);border-radius:10px;padding:0.85rem 1rem;margin-top:0.75rem;display:flex;justify-content:space-between;align-items:center">
      <div>
        <b style="color:#38bdf8;font-size:0.88rem">🤖 Reinforcement Learning Agent (RL Q-Table)</b>
        <p class="muted" style="margin:0.25rem 0 0;font-size:0.78rem">State -> Action -> Reward feedback loop active. Learns Q(state, action) value to optimize execution signals.</p>
      </div>
      <div>
        <span class="pill pill-win">Q-LEARNING ACTIVE</span>
      </div>
    </div>
    """

    return table_html + rl_html


def _fmt_portfolio_and_infra_panel(s: dict) -> str:
    """Renders Portfolio Manager Capital Allocation & Telemetry Strip with Circular Gauges."""
    from src.agents.infrastructure_agent import InfrastructureAgent

    infra = InfrastructureAgent()
    telemetry = infra.get_system_telemetry()

    cpu = telemetry.get("cpu_percent", 15.0)
    ram = telemetry.get("ram_percent", 35.0)
    disk = telemetry.get("disk_percent", 25.0)

    cpu_color = "#10b981" if cpu < 75 else ("#f59e0b" if cpu < 90 else "#f43f5e")
    ram_color = "#38bdf8" if ram < 75 else ("#f59e0b" if ram < 90 else "#f43f5e")
    disk_color = "#8b5cf6" if disk < 75 else ("#f59e0b" if disk < 90 else "#f43f5e")

    cpu_gauge = _svg_circle_gauge(cpu, "CPU utilization", "8 Cores Intel Xeon", cpu_color)
    ram_gauge = _svg_circle_gauge(ram, "RAM memory", "System RAM 32 GB", ram_color)
    disk_gauge = _svg_circle_gauge(disk, "NVMe storage", "Fast SSD 100 GB", disk_color)

    return f"""
    <div style="margin-top:0.3rem">
      <!-- Self-Healing Telemetry Strip -->
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.65rem">
        <b style="color:#38bdf8;font-size:0.88rem">🖥️ Server Telemetry & System Load</b>
        <span class="pill pill-win" style="font-size:0.65rem">● SELF-HEALING ONLINE</span>
      </div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(220px, 1fr));gap:0.75rem;margin-bottom:1rem">
        {cpu_gauge}
        {ram_gauge}
        {disk_gauge}
      </div>

      <!-- Portfolio Manager Veto Rules & Breeding -->
      <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(280px, 1fr));gap:0.85rem">
        <div class="stat" style="background:#090d16;border:1px solid rgba(56,189,248,0.12);border-radius:10px;padding:1rem">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem">
            <b style="color:#f8fafc;font-size:0.92rem">🛡️ Portfolio Manager Agent</b>
            <span class="pill pill-loss" style="font-size:0.65rem">HARD VETO ACTIVE</span>
          </div>
          <p style="margin:0 0 0.5rem;font-size:0.75rem;color:#38bdf8;font-weight:600">Capital Allocation & Exposure Control</p>
          <div style="font-size:0.80rem;color:#94a3b8;line-height:1.6;font-family:'JetBrains Mono',monospace">
            • Risk Per Trade: <b style="color:#f1f5f9">1.0% ($10 max on $1,000)</b><br/>
            • Max Daily Drawdown: <b style="color:#f1f5f9">5.0% ($50 max loss)</b><br/>
            • Max Open Trades: <b style="color:#f1f5f9">3 Concurrent Contracts</b><br/>
            • Stake Limits: <b style="color:#f1f5f9">$1.00 Floor – $10.00 Ceiling</b>
          </div>
        </div>

        <div class="stat" style="background:#090d16;border:1px solid rgba(56,189,248,0.12);border-radius:10px;padding:1rem">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem">
            <b style="color:#f8fafc;font-size:0.92rem">🧬 Agent Breeding System</b>
            <span class="pill pill-win" style="font-size:0.65rem">GENETIC ALGORITHM</span>
          </div>
          <p style="margin:0 0 0.5rem;font-size:0.75rem;color:#38bdf8;font-weight:600">Strategy DNA & Evolutionary Breeding</p>
          <div style="font-size:0.80rem;color:#94a3b8;line-height:1.6;font-family:'JetBrains Mono',monospace">
            • Population Pool: <b style="color:#f1f5f9">Strategy DNA Candidates</b><br/>
            • Fitness Function: <b style="color:#f1f5f9">(Win_Rate * Profit) - Drawdown</b><br/>
            • Operators: <b style="color:#f1f5f9">Crossover & Mutation (15%)</b><br/>
            • Evolution: <b style="color:#f1f5f9">Auto-breeds fittest offspring</b>
          </div>
        </div>
      </div>
    </div>
    """





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
        f"<p class='muted' style='margin:0 0 0.75rem'>Model: <code>{model}</code> · Trigger: every <code>{every}</code> closes per symbol · History depth: <code>{ds.get('max_history_trades', 1000)}</code> trades · Analyzed: <code>{ds.get('total_symbols_analyzed', 0)}</code> markets</p>"
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



async def diag(_: Request) -> JSONResponse:
    """
    Diagnostic endpoint: exposes live bot internals for debugging.
    Shows last proposal/buy errors, offer gate blocks, WS state, account info.
    """
    rt = runtime
    orch = rt.orchestrator
    client = rt.client
    out: dict = {
        "bot_status": rt.status,
        "last_error": rt.last_error,
        "ws_connected": client.connected if client else None,
        "ws_authorized": client.authorized if client else None,
        "api_mode": client.api_mode if client else None,
        "account": {
            "loginid": (client.account or {}).get("loginid"),
            "balance": (client.account or {}).get("balance"),
            "currency": (client.account or {}).get("currency"),
        } if client else {},
    }
    if orch is not None:
        try:
            out["executor_last_error"] = orch.executor.last_error
        except Exception:
            pass
        try:
            out["offer_gate"] = orch.offer_gate.snapshot()
        except Exception:
            pass
        try:
            risk = orch.risk_status()
            out["recent_trades_errors"] = [
                {k: t.get(k) for k in ("status", "symbol", "contract_type", "error", "offer_reason", "ts")}
                for t in (risk.get("recent_trades") or [])
                if t.get("status") in {"failed_offer", "failed", "buy_failed"}
            ][-10:]
        except Exception:
            pass
    return JSONResponse(out)


async def api_system_status(_: Request) -> JSONResponse:
    """Mobile API: System summary overview."""
    st = runtime.public_status()
    risk = st.get("risk") or {}
    return JSONResponse(
        {
            "status": st.get("status"),
            "mode": os.getenv("MODE", "demo"),
            "started_at": st.get("started_at"),
            "last_cycle_at": st.get("last_cycle_at"),
            "balance": risk.get("balance"),
            "currency": risk.get("currency", "USD"),
            "daily_pnl": risk.get("daily_pnl", 0.0),
            "open_trades_count": risk.get("open_trades", 0),
            "trades_today": risk.get("trades_today", 0),
            "paused": risk.get("paused", False),
            "pause_reason": risk.get("pause_reason"),
            "symbols": st.get("symbols", []),
        }
    )


async def api_get_agents(_: Request) -> JSONResponse:
    """Mobile API: Get live agent listing, weights, and status."""
    from src.database.db import db_manager

    states = db_manager.fetch_agent_states()
    if not states and runtime.orchestrator:
        for name in [
            "TrendAgent",
            "VolatilityAgent",
            "PatternAgent",
            "ScalpingAgent",
            "RiskAgent",
            "ExecutionAgent",
            "LearningAgent",
        ]:
            states.append(
                {
                    "agent_name": name,
                    "enabled": True,
                    "status": "running",
                    "weight": 1.0,
                    "win_rate": 50.0,
                }
            )

    return JSONResponse({"agents": states})


async def api_toggle_agent(request: Request) -> JSONResponse:
    """Mobile API: Toggle agent state on/off."""
    agent_name = request.path_params.get("agent_name")
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    enabled = body.get("enabled", True)

    from src.database.db import db_manager

    db_manager.update_agent_state(agent_name, enabled=enabled)

    if runtime.orchestrator and hasattr(runtime.orchestrator, "agent_manager"):
        runtime.orchestrator.agent_manager.set_agent_status(agent_name, enabled)

    return JSONResponse(
        {
            "ok": True,
            "agent_name": agent_name,
            "enabled": enabled,
            "message": f"Agent {agent_name} enabled={enabled}",
        }
    )


async def api_trade_history(request: Request) -> JSONResponse:
    """Mobile API: Fetch historical trade logs."""
    from src.database.db import db_manager

    limit = int(request.query_params.get("limit", "50"))
    trades = db_manager.fetch_recent_trades(limit=limit)

    if not trades and runtime.orchestrator:
        st = runtime.public_status()
        trades = st.get("recent_trades") or []

    return JSONResponse({"trades": trades, "count": len(trades)})


async def api_metrics(_: Request) -> JSONResponse:
    """Mobile API: Performance metrics summary."""
    st = runtime.public_status()
    risk = st.get("risk") or {}
    learning = st.get("learning") or {}
    return JSONResponse(
        {
            "daily_pnl": risk.get("daily_pnl", 0.0),
            "win_rate": learning.get("overall_win_rate", 50.0),
            "total_trades": risk.get("trades_today", 0),
            "consecutive_losses": risk.get("consecutive_losses", 0),
            "max_drawdown": risk.get("max_drawdown", 0.0),
        }
    )


async def api_decision_intelligence(_: Request) -> JSONResponse:
    """REST API: Decision Intelligence, Rejection Funnel, Shadow Trading, & A/B Experiments."""
    if runtime.orchestrator:
        di = runtime.orchestrator.decision_intelligence_status()
    else:
        di = {}
    return JSONResponse(di)


from starlette.endpoints import WebSocketEndpoint
from starlette.websockets import WebSocket


class AndroidDashboardStream(WebSocketEndpoint):
    """Real-time WebSocket stream for Android control dashboard."""

    encoding = "text"

    async def on_connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        logger.info("Android WebSocket client connected: %s", websocket.client)
        # Send initial state snapshot
        st = runtime.public_status()
        await websocket.send_json({"type": "INIT_STATE", "payload": st})

    async def on_receive(self, websocket: WebSocket, data: str) -> None:
        # Echo back heartbeat / handle ping
        if data == "ping":
            await websocket.send_text("pong")

    async def on_disconnect(self, websocket: WebSocket, close_code: int) -> None:
        logger.info("Android WebSocket client disconnected code=%s", close_code)


routes = [
    Route("/", root),
    Route("/health", health),
    Route("/ready", ready),
    Route("/status", status),
    Route("/diag", diag),
    Route("/control/resume", control_resume, methods=["GET", "POST"]),
    Route("/control/pause", control_pause, methods=["GET", "POST"]),
    Route("/control/restart", control_restart, methods=["GET", "POST"]),
    Route("/oauth/login", oauth_login),
    Route("/oauth/callback", oauth_callback),
    Route("/oauth/logout", oauth_logout),
    # Mobile-First REST API routes for Android Dashboard
    Route("/api/v1/system/status", api_system_status, methods=["GET"]),
    Route("/api/v1/agents", api_get_agents, methods=["GET"]),
    Route("/api/v1/agents/{agent_name}/toggle", api_toggle_agent, methods=["POST"]),
    Route("/api/v1/trades/history", api_trade_history, methods=["GET"]),
    Route("/api/v1/metrics", api_metrics, methods=["GET"]),
    Route("/api/v1/decision_intelligence", api_decision_intelligence, methods=["GET"]),
    WebSocketRoute("/api/v1/ws/stream", AndroidDashboardStream),
]

app = Starlette(
    debug=os.getenv("DEBUG", "").lower() in {"1", "true"},
    routes=routes,
    lifespan=lifespan,
)

