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
    pnl_val = float(pnl or 0.0)
    pnl_str = f"{pnl_val:+.2f} USD" if pnl is not None else "0.00 USD"
    pnl_cls = "ok" if pnl_val >= 0 else "bad"
    wr_val = float((s.get("learning") or {}).get("overall_win_rate") or 50.0)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Deriv AI Bot — Institutional Quant Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap" rel="stylesheet">
  <style>
    * {{ box-sizing: border-box; }}
    body {{ font-family: 'Inter', system-ui, sans-serif; max-width: 1280px; margin: 1.5rem auto; padding: 0 1.25rem;
           background: #070b14; color: #e2e8f0; line-height: 1.5; }}
    .card {{ background: linear-gradient(145deg, rgba(15, 23, 42, 0.95), rgba(11, 18, 32, 0.98));
            border-radius: 14px; padding: 1.35rem 1.6rem; margin-bottom: 1.25rem;
            border: 1px solid rgba(56, 189, 248, 0.12);
            box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5); backdrop-filter: blur(12px);
            transition: border-color 0.25s ease, box-shadow 0.25s ease; }}
    .card:hover {{ border-color: rgba(56, 189, 248, 0.25); }}
    h1 {{ margin-top: 0; font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em; color: #f8fafc; display:flex; align-items:center; gap:0.5rem; }}
    h2 {{ margin: 0 0 0.75rem; font-size: 1.1rem; font-weight: 600; color: #f8fafc; letter-spacing: -0.01em; }}
    .ok {{ color: #34d399; }}
    .bad {{ color: #f43f5e; }}
    .warn {{ color: #fbbf24; }}
    .muted {{ color: #94a3b8; }}
    a {{ color: #38bdf8; text-decoration: none; transition: color 0.15s ease; }}
    a:hover {{ color: #7dd3fc; text-decoration: underline; }}
    code {{ background: #090d16; padding: 0.15rem 0.4rem; border-radius: 5px; font-family: 'JetBrains Mono', monospace; font-size: 0.85em; color: #f1f5f9; border: 1px solid rgba(255,255,255,0.06); }}
    ul {{ margin: 0.4rem 0 0; padding-left: 1.2rem; }}
    li {{ margin: 0.25rem 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.85rem; }}
    .grid3 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 0.85rem; }}
    .stat {{ background: #090d16; border: 1px solid rgba(56, 189, 248, 0.12); border-radius: 10px; padding: 0.85rem 1rem; }}
    .stat .label {{ font-size: 0.75rem; color: #94a3b8; font-weight: 500; text-transform: uppercase; letter-spacing: 0.04em; }}
    .stat .val {{ font-size: 1.3rem; font-weight: 700; margin-top: 0.25rem; font-family: 'JetBrains Mono', monospace; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
    th {{ text-align: left; color: #94a3b8; font-weight: 600; padding: 0.6rem 0.45rem; border-bottom: 1px solid rgba(255,255,255,0.08); text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.05em; }}
    td {{ padding: 0.65rem 0.45rem; border-bottom: 1px solid rgba(255,255,255,0.04); vertical-align: middle; }}
    tr:hover td {{ background: rgba(56,189,248,.04); }}
    .btnrow {{ display:flex; flex-wrap:wrap; gap:0.75rem; align-items:center; }}
    .btn {{ color:#fff !important; padding:0.6rem 1.2rem; border-radius:8px; text-decoration:none; font-weight:600; font-size:0.88rem; display:inline-flex; align-items:center; gap:0.4rem; transition: transform 0.15s ease, filter 0.15s ease; }}
    .btn:hover {{ transform: translateY(-1px); filter: brightness(1.1); text-decoration:none; }}
    .btn-go {{ background: linear-gradient(135deg, #059669, #10b981); box-shadow: 0 4px 12px rgba(16,185,129,0.3); }}
    .btn-stop {{ background: linear-gradient(135deg, #be123c, #f43f5e); box-shadow: 0 4px 12px rgba(244,63,94,0.3); }}
    .btn-blue {{ background: linear-gradient(135deg, #0284c7, #38bdf8); box-shadow: 0 4px 12px rgba(56,189,248,0.3); }}
    .pill {{ display: inline-flex; align-items: center; gap: 0.35rem; padding: 0.2rem 0.6rem; border-radius: 999px; font-size: 0.72rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; text-transform: uppercase; letter-spacing: 0.05em; }}
    .pill-win {{ background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); box-shadow: 0 0 8px rgba(16, 185, 129, 0.2); }}
    .pill-loss {{ background: rgba(244, 63, 94, 0.15); color: #f43f5e; border: 1px solid rgba(244, 63, 94, 0.4); box-shadow: 0 0 8px rgba(244, 63, 94, 0.2); }}
    .pill-offer {{ background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }}
    .pill-open {{ background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); box-shadow: 0 0 8px rgba(56, 189, 248, 0.25); }}
    .pill-muted {{ background: rgba(148, 163, 184, 0.12); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.2); }}
    .beacon {{ display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #34d399; box-shadow: 0 0 8px #34d399; animation: pulse 1.8s infinite; }}
    @keyframes pulse {{ 0% {{ opacity: 1; transform: scale(1); }} 50% {{ opacity: 0.4; transform: scale(1.2); }} 100% {{ opacity: 1; transform: scale(1); }} }}
    .badge {{ display:inline-block; padding:0.15rem 0.5rem; border-radius:999px; font-size:0.7rem; font-weight:700; font-family:'JetBrains Mono',monospace; }}
    .badge-low {{ background:rgba(244,63,94,0.15); color:#f43f5e; border:1px solid rgba(244,63,94,0.3); }}
    .badge-med {{ background:rgba(245,158,11,0.15); color:#fbbf24; border:1px solid rgba(245,158,11,0.3); }}
    .badge-high {{ background:rgba(16,185,129,0.15); color:#34d399; border:1px solid rgba(16,185,129,0.3); }}
    .badge-block {{ background:rgba(244,63,94,0.2); color:#ff8080; }}
    .badge-warn {{ background:rgba(245,158,11,0.2); color:#fbbf24; }}
    .badge-watch {{ background:rgba(56,189,248,0.2); color:#7eb6ff; }}
    .badge-healthy {{ background:rgba(16,185,129,0.2); color:#a0ffcb; }}
    .panel-pair {{ display:grid; grid-template-columns:1fr 1fr; gap:0.85rem; }}
    @media(max-width:768px) {{ .panel-pair {{ grid-template-columns:1fr; }} }}
  </style>
  <meta http-equiv="refresh" content="30"/>
</head>
<body>
  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem;margin-bottom:1rem">
      <div>
        <h1><span class="beacon"></span> Institutional Quant Terminal</h1>
        <p class="muted" style="margin:0.25rem 0 0;font-size:0.85rem">
           Deriv Multi-Agent AI Engine · API <code style="color:#38bdf8">{DERIV_API_MODE}</code>
           · stake <code style="color:#38bdf8">{s.get('stake_mode') or 'flat'}</code>
           · horizon <code style="color:#38bdf8">{'on' if s.get('enable_minute') else 'off'} ({s.get('minute_duration') or 2}m)</code>
        </p>
      </div>
      <div>
        <span class="pill pill-open" style="font-size:0.8rem">AUTO-REFRESH 15S</span>
      </div>
    </div>

    <!-- High-Density Metric Hub Cards -->
    <div class="grid" style="grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));">
      <!-- Status Card -->
      <div class="stat">
        <div class="label">System Status</div>
        <div class="val {status_cls}" style="display:flex;align-items:center;gap:0.4rem;margin-top:0.3rem">
          <span class="beacon"></span> {s.get('status').upper()}
        </div>
        <div style="font-size:0.7rem;color:#64748b;margin-top:0.25rem;font-family:'JetBrains Mono',monospace">Active scan loop</div>
      </div>

      <!-- Balance Card -->
      <div class="stat">
        <div class="label">Account Balance</div>
        <div class="val" style="color:#f8fafc">{risk.get('balance')} <span style="font-size:0.85rem;color:#94a3b8">{risk.get('currency') or ''}</span></div>
        <div style="font-size:0.7rem;color:#64748b;margin-top:0.25rem;font-family:'JetBrains Mono',monospace">Live trading capital</div>
      </div>

      <!-- Daily PnL Card with Sparkline -->
      <div class="stat" style="display:flex;flex-direction:column;justify-content:space-between">
        <div>
          <div class="label">Daily Net P&L</div>
          <div class="val {pnl_cls}">{pnl_str}</div>
        </div>
        <div style="margin-top:0.35rem">
          {_svg_sparkline(pnl_val)}
        </div>
      </div>

      <!-- Win Rate Card with Progress Gauge -->
      <div class="stat">
        <div class="label">Overall Win Rate</div>
        <div class="val" style="color:#38bdf8">{wr_val:.1f}%</div>
        <div style="width:100%;background:rgba(255,255,255,0.06);border-radius:999px;height:4px;overflow:hidden;margin-top:0.4rem">
          <div style="width:{min(100, int(wr_val))}%;background:linear-gradient(90deg, #38bdf8, #34d399);height:100%"></div>
        </div>
        <div style="font-size:0.68rem;color:#64748b;margin-top:0.25rem">Target: 65.0%+</div>
      </div>

      <!-- Open Positions Card -->
      <div class="stat">
        <div class="label">Open Positions</div>
        <div class="val" style="color:#f8fafc">{risk.get('open_trades')} <span style="font-size:0.8rem;color:#64748b">/ {risk.get('max_open_trades', 3)} max</span></div>
        <div style="font-size:0.7rem;color:#38bdf8;margin-top:0.25rem;font-family:'JetBrains Mono',monospace">Active portfolio trades</div>
      </div>

      <!-- Risk Drawdown Limit Card -->
      <div class="stat">
        <div class="label">Risk & Drawdown</div>
        <div class="val {'bad' if risk.get('paused') else 'ok'}" style="font-size:1.1rem">
          {'PAUSED' if risk.get('paused') else 'ACTIVE'}
          {(' (' + str(risk.get('pause_remaining_min')) + 'm left)') if risk.get('paused') and risk.get('pause_remaining_min') is not None else ''}
        </div>
        <div style="width:100%;background:rgba(255,255,255,0.06);border-radius:999px;height:4px;overflow:hidden;margin-top:0.4rem">
          <div style="width:{min(100, int(risk.get('consecutive_losses', 0) / 6 * 100))}%;background:#f43f5e;height:100%"></div>
        </div>
        <div style="font-size:0.68rem;color:#64748b;margin-top:0.25rem">Loss streak: {risk.get('consecutive_losses', 0)}/6 max</div>
      </div>
    </div>

    <p style="margin-top:1rem" class="muted">Active Assets: <code>{symbols}</code></p>
    <p class="muted">Started: {s.get('started_at') or '—'} · Last cycle: {s.get('last_cycle_at') or '—'}</p>
    <p class="muted">Setup bans: {ban_html}</p>
    {err_html}
    {oauth_hint}
  </div>

  <div class="card">
    <h2>🛡️ Portfolio Manager & Infrastructure Health (Stages 5 & 8)</h2>
    <p class="muted">1% risk position sizing, hard capital vetoes, 24/7 home server CPU/RAM/Disk telemetry, and self-healing status.</p>
    {_fmt_portfolio_and_infra_panel(s)}
  </div>

  <div class="card">
    <h2>🔬 Decision Audit, Shadow Trading & Gate Effectiveness Subsystem</h2>
    <p class="muted">Scientific pipeline decision tracing, rejection funnel analytics, shadow trade tracking, and CONTROL vs CHALLENGER A/B testing.</p>
    {_fmt_decision_intelligence_panel(s)}
  </div>

  <div class="card">
    <h2>🏆 Chief Strategy Meta-Agent Leaderboard & RL Engine (Stages 1-4)</h2>
    <p class="muted">Live agent accuracy over last 100 trades, dynamic influence weighting, auto-promotions, and Q-Learning RL state.</p>
    {_fmt_meta_agent_panel(s)}
  </div>


  <div class="card">
    <h2>🤖 Active Intelligence Agents ("The CEO & Specialists")</h2>
    <p class="muted">Independent micro-agent specialists evaluating market ticks and voting via Redis ensemble consensus.</p>
    {_fmt_agents_panel(s)}
  </div>


  <div class="card">
    <h2>🌐 Dedicated Market Sub-Agents (20 Active Watchers)</h2>
    <p class="muted">Sub-agents actively scanning ticks, session readiness, and setup opportunities across all 20 configured assets.</p>
    {_fmt_market_watchers_panel(s)}
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
    <p class="muted">Per-market deep analysis triggers every 100 closed trades per symbol. Reads up to <b>1000</b> most-recent trades from full GCS trade history for accuracy.</p>
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

