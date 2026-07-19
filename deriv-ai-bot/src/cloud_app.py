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
from typing import Dict, Optional
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
    # Config health (no secrets leaked)
    import os as _os
    from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_ENABLED, DEEPSEEK_MODEL

    # Merge runtime advisor snapshot (recommendation / errors) with key config
    runtime_ds = body.get("deepseek") if isinstance(body.get("deepseek"), dict) else {}
    body["deepseek"] = {
        **runtime_ds,
        "enabled": bool(runtime_ds.get("enabled", DEEPSEEK_ENABLED)),
        "configured": bool(DEEPSEEK_API_KEY) or bool(runtime_ds.get("ready")),
        "model": runtime_ds.get("model") or DEEPSEEK_MODEL,
        "key_prefix": (str(DEEPSEEK_API_KEY)[:6] + "…") if DEEPSEEK_API_KEY else runtime_ds.get("key_prefix"),
        "ready": bool(runtime_ds.get("ready")),
        "last_error": runtime_ds.get("last_error"),
        "recommendation": runtime_ds.get("recommendation"),
        "type_multipliers": runtime_ds.get("type_multipliers") or {},
        "analyze_every": runtime_ds.get("analyze_every"),
        "closes_since_analysis": runtime_ds.get("closes_since_analysis"),
    }
    body["analytics_config"] = {
        "gate": _os.getenv("ANALYTICS_GATE", "true"),
        "min_sample": _os.getenv("MIN_SAMPLE_SIZE", "500"),
        "min_pattern": _os.getenv("MIN_PATTERN_STRENGTH", "75"),
        "min_clarity": _os.getenv("MIN_PATTERN_CLARITY", "80"),
        "min_edge": _os.getenv("MIN_EDGE_SCORE", "80"),
        "min_live_edge": _os.getenv("MIN_LIVE_EDGE", "80"),
    }
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


def _parse_float(raw: Optional[str], default: Optional[float] = None) -> Optional[float]:
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


async def control_risk(request: Request):
    """
    Update stake / session stop-loss / profit target from the dashboard.

    GET or POST /control/risk?base_stake=1&stop_loss_pct=5&target_rr=3&max_stake_pct=1.5
    Optional: reset_session=1 to start a fresh PnL run from current balance.
    """
    orch = runtime.orchestrator
    if orch is None:
        return HTMLResponse(
            "<h1>Bot not ready</h1><p><a href='/'>Home</a></p>", status_code=503
        )
    # Merge query + form body
    params = dict(request.query_params)
    if request.method == "POST":
        try:
            form = await request.form()
            for k, v in form.items():
                params[str(k)] = str(v)
        except Exception:
            pass
        try:
            body = await request.json()
            if isinstance(body, dict):
                params.update({str(k): v for k, v in body.items()})
        except Exception:
            pass

    base_stake = _parse_float(params.get("base_stake"))
    stop_loss_pct = _parse_float(params.get("stop_loss_pct"))
    target_rr = _parse_float(params.get("target_rr"))
    max_stake_pct = _parse_float(params.get("max_stake_pct"))
    stop_on_raw = params.get("stop_on_target")
    stop_on_target = None
    if stop_on_raw is not None and str(stop_on_raw).strip() != "":
        stop_on_target = str(stop_on_raw).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    reset_session = str(params.get("reset_session") or "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    st = orch.configure_risk_ui(
        base_stake=base_stake,
        stop_loss_pct=stop_loss_pct,
        target_rr=target_rr,
        max_stake_pct=max_stake_pct,
        stop_on_target=stop_on_target,
        reset_session=reset_session,
    )
    want_json = "application/json" in (request.headers.get("accept") or "")
    if want_json or str(params.get("format") or "") == "json":
        return JSONResponse({"ok": True, "action": "risk_update", "risk": st})

    return HTMLResponse(
        f"""<!doctype html><html><body style="font-family:system-ui;background:#0b1220;color:#e8eefc;padding:2rem">
        <h1>⚙️ Risk settings updated</h1>
        <p>Stop-loss: <b>{st.get('session_stop_loss_pct')}%</b>
           (~{st.get('session_stop_loss_amount')}) ·
           Target 1:{st.get('session_target_rr')}
           (~{st.get('session_target_amount')})</p>
        <p>Base stake: <b>{st.get('base_stake')}</b> ·
           Max stake %: <b>{st.get('max_stake_pct')}</b></p>
        <p><a href="/" style="color:#7eb6ff">Dashboard</a></p>
        <meta http-equiv="refresh" content="2;url=/"/>
        </body></html>"""
    )


async def control_deepseek_analyze(request: Request):
    """Run DeepSeek analysis now (requires DEEPSEEK_API_KEY)."""
    orch = runtime.orchestrator
    if orch is None:
        return JSONResponse({"ok": False, "error": "bot_not_ready"}, status_code=503)
    rec = orch._run_deepseek_analysis(source="dashboard")
    want_json = "application/json" in (request.headers.get("accept") or "")
    if want_json or request.query_params.get("format") == "json":
        return JSONResponse(
            {
                "ok": rec is not None,
                "recommendation": rec,
                "deepseek": orch.deepseek.snapshot(),
                "error": orch.deepseek.last_error,
                "n_trades_sent": (rec or {}).get("_meta", {}).get("n_trades"),
            }
        )
    # Always land back on dashboard with flash query so user sees result
    if rec is None:
        err = orch.deepseek.last_error or "no recommendation"
        from urllib.parse import quote

        return RedirectResponse(
            url=f"/?ds=fail&ds_err={quote(str(err)[:200])}",
            status_code=303,
        )
    return RedirectResponse(url="/?ds=ok", status_code=303)


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


def _fmt_trade_rows(trades: list, *, open_mode: bool = False) -> str:
    if not trades:
        return (
            "<tr><td colspan='8' class='muted' style='text-align:center;padding:1rem'>"
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
        profit_s = "—" if profit is None else f"{float(profit):+.2f}"
        p_cls = "ok" if profit is not None and float(profit) > 0 else (
            "bad" if profit is not None and float(profit) < 0 else "muted"
        )
        conf = t.get("confidence")
        conf_s = f"{float(conf):.0%}" if conf is not None else "—"
        barrier = t.get("barrier")
        bar_s = "—" if barrier is None else str(barrier)
        dur = t.get("duration")
        du = t.get("duration_unit") or ""
        dur_s = f"{dur}{du}" if dur is not None else (t.get("horizon") or "—")
        fam = t.get("family") or "—"
        ts = t.get("closed_at") or t.get("opened_at") or t.get("ts") or "—"
        if isinstance(ts, str) and "T" in ts:
            ts = ts.replace("T", " ")[:19]
        rows.append(
            f"<tr>"
            f"<td class='{st_cls}'><b>{st.upper()}</b></td>"
            f"<td><code>{t.get('symbol') or '—'}</code></td>"
            f"<td>{t.get('contract_type') or '—'}</td>"
            f"<td>{bar_s}</td>"
            f"<td>{t.get('stake') if t.get('stake') is not None else '—'}</td>"
            f"<td class='{p_cls}'>{profit_s}</td>"
            f"<td>{conf_s}<br/><span class='muted' style='font-size:0.75rem'>{fam} · {dur_s}</span></td>"
            f"<td class='muted' style='font-size:0.8rem'>{ts}<br/>"
            f"<span class='muted'>#{t.get('contract_id') or '—'}</span></td>"
            f"</tr>"
        )
    return "".join(rows)


async def root(request: Request) -> HTMLResponse:
    s = runtime.public_status()
    risk = s.get("risk") or {}
    status_cls = "ok" if s.get("status") == "running" else "bad"
    # Market + contract selectors for multi-card views
    tracked = list(s.get("symbols") or [])
    q_market = (request.query_params.get("market") or request.query_params.get("symbol") or "").strip()
    q_contract = (request.query_params.get("contract") or "").strip().upper()
    if q_market and q_market in tracked:
        selected_market = q_market
    elif tracked:
        selected_market = tracked[0]
    else:
        selected_market = "R_25"
    hpp_contracts = ["DIGITDIFF", "DIGITMATCH", "DIGITEVEN", "DIGITODD", "DIGITOVER", "DIGITUNDER", "CALL", "PUT"]
    if q_contract and q_contract in hpp_contracts:
        selected_contract = q_contract
    else:
        selected_contract = "DIGITDIFF"
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
    ds = s.get("deepseek") or {}
    # Prefer orchestrator snapshot if public_status nested it under risk/learning paths
    if runtime.orchestrator is not None:
        try:
            live_ds = runtime.orchestrator.deepseek.snapshot()
            if live_ds:
                ds = {**ds, **live_ds}
        except Exception:
            pass
    ds_rec = ds.get("recommendation") or {}
    ds_summary = ds_rec.get("summary") or (
        "not run yet — click Analyze (works with 0 trades; better after ~10–20 closes)"
        if not ds.get("ready")
        else "ready — click Analyze"
    )
    ds_score = ds_rec.get("risk_score")
    ds_ready = "ready" if (ds.get("ready") or ds.get("configured")) else "no API key"
    ds_err = ds.get("last_error")
    ds_flash = (request.query_params.get("ds") or "").strip().lower()
    ds_flash_err = (request.query_params.get("ds_err") or "").strip()
    ds_n = (ds_rec.get("_meta") or {}).get("n_trades")
    ds_hints = ds_rec.get("learning_hints") or ds_rec.get("strategy_changes") or []
    ds_hints_html = "".join(f"<li>{h}</li>" for h in list(ds_hints)[:6])
    ds_types = ds_rec.get("trade_type_analysis") or []
    ds_types_html = "".join(
        f"<li><code>{(t or {}).get('symbol') or '—'}</code> "
        f"{(t or {}).get('contract_type')} · <b>{(t or {}).get('verdict')}</b> — "
        f"{(t or {}).get('reason')}</li>"
        for t in ds_types[:8]
        if isinstance(t, dict)
    )
    ds_banner = ""
    if ds_flash == "ok":
        ds_banner = (
            "<div class='card' style='border-color:#1f6f4a'>"
            "<p class='ok'><b>DeepSeek analysis complete.</b> Results are on this page below.</p></div>"
        )
    elif ds_flash == "fail":
        ds_banner = (
            f"<div class='card' style='border-color:#6b2d2d'>"
            f"<p class='bad'><b>DeepSeek failed:</b> {ds_flash_err or ds_err or 'unknown'}</p></div>"
        )
    stop_hit = risk.get("session_stop_hit")
    tgt_hit = risk.get("session_target_hit")
    analytics = s.get("analytics") or {}
    filt = analytics.get("last_filter") or {}
    scan = analytics.get("edge_scan") or {}
    ranked = scan.get("ranked") or []
    heatmaps = analytics.get("digit_heatmaps") or {}
    probs = analytics.get("probability") or {}
    sess = analytics.get("session") or {}
    mg = analytics.get("martingale_safety") or {}
    risk_sug = analytics.get("risk_suggestion") or {}
    market_book = analytics.get("market_book") or {}
    mb_profiles = market_book.get("profiles") or {}
    mb_cats = market_book.get("categories") or []
    # Per-selected-market category line
    sel_prof = mb_profiles.get(selected_market) or {}
    market_book_html = (
        f"<p class='muted'>Selected <code>{selected_market}</code> → "
        f"<b>{sel_prof.get('label') or sel_prof.get('category') or '—'}</b> · "
        f"path <code>{sel_prof.get('scoring_path') or '—'}</code> · "
        f"metrics: {', '.join(sel_prof.get('primary_metrics') or []) or '—'}</p>"
        f"<p class='muted'>Allowed contracts: "
        f"{', '.join(f'<code>{c}</code>' for c in (sel_prof.get('allowed_contracts') or [])) or '—'}</p>"
        "<table><thead><tr><th>Category</th><th>Path</th><th>Primary metrics</th><th>Contracts</th></tr></thead><tbody>"
        + (
            "".join(
                f"<tr><td>{c.get('label')}</td><td><code>{c.get('path')}</code></td>"
                f"<td class='muted' style='font-size:0.8rem'>{c.get('metrics')}</td>"
                f"<td class='muted' style='font-size:0.8rem'>{c.get('contracts')}</td></tr>"
                for c in mb_cats
            )
            or "<tr><td colspan='4' class='muted'>Category table loads after first status cycle.</td></tr>"
        )
        + "</tbody></table>"
    )
    # Build edge rank rows (highlight selected market) — self-optimizing scanner
    rank_rows = "".join(
        f"<tr class=\"{'sel' if r.get('symbol')==selected_market else ''}\">"
        f"<td>{i+1}</td><td><code>{r.get('symbol')}</code></td>"
        f"<td class='muted' style='font-size:0.75rem'>{r.get('category') or '—'}</td>"
        f"<td><b>{r.get('tier') or '—'}</b></td>"
        f"<td>{r.get('best_type') or '—'}</td>"
        f"<td><b>{r.get('opportunity_score') or r.get('score')}</b></td>"
        f"<td>{r.get('opportunity_velocity') if r.get('opportunity_velocity') is not None else '—'}</td>"
        f"<td>{r.get('pattern_clarity') or r.get('pattern_strength')}</td>"
        f"<td class=\"{'ok' if r.get('tradeable') or r.get('allow') else 'muted'}\">"
        f"{'TRADE' if r.get('tradeable') else (r.get('recommendation') or '—')}</td></tr>"
        for i, r in enumerate(ranked[:10])
    ) or "<tr><td colspan='9' class='muted'>Waiting for ticks…</td></tr>"
    scan_display = scan.get("display") or []
    scan_lines = "".join(f"<li><code>{ln}</code></li>" for ln in scan_display[:8])
    pri = scan.get("priority_book") or {}
    scan_report = (scan.get("last_report") or {}).get("display") or ""
    tiers = scan.get("tiers") or {}
    tier_html = (
        f"<p class='muted'>ELITE: {', '.join(tiers.get('ELITE') or ['—'])} · "
        f"STRONG: {', '.join(tiers.get('STRONG') or ['—'])} · "
        f"WATCH: {', '.join(tiers.get('WATCHLIST') or ['—'])} · "
        f"IGNORE: {', '.join((tiers.get('IGNORE') or [])[:4]) or '—'}</p>"
    )
    # Heatmap for selected market
    heat_html = "<p class='muted'>No digit data yet for this market.</p>"
    snap = heatmaps.get(selected_market) if heatmaps else None
    if not snap and heatmaps:
        # fallback first available
        snap = heatmaps.get(next(iter(heatmaps)))
    if snap:
        w100 = ((snap.get("heatmap") or {}).get("windows") or {}).get("100") or {}
        table = w100.get("table") or []
        heat_rows = "".join(
            f"<tr><td>{row.get('digit')}</td><td>{row.get('pct')}%</td>"
            f"<td>{row.get('count')}</td></tr>"
            for row in table
        )
        heat_html = (
            f"<p class='muted'><code>{selected_market}</code> · last 100 ticks · "
            f"hot {w100.get('hot')} · cold {w100.get('cold')}</p>"
            f"<table><thead><tr><th>Digit</th><th>Share</th><th>Count</th></tr></thead>"
            f"<tbody>{heat_rows}</tbody></table>"
        )
    # Probability table for selected market
    prob_html = "<p class='muted'>No probability table yet for this market.</p>"
    prow_src = (probs.get(selected_market) if probs else None) or (
        probs.get(next(iter(probs))) if probs else None
    )
    if prow_src:
        prow = (prow_src.get("rows") or [])[:8]
        prob_rows = "".join(
            f"<tr><td>{r.get('trade_type')}</td>"
            f"<td>{(float(r.get('confidence') or 0)*100):.0f}%</td></tr>"
            for r in prow
        )
        prob_html = (
            f"<p class='muted'><code>{selected_market}</code></p>"
            f"<table><thead><tr><th>Trade type</th><th>Confidence</th></tr></thead>"
            f"<tbody>{prob_rows}</tbody></table>"
        )
    # Market selector HTML options
    market_opts = "".join(
        f"<option value=\"{m}\" {'selected' if m==selected_market else ''}>{m}</option>"
        for m in (tracked or [selected_market])
    )
    contract_opts = "".join(
        f"<option value=\"{c}\" {'selected' if c==selected_contract else ''}>{c}</option>"
        for c in hpp_contracts
    )
    # Strategy detail for selected market
    strat_sel = (strats.get(selected_market) if strats else None) or {}
    strat_types = strat_sel.get("allowed_types") or []
    sel_cat = (mb_profiles.get(selected_market) or {}) if mb_profiles else {}
    # mb_profiles may not exist yet if market_book_html block order differs — recompute safely
    try:
        from src.strategy.market_categories import market_profile as _mp

        sel_cat = _mp(selected_market)
    except Exception:
        sel_cat = sel_cat or {}
    strat_detail = (
        f"<p class='muted'><code>{selected_market}</code> · category <b>{sel_cat.get('label') or '—'}</b> · "
        f"path <code>{sel_cat.get('scoring_path') or '—'}</code> · "
        f"type <b>{strat_sel.get('type') or '—'}</b> · "
        f"tradeable <b>{'yes' if strat_sel.get('tradeable') else 'no'}</b></p>"
        f"<p class='muted'>Engine metrics: {', '.join(sel_cat.get('primary_metrics') or []) or '—'}</p>"
        f"<p class='muted'>Contracts: {', '.join(f'<code>{t}</code>' for t in strat_types) or '—'}</p>"
    )
    filt_rec = filt.get("recommendation") or "—"
    filt_cond = filt.get("market_condition") or "—"
    filt_edge = filt.get("expected_edge") or "—"
    filt_copilot = filt.get("copilot") or "Filter runs when a candidate is scored."
    hist_edge = filt.get("historical_edge") or {}
    live_blk = filt.get("live_edge") or {}
    hist_comp = hist_edge.get("components") or {}
    nt = filt.get("no_trade") or {}
    nt_status = nt.get("status") or ("ALLOWED" if filt.get("allow") else "—")
    nt_reason = nt.get("reason") or "—"
    nt_ev = filt.get("ev") if filt.get("ev") is not None else nt.get("ev")
    nt_dq = filt.get("decision_quality")
    if nt_dq is None:
        nt_dq = (nt.get("trade_quality") or {})
        if isinstance(nt_dq, dict):
            nt_dq = nt_dq.get("trade_quality")
    nt_regime = filt.get("regime") or nt.get("regime") or "—"
    nt_risk = filt.get("risk_pct") if filt.get("risk_pct") is not None else nt.get("risk_pct")
    filt_reasons = "".join(
        f"<li>{r}</li>" for r in (filt.get("explain") or filt.get("reasons") or [])[:12]
    )
    edge_breakdown = (
        f"<p class='muted'>EV {hist_comp.get('ev', '—')}/40 · "
        f"WR {hist_comp.get('win_rate', '—')}/20 · "
        f"PF {hist_comp.get('profit_factor', '—')}/20 · "
        f"DD {hist_comp.get('drawdown', '—')}/20</p>"
        f"<p><b>Edge Score: {hist_edge.get('edge_score', '—')}</b> "
        f"({hist_edge.get('label', '—')}) · "
        f"<b>LIVE EDGE: {live_blk.get('live_edge', '—')}</b> · "
        f"{live_blk.get('status', '—')} · Conf {live_blk.get('confidence', '—')} · "
        f"EV {live_blk.get('expected_value', '—')} · Risk {live_blk.get('risk', '—')}</p>"
    )
    sess_lines = "".join(f"<li>{ln}</li>" for ln in (sess.get("lines") or [])[:6]) or (
        "<li class='muted'>Need ~10+ trades for hour/index insights.</li>"
    )
    mg_levels = "".join(
        f"<tr><td>L{lv.get('level')}</td><td>{lv.get('stake')}</td>"
        f"<td>{lv.get('cumulative_risk')}</td></tr>"
        for lv in (mg.get("ladder") or [])[:5]
    )
    risk_sug_html = ""
    if risk_sug:
        risk_sug_html = (
            f"<p class='bad'><b>{risk_sug.get('current_risk')}</b> — "
            f"{risk_sug.get('message')}</p>"
            f"<p class='muted'>Adaptive risk: "
            f"{(risk_sug.get('risk_plan') or {}).get('risk_pct')}% "
            f"({(risk_sug.get('risk_plan') or {}).get('reason')})</p>"
        )
    # Rolling entropy regime card (selected market)
    roll_map = analytics.get("rolling_entropy") or {}
    roll0 = {}
    roll_sym = selected_market
    if roll_map.get(selected_market):
        roll0 = roll_map[selected_market] or {}
    elif roll_map:
        roll_sym = next(iter(roll_map))
        roll0 = roll_map[roll_sym] or {}
    elif (filt.get("rolling_entropy") or {}):
        roll0 = filt.get("rolling_entropy") or {}
        roll_sym = filt.get("symbol") or selected_market
    prim = roll0.get("primary") or {}
    roll_windows = roll0.get("windows") or {}
    win_rows = "".join(
        f"<tr><td>{k}</td><td>{(v or {}).get('h')}</td>"
        f"<td>{(v or {}).get('compression_pct')}%</td>"
        f"<td>{(v or {}).get('velocity')}</td>"
        f"<td>{(v or {}).get('regime')}</td></tr>"
        for k, v in sorted(roll_windows.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0)
    ) or "<tr><td colspan='5' class='muted'>Waiting for ticks…</td></tr>"

    # ----- HPP time series dashboard (selected contract) -----
    hpp_bundle = analytics.get("hpp_timeseries") or {}
    hpp_boards = hpp_bundle.get("contracts") or {}
    hpp_primary = hpp_boards.get(selected_contract) or hpp_bundle.get("primary") or {}
    if not hpp_primary and hpp_boards:
        hpp_primary = next(iter(hpp_boards.values()), {}) or {}
    hpp_ct = hpp_primary.get("contract") or selected_contract
    meta = hpp_primary.get("meta") or {}
    series = hpp_primary.get("series") or {}
    hpp_vals = series.get("hpp") or []
    hpp_days = series.get("days") or []
    # Sparkline as simple bars
    spark = ""
    if hpp_vals:
        mx = max(hpp_vals) or 1
        spark = "".join(
            f"<span title='{d}: {v}' style='display:inline-block;width:8px;"
            f"height:{max(4, int(v/mx*40))}px;background:#7eb6ff;"
            f"margin:0 1px;vertical-align:bottom;border-radius:2px'></span>"
            for d, v in list(zip(hpp_days, hpp_vals))[-30:]
        )
    metric_rows = "".join(
        f"<tr><td>{m.get('metric')}</td><td><b>{m.get('hpp')}</b></td>"
        f"<td class=\"{'ok' if (m.get('delta') or 0) > 0 else 'bad' if (m.get('delta') or 0) < 0 else 'muted'}\">"
        f"{m.get('arrow')} {m.get('delta')}</td>"
        f"<td class='muted' style='font-size:0.75rem'>{m.get('status') or ''}<br/>"
        f"S {m.get('short')} · M {m.get('medium')} · L {m.get('long')}</td></tr>"
        for m in (hpp_primary.get("metrics") or [])[:8]
    ) or "<tr><td colspan='4' class='muted'>HPP builds after closed trades with metrics.</td></tr>"
    win_tbl = "".join(
        f"<tr><td>{r.get('metric')}</td><td>{r.get('short')}</td>"
        f"<td>{r.get('mid')}</td><td>{r.get('long')}</td>"
        f"<td>{r.get('interpretation')}</td></tr>"
        for r in ((hpp_primary.get("windows") or {}).get("rows") or [])
    ) or ""
    wf = hpp_primary.get("waterfall") or {}
    wf_steps = "".join(
        f"<tr><td>{s.get('metric')}</td>"
        f"<td class=\"{'ok' if (s.get('delta') or 0) >= 0 else 'bad'}\">"
        f"{s.get('delta'):+.1f}</td></tr>"
        for s in (wf.get("steps") or [])[:8]
    )
    heat = hpp_primary.get("heatmap") or {}
    heat_months = heat.get("months") or []
    heat_head = "".join(f"<th>{m}</th>" for m in heat_months)
    heat_body = ""
    for row in (heat.get("rows") or [])[:6]:
        cells = ""
        for mo in heat_months:
            cell = (row.get("months") or {}).get(mo) or {}
            band = cell.get("band") or "none"
            color = {
                "strong": "#1f6f4a",
                "neutral": "#2a3f6b",
                "weak": "#6b2d2d",
                "none": "#0b1220",
            }.get(band, "#0b1220")
            cells += (
                f"<td style='background:{color};text-align:center'>"
                f"{cell.get('hpp') if cell.get('hpp') is not None else '—'}</td>"
            )
        heat_body += f"<tr><td>{row.get('metric')}</td>{cells}</tr>"
    radar = (hpp_primary.get("radar") or {}).get("axes") or {}
    radar_html = "".join(
        f"<div class='stat'><div class='label'>{k}</div>"
        f"<div class='val'>{v}</div>"
        f"<div class='bar'><i style='width:{min(100, float(v or 0))}%'></i></div></div>"
        for k, v in radar.items()
    )
    lifecycle = hpp_primary.get("lifecycle") or meta.get("lifecycle") or "—"
    meta_status = meta.get("status") or hpp_primary.get("trend") or "—"

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Deriv AI Bot</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 960px; margin: 1.5rem auto; padding: 0 1rem;
           background: #0b1220; color: #e8eefc; }}
    .card {{ background: #151e33; border-radius: 12px; padding: 1.25rem 1.5rem; margin-bottom: 1rem;
             box-shadow: 0 4px 24px rgba(0,0,0,.25); }}
    h1 {{ margin-top: 0; font-size: 1.45rem; }}
    h2 {{ margin: 0 0 0.75rem; font-size: 1.05rem; }}
    .ok {{ color: #3ddc97; }}
    .bad {{ color: #ff6b6b; }}
    .muted {{ color: #9bb0d3; }}
    a {{ color: #7eb6ff; }}
    code {{ background: #0b1220; padding: 0.1rem 0.35rem; border-radius: 4px; font-size: 0.9em; }}
    ul {{ margin: 0.4rem 0 0; padding-left: 1.2rem; }}
    li {{ margin: 0.25rem 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0.75rem; }}
    .stat {{ background: #0b1220; border-radius: 8px; padding: 0.75rem; }}
    .stat .label {{ font-size: 0.75rem; color: #9bb0d3; }}
    .stat .val {{ font-size: 1.15rem; font-weight: 650; margin-top: 0.2rem; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
    th {{ text-align: left; color: #9bb0d3; font-weight: 600; padding: 0.45rem 0.35rem;
         border-bottom: 1px solid #243049; }}
    td {{ padding: 0.5rem 0.35rem; border-bottom: 1px solid #1a2438; vertical-align: top; }}
    tr:hover td {{ background: rgba(126,182,255,.06); }}
    .btnrow {{ display:flex; flex-wrap:wrap; gap:0.65rem; align-items:center; }}
    tr.sel {{ background: rgba(126,182,255,0.12); }}
    .market-bar {{ display:flex; flex-wrap:wrap; gap:0.75rem 1.25rem; align-items:flex-end;
      background:#101827; border:1px solid #24304d; border-radius:12px; padding:0.9rem 1.1rem; margin-bottom:1rem; }}
    .market-bar label {{ display:block; font-size:0.72rem; color:#9bb0d3; margin-bottom:0.25rem; }}
    .market-bar select {{ background:#0b1220; color:#e8eefc; border:1px solid #2a3f6b; border-radius:8px;
      padding:0.45rem 0.65rem; min-width:9rem; font-size:0.95rem; }}
    .market-bar .hint {{ color:#9bb0d3; font-size:0.8rem; flex:1; min-width:12rem; }}
    .pill {{ display:inline-block; background:#1a2744; color:#9bb0d3; border-radius:999px;
      padding:0.15rem 0.55rem; font-size:0.75rem; margin-right:0.25rem; }}
    .pill.on {{ background:#1f4d7a; color:#dcecff; }}
    .btn {{ color:#fff !important; padding:0.55rem 1rem; border-radius:8px; text-decoration:none; font-weight:600;
            border:none; cursor:pointer; font-size:0.95rem; }}
    .btn-go {{ background:#1f6f4a; }}
    .btn-stop {{ background:#6b2d2d; }}
    .btn-blue {{ background:#2a3f6b; }}
    .btn-purple {{ background:#4a2f7a; }}
    label {{ display:block; font-size:0.78rem; color:#9bb0d3; margin-bottom:0.25rem; }}
    input, select {{ width:100%; box-sizing:border-box; background:#0b1220; color:#e8eefc;
                     border:1px solid #243049; border-radius:8px; padding:0.5rem 0.6rem; }}
    .formgrid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap:0.75rem; }}
    .bar {{ height:8px; background:#0b1220; border-radius:99px; overflow:hidden; margin-top:0.35rem; }}
    .bar > i {{ display:block; height:100%; background:#3ddc97; }}
    .bar.bad > i {{ background:#ff6b6b; }}
    tr.sel {{ background: rgba(126,182,255,0.12); }}
    .market-bar {{ display:flex; flex-wrap:wrap; gap:0.75rem 1.25rem; align-items:flex-end;
      background:#101827; border:1px solid #24304d; border-radius:12px; padding:0.9rem 1.1rem; margin-bottom:1rem; }}
    .market-bar label {{ display:block; font-size:0.72rem; color:#9bb0d3; margin-bottom:0.25rem; }}
    .market-bar select {{ background:#0b1220; color:#e8eefc; border:1px solid #2a3f6b; border-radius:8px;
      padding:0.45rem 0.65rem; min-width:9rem; font-size:0.95rem; width:auto; }}
    .market-bar .hint {{ color:#9bb0d3; font-size:0.8rem; flex:1; min-width:12rem; }}
    .pill {{ display:inline-block; background:#1a2744; color:#9bb0d3; border-radius:999px;
      padding:0.15rem 0.55rem; font-size:0.75rem; margin-right:0.25rem; }}
    .pill.on {{ background:#1f4d7a; color:#dcecff; }}
  </style>
  <meta http-equiv="refresh" content="15;url=/?market={selected_market}&amp;contract={selected_contract}"/>
</head>
<body>
  {ds_banner}
  <div class="market-bar">
    <form method="get" action="/" id="market-form" style="display:flex;flex-wrap:wrap;gap:0.75rem 1.25rem;align-items:flex-end;width:100%">
      <div>
        <label for="market">Tracked market</label>
        <select id="market" name="market" onchange="this.form.submit()">
          {market_opts}
        </select>
      </div>
      <div>
        <label for="contract">HPP contract</label>
        <select id="contract" name="contract" onchange="this.form.submit()">
          {contract_opts}
        </select>
      </div>
      <div class="hint">
        Viewing <b>{selected_market}</b> · HPP <b>{selected_contract}</b><br/>
        Heatmap, probability, entropy &amp; strategy detail follow this market.
        Edge scanner ranks all markets. HPP uses the contract dropdown.
        <div style="margin-top:0.35rem">
          {"".join(f"<span class='pill {'on' if m==selected_market else ''}'>{m}</span>" for m in (tracked or []))}
        </div>
      </div>
    </form>
  </div>

  <div class="card">
    <h1>Deriv AI Bot</h1>
    <p class="muted">Auto-refresh 15s · API <code>{DERIV_API_MODE}</code>
       · stake <code>{s.get('stake_mode') or 'flat'}</code>
       · minutes <code>{'on' if s.get('enable_minute') else 'off'} ({s.get('minute_duration') or 2}m)</code>
       · DeepSeek <code>{ds_ready}</code>
       · market <code>{selected_market}</code>
    </p>
    <div class="grid">
      <div class="stat"><div class="label">Status</div>
        <div class="val {status_cls}">{s.get('status')}</div></div>
      <div class="stat"><div class="label">Balance</div>
        <div class="val">{risk.get('balance')} {risk.get('currency') or ''}</div></div>
      <div class="stat"><div class="label">Session PnL</div>
        <div class="val {pnl_cls}">{risk.get('daily_pnl')}</div></div>
      <div class="stat"><div class="label">Target (1:{risk.get('session_target_rr') or 3})</div>
        <div class="val {'ok' if tgt_hit else ''}">{risk.get('session_target_amount')}
        {' ✓ HIT' if tgt_hit else ''}</div>
        <div class="bar"><i style="width:{risk.get('progress_to_target_pct') or 0}%"></i></div>
      </div>
      <div class="stat"><div class="label">Stop-loss ({risk.get('session_stop_loss_pct')}%)</div>
        <div class="val {'bad' if stop_hit else ''}">{risk.get('session_stop_loss_amount')}
        {' ✕ HIT' if stop_hit else ''}</div>
        <div class="bar bad"><i style="width:{risk.get('progress_to_stop_pct') or 0}%"></i></div>
      </div>
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
    <h2>HPP time series — is the edge growing or dying?</h2>
    <p class="muted">Contract <code>{hpp_ct}</code> (dropdown) · market view <code>{selected_market}</code> · lifecycle <b>{lifecycle}</b> ·
       Meta-HPP <b>{meta.get('meta_hpp', '—')}</b> · {meta_status} ·
       conf {meta.get('confidence', '—')} ·
       weight adj {meta.get('recommended_weight_change_pct', 0):+.0f}%</p>
    <div class="grid">
      <div class="stat"><div class="label">Current HPP</div>
        <div class="val">{hpp_primary.get('current_hpp', '—')}</div></div>
      <div class="stat"><div class="label">HPP Velocity (EMA)</div>
        <div class="val {'ok' if (hpp_primary.get('velocity') or 0) > 0 else 'bad' if (hpp_primary.get('velocity') or 0) < 0 else ''}">
        {hpp_primary.get('arrow', '')} {hpp_primary.get('velocity', '—')}
        <span class="muted" style="font-size:0.75rem"> ({hpp_primary.get('velocity_pct', '—')}%)</span></div>
        <div class="muted" style="font-size:0.75rem">{hpp_primary.get('velocity_status') or ''}</div></div>
      <div class="stat"><div class="label">Effective velocity</div>
        <div class="val">{hpp_primary.get('effective_velocity', '—')}</div>
        <div class="muted" style="font-size:0.75rem">velocity × sample conf</div></div>
      <div class="stat"><div class="label">Acceleration</div>
        <div class="val">{hpp_primary.get('acceleration', '—')}</div></div>
      <div class="stat"><div class="label">Edge flag</div>
        <div class="val">{hpp_primary.get('edge_flag') or '—'}</div></div>
      <div class="stat"><div class="label">Engine velocity</div>
        <div class="val">{(hpp_primary.get('overall_velocity') or {}).get('overall_velocity', '—')}</div>
        <div class="muted" style="font-size:0.75rem">{(hpp_primary.get('overall_velocity') or {}).get('status', '')}</div></div>
    </div>
    <p class="muted" style="margin-top:0.75rem">HPP trend (sparkline)</p>
    <div style="height:48px;display:flex;align-items:flex-end">{spark or "<span class='muted'>Need more closed trades…</span>"}</div>
    <div class="grid" style="margin-top:1rem">
      <div>
        <h2 style="font-size:0.95rem">Contract metrics</h2>
        <table><thead><tr><th>Metric</th><th>HPP</th><th>Vel EMA</th><th>Status / S·M·L</th></tr></thead>
        <tbody>{metric_rows}</tbody></table>
      </div>
      <div>
        <h2 style="font-size:0.95rem">Rolling windows (short / mid / long)</h2>
        <table><thead><tr><th>Metric</th><th>100</th><th>500</th><th>1000</th><th>Read</th></tr></thead>
        <tbody>{win_tbl or "<tr><td colspan='5' class='muted'>—</td></tr>"}</tbody></table>
      </div>
    </div>
    <h2 style="font-size:0.95rem;margin-top:1rem">HPP contribution waterfall</h2>
    <p class="muted">Base {wf.get('base', '—')} → Current {wf.get('current', '—')}
       (Δ {wf.get('total_delta', 0):+.1f})</p>
    <table><thead><tr><th>Driver</th><th>Δ HPP</th></tr></thead>
    <tbody>{wf_steps or "<tr><td colspan='2' class='muted'>Need 2+ snapshots</td></tr>"}</tbody></table>
    <h2 style="font-size:0.95rem;margin-top:1rem">Metric heatmap (month × metric)</h2>
    <div style="overflow-x:auto">
    <table><thead><tr><th>Metric</th>{heat_head}</tr></thead>
    <tbody>{heat_body or "<tr><td class='muted'>No history yet</td></tr>"}</tbody></table>
    </div>
    <h2 style="font-size:0.95rem;margin-top:1rem">Radar axes</h2>
    <div class="grid">{radar_html or "<p class='muted'>—</p>"}</div>
    <p class="muted" style="margin-top:0.75rem">Lifecycle: 85+ Peak · 70–85 Mature · 55–70 Declining · &lt;55 Retire</p>
  </div>

  <div class="card">
    <h2>Entropy regime (rolling) — {selected_market}</h2>
    <p class="muted">Symbol <code>{roll_sym}</code> · updates with tick windows 25/50/100/200/500 · switch market above</p>
    <div class="grid">
      <div class="stat"><div class="label">Market regime</div>
        <div class="val">{roll0.get('regime') or prim.get('regime') or '—'}</div></div>
      <div class="stat"><div class="label">Entropy (short)</div>
        <div class="val">{prim.get('entropy') if prim.get('entropy') is not None else '—'}</div></div>
      <div class="stat"><div class="label">Compression</div>
        <div class="val">{prim.get('compression_pct') if prim.get('compression_pct') is not None else '—'}%</div>
        <div class="muted" style="font-size:0.75rem">{prim.get('bias_label') or ''}</div></div>
      <div class="stat"><div class="label">Momentum (bits)</div>
        <div class="val">{prim.get('momentum_bits') if prim.get('momentum_bits') is not None else '—'}</div></div>
      <div class="stat"><div class="label">RT pattern strength</div>
        <div class="val">{roll0.get('realtime_pattern_strength', '—')}</div></div>
      <div class="stat"><div class="label">Composite entropy</div>
        <div class="val">{roll0.get('composite_entropy', '—')}</div></div>
      <div class="stat"><div class="label">Entropy clarity</div>
        <div class="val">{roll0.get('entropy_clarity', '—')}</div></div>
      <div class="stat"><div class="label">Confidence</div>
        <div class="val">{roll0.get('confidence_label') or roll0.get('confidence') or '—'}</div></div>
    </div>
    <ul style="margin-top:0.75rem">
      {"".join(f"<li>{c}</li>" for c in (roll0.get('contributors') or [])[:8]) or "<li class='muted'>Contributors appear after ticks fill windows.</li>"}
    </ul>
    <div style="overflow-x:auto;margin-top:0.75rem">
    <table>
      <thead><tr><th>Window</th><th>H</th><th>Compression</th><th>Velocity</th><th>Regime</th></tr></thead>
      <tbody>{win_rows}</tbody>
    </table>
    </div>
  </div>

  <div class="card">
    <h2>AI trade filter + No-Trade engine</h2>
    <p class="muted">Not “BUY NOW” — block bad trades even when a signal looks strong. EV &gt; 0 required.</p>
    <div class="grid">
      <div class="stat"><div class="label">Market condition</div>
        <div class="val">{filt_cond}</div></div>
      <div class="stat"><div class="label">Expected edge</div>
        <div class="val">{filt_edge}</div></div>
      <div class="stat"><div class="label">Recommendation</div>
        <div class="val {'ok' if filt_rec == 'Trade' else 'bad' if filt_rec == 'Skip' else ''}">{filt_rec}</div></div>
      <div class="stat"><div class="label">No-Trade status</div>
        <div class="val {'ok' if nt_status == 'ALLOWED' else 'bad' if nt_status == 'REJECTED' else ''}">{nt_status}</div>
        <div class="muted" style="font-size:0.75rem">{nt_reason if nt_status == 'REJECTED' else ''}</div></div>
      <div class="stat"><div class="label">Decision quality</div>
        <div class="val">{nt_dq if nt_dq is not None else '—'}</div></div>
      <div class="stat"><div class="label">EV</div>
        <div class="val">{nt_ev if nt_ev is not None else '—'}</div></div>
      <div class="stat"><div class="label">Regime</div>
        <div class="val">{nt_regime}</div></div>
      <div class="stat"><div class="label">Risk %</div>
        <div class="val">{nt_risk if nt_risk is not None else '—'}</div></div>
      <div class="stat"><div class="label">Live edge</div>
        <div class="val">{(filt.get('live_edge') or {}).get('live_edge', '—')}</div></div>
      <div class="stat"><div class="label">Pattern strength</div>
        <div class="val">{(filt.get('pattern_strength') or {}).get('pattern_strength', '—')}</div></div>
      <div class="stat"><div class="label">Pattern clarity</div>
        <div class="val">{(filt.get('pattern_clarity') or {}).get('pattern_clarity', '—')}
        <span class="muted" style="font-size:0.75rem"> {(filt.get('pattern_clarity') or {}).get('class', '')}</span></div></div>
      <div class="stat"><div class="label">Edge score</div>
        <div class="val">{(filt.get('historical_edge') or {}).get('edge_score', '—')}</div></div>
      <div class="stat"><div class="label">Sample size</div>
        <div class="val">{filt.get('sample_size', '—')}</div></div>
      <div class="stat"><div class="label">Setup quality</div>
        <div class="val">{(filt.get('quality') or {}).get('quality_score', '—')}</div></div>
    </div>
    <p class="muted">Gates: Strength≥75 · Clarity≥80 · Edge≥80 · Live≥80 · Decision TQ≥80 · EV&gt;0 · Ensemble agree · Regime allow · Sample≥500 (prod)</p>
    <p class="muted">Risk sizing: TQ 90+ → 1% · 80–90 → 0.5% · below 80 → 0% (no trade)</p>
    {edge_breakdown}
    <p style="margin-top:0.85rem">{filt_copilot}</p>
    <p class="muted"><b>Why this decision</b> (no-trade reasons + edge explain)</p>
    <ul>{filt_reasons}</ul>
  </div>

  <div class="card">
    <h2>Market Opportunity Ranking (MOR)</h2>
    <p class="muted">Not “is this trade good?” — <b>where is the best edge across all markets right now?</b><br/>
       Score = strength/clarity/HPP/velocity/MP/regime/EV/conf − risk penalties.
       Trade only ELITE/STRONG + Edge≥80 · Clarity≥75 · HPP≥75 · MP≥70 · EV&gt;0.</p>
    {tier_html}
    <div style="overflow-x:auto">
    <table>
      <thead><tr>
        <th>#</th><th>Index</th><th>Category</th><th>Tier</th><th>Type</th>
        <th>Opp score</th><th>Opp vel</th><th>Clarity</th><th>Gate</th>
      </tr></thead>
      <tbody>{rank_rows}</tbody>
    </table>
    </div>
    <ul style="margin-top:0.75rem">{scan_lines or "<li class='muted'>Scanner fills after first status cycle.</li>"}</ul>
    {f"<pre class='muted' style='white-space:pre-wrap;font-size:0.8rem'>{scan_report}</pre>" if scan_report else ""}
  </div>

  <div class="card">
    <h2>Digit frequency heatmap (100 ticks) — {selected_market}</h2>
    <p class="muted">Switch market with the dropdown at the top to inspect each tracked index.</p>
    {heat_html}
  </div>

  <div class="card">
    <h2>Probability engine — {selected_market}</h2>
    {prob_html}
  </div>

  <div class="card">
    <h2>Strategy on {selected_market}</h2>
    {strat_detail}
  </div>

  <div class="card">
    <h2>Risk &amp; martingale safety</h2>
    <div class="grid">
      <div class="stat"><div class="label">Balance</div>
        <div class="val">{risk.get('balance')}</div></div>
      <div class="stat"><div class="label">Daily P/L</div>
        <div class="val {pnl_cls}">{risk.get('daily_pnl')}</div></div>
      <div class="stat"><div class="label">Risk / trade</div>
        <div class="val">{risk.get('max_stake_pct')}%</div></div>
      <div class="stat"><div class="label">MG danger</div>
        <div class="val {'bad' if mg.get('danger_level') in ('HIGH','CRITICAL') else 'ok'}">{mg.get('danger_level') or '—'}</div></div>
      <div class="stat"><div class="label">Survival (ladder)</div>
        <div class="val">{mg.get('survival_pct')}%</div></div>
    </div>
    {risk_sug_html}
    <p class="muted" style="margin-top:0.75rem">{mg.get('recommendation') or ''}</p>
    <table>
      <thead><tr><th>Level</th><th>Stake</th><th>Cumulative risk</th></tr></thead>
      <tbody>{mg_levels or "<tr><td colspan='3' class='muted'>—</td></tr>"}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>Advanced analytics (session recorder)</h2>
    <ul>{sess_lines}</ul>
    <p class="muted">Trades stored: {sess.get('n_trades') or 0} · used to skip weak hours/markets</p>
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
    <h2>Controls</h2>
    <div class="btnrow">
      <a class="btn btn-go" href="/control/resume">▶️ Resume</a>
      <a class="btn btn-stop" href="/control/pause">⏸ Pause</a>
      <a class="btn btn-blue" href="/control/restart">🔄 Restart</a>
      <a class="btn btn-purple" href="/control/deepseek-analyze">🧠 DeepSeek analyze</a>
    </div>
    <p class="muted" style="margin-top:0.75rem">Resume clears risk cooldown + anti-spiral bans. Target/stop-loss flags clear on Resume or Reset session.</p>
  </div>

  <div class="card">
    <h2>Session risk &amp; stake</h2>
    <p class="muted">Risk only <b>1–2%</b> per trade. Session stop-loss is dynamic <b>5–10%</b>.
       Profit target uses standard <b>1:3</b> R:R (target = stop-loss amount × 3). When target is hit the bot stops.</p>
    <form method="GET" action="/control/risk">
      <div class="formgrid">
        <div>
          <label for="base_stake">Base stake</label>
          <input id="base_stake" name="base_stake" type="number" step="0.01" min="0.35"
                 value="{risk.get('base_stake') if risk.get('base_stake') is not None else 1.0}"/>
        </div>
        <div>
          <label for="max_stake_pct">Max stake % of balance (1–2)</label>
          <input id="max_stake_pct" name="max_stake_pct" type="number" step="0.1" min="0.5" max="5"
                 value="{risk.get('max_stake_pct') if risk.get('max_stake_pct') is not None else 1.5}"/>
        </div>
        <div>
          <label for="stop_loss_pct">Session stop-loss % (5–10)</label>
          <input id="stop_loss_pct" name="stop_loss_pct" type="number" step="0.5" min="5" max="10"
                 value="{risk.get('session_stop_loss_pct') if risk.get('session_stop_loss_pct') is not None else 5}"/>
        </div>
        <div>
          <label for="target_rr">Target R:R (default 3 = 1:3)</label>
          <input id="target_rr" name="target_rr" type="number" step="0.5" min="1" max="5"
                 value="{risk.get('session_target_rr') if risk.get('session_target_rr') is not None else 3}"/>
        </div>
        <div>
          <label for="stop_on_target">Stop when target hit</label>
          <select id="stop_on_target" name="stop_on_target">
            <option value="true" {'selected' if risk.get('session_stop_on_target', True) else ''}>Yes</option>
            <option value="false" {'selected' if not risk.get('session_stop_on_target', True) else ''}>No</option>
          </select>
        </div>
        <div>
          <label for="reset_session">Reset session PnL run</label>
          <select id="reset_session" name="reset_session">
            <option value="0" selected>No — keep current PnL</option>
            <option value="1">Yes — fresh run from balance</option>
          </select>
        </div>
      </div>
      <div class="btnrow" style="margin-top:1rem">
        <button class="btn btn-go" type="submit">Apply risk settings</button>
      </div>
    </form>
  </div>

  <div class="card">
    <h2>DeepSeek advisor</h2>
    <p class="muted">Click <b>DeepSeek analyze</b> in Controls (or
       <a href="/control/deepseek-analyze">run now</a>). Works with 0 trades (general advice);
       much better after ~10–20 closed trades of real outcomes.</p>
    <div class="grid">
      <div class="stat"><div class="label">Status</div>
        <div class="val {'ok' if ds_ready=='ready' else 'bad'}">{ds_ready}</div></div>
      <div class="stat"><div class="label">Risk score</div>
        <div class="val">{ds_score if ds_score is not None else '—'}</div></div>
      <div class="stat"><div class="label">Trades in last run</div>
        <div class="val">{ds_n if ds_n is not None else '—'}</div></div>
      <div class="stat"><div class="label">Auto every N closes</div>
        <div class="val">{ds.get('analyze_every') if ds.get('analyze_every') is not None else '5'}</div></div>
    </div>
    <p style="margin-top:0.85rem"><b>Summary</b></p>
    <p>{ds_summary}</p>
    <p class="muted" style="font-size:0.85rem">Last error: {ds_err or ds.get('last_error') or 'none'} ·
       model <code>{ds.get('model') or '—'}</code> · key <code>{ds.get('key_prefix') or '—'}</code></p>
    <p style="margin-top:0.75rem"><b>Trade-type verdicts</b></p>
    <ul>{ds_types_html or "<li class='muted'>Run Analyze to populate (or wait for auto after closed trades).</li>"}</ul>
    <p style="margin-top:0.75rem"><b>Hints / strategy changes</b></p>
    <ul>{ds_hints_html or "<li class='muted'>—</li>"}</ul>
  </div>

  <div class="card">
    <h2>Strategy markets</h2>
    <p class="muted">Pro-trend: 50/200 EMA · RSI · structure · break/retest · Boom/Crash no-chase · digits + CALL/PUT · conf ≥ 80%</p>
    {strat_html}
  </div>

  <div class="card">
    <h2>Market categories (scoring engines)</h2>
    <p class="muted">Each market class uses a different metric stack. Synthetics keep entropy; forex/crypto use directional engines only.</p>
    {market_book_html}
  </div>

  <div class="card">
    <p>
      <a href="/status">JSON /status</a> ·
      <a href="/health">/health</a> ·
      <a href="/oauth/login">OAuth</a>
    </p>
  </div>
</body>
</html>"""
    return HTMLResponse(html)


routes = [
    Route("/", root),
    Route("/health", health),
    Route("/ready", ready),
    Route("/status", status),
    Route("/control/resume", control_resume, methods=["GET", "POST"]),
    Route("/control/pause", control_pause, methods=["GET", "POST"]),
    Route("/control/restart", control_restart, methods=["GET", "POST"]),
    Route("/control/risk", control_risk, methods=["GET", "POST"]),
    Route(
        "/control/deepseek-analyze",
        control_deepseek_analyze,
        methods=["GET", "POST"],
    ),
    Route("/oauth/login", oauth_login),
    Route("/oauth/callback", oauth_callback),
    Route("/oauth/logout", oauth_logout),
]

app = Starlette(
    debug=os.getenv("DEBUG", "").lower() in {"1", "true"},
    routes=routes,
    lifespan=lifespan,
)
