"""
Shared bot lifecycle used by CLI (src/main.py) and Cloud Run HTTP server.

Keeps a single long-running trading loop that can be cancelled cleanly.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from config.settings import (
    DERIV_ACCOUNT_ID,
    DERIV_API_BASE,
    DERIV_API_MODE,
    MODE,
    SAVE_TICK_HISTORY,
    SYMBOLS,
    TICK_HISTORY_COUNT,
    TICK_HISTORY_MIN,
)
from src.api.auth_manager import AuthManager
from src.api.deriv_client import DerivClient
from src.api.price_fetcher import PriceFetcher
from src.api.token_store import load_access_token
from src.orchestrator import TradingOrchestrator

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class BotRuntime:
    """Holds live bot state for health/status endpoints."""

    mode: str = MODE
    started_at: Optional[str] = None
    status: str = "stopped"  # starting | running | error | stopped
    last_error: Optional[str] = None
    last_cycle_at: Optional[str] = None
    last_heartbeat: Dict[str, Any] = field(default_factory=dict)
    buffer_sizes: Dict[str, int] = field(default_factory=dict)
    client: Optional[DerivClient] = None
    fetcher: Optional[PriceFetcher] = None
    orchestrator: Optional[TradingOrchestrator] = None
    _task: Optional[asyncio.Task] = None
    _stop: Optional[asyncio.Event] = None

    def public_status(self) -> Dict[str, Any]:
        risk = {}
        if self.orchestrator is not None:
            try:
                risk = self.orchestrator.risk_status()
            except Exception as e:
                risk = {"error": str(e)}
        return {
            "status": self.status,
            "mode": self.mode,
            "started_at": self.started_at,
            "last_error": self.last_error,
            "last_cycle_at": self.last_cycle_at,
            "symbols": risk.get("symbols") or SYMBOLS,
            "buffer_sizes": self.buffer_sizes,
            "heartbeat": self.last_heartbeat,
            "closed_trades": risk.get("closed_trades"),
            "strategies": risk.get("strategies") or {},
            "risk": {
                k: risk.get(k)
                for k in (
                    "balance",
                    "currency",
                    "open_trades",
                    "daily_pnl",
                    "paused",
                    "telegram_trading",
                    "execute_trades",
                    "consecutive_losses",
                    "trades_today",
                    "max_open_trades",
                )
            },
        }


# Process-wide singleton (one service instance per container)
runtime = BotRuntime()


async def _trading_loop(rt: BotRuntime, cycle_seconds: int = 60) -> None:
    assert rt.orchestrator is not None
    assert rt.fetcher is not None
    assert rt.client is not None
    assert rt._stop is not None

    orch = rt.orchestrator
    fetcher = rt.fetcher
    client = rt.client

    while not rt._stop.is_set():
        try:
            await orch.execute_trade_cycle()
            bal = await orch._live_balance(refresh=True)
            status = orch.risk_status()
            sizes = fetcher.buffer_sizes(SYMBOLS)
            rt.buffer_sizes = sizes
            rt.last_cycle_at = datetime.now(timezone.utc).isoformat()
            rt.last_heartbeat = {
                "balance": bal,
                "currency": client.get_currency(),
                "open_trades": status.get("open_trades"),
                "daily_pnl": status.get("daily_pnl"),
                "risk_paused": status.get("paused"),
                "telegram_trading": status.get("telegram_trading"),
                "buffers": sizes,
            }
            logger.info(
                "Heartbeat balance=%s %s open=%s daily_pnl=%s "
                "risk_paused=%s tg_trading=%s buffers=%s",
                bal,
                client.get_currency(),
                status.get("open_trades"),
                status.get("daily_pnl"),
                status.get("paused"),
                status.get("telegram_trading"),
                sizes,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            rt.last_error = str(e)
            logger.exception("Trade cycle error: %s", e)

        try:
            await asyncio.wait_for(rt._stop.wait(), timeout=cycle_seconds)
        except asyncio.TimeoutError:
            pass


async def start_bot(mode: Optional[str] = None, *, cycle_seconds: int = 60) -> BotRuntime:
    """Connect to Deriv, bootstrap ticks, start trading loop in background."""
    rt = runtime
    if rt.status == "running" and rt._task and not rt._task.done():
        logger.info("Bot already running")
        return rt

    rt.mode = mode or MODE
    rt.status = "starting"
    rt.last_error = None
    rt.started_at = datetime.now(timezone.utc).isoformat()
    rt._stop = asyncio.Event()

    auth = AuthManager()
    app_id, env_token = auth.get_credentials()
    # Prefer OAuth-stored access token (from /oauth/callback), else env PAT/token
    stored = load_access_token()
    token = (stored or env_token or "").strip()

    if not token or str(token).startswith("your_"):
        rt.status = "error"
        rt.last_error = (
            "Missing access token. Set DERIV_API_TOKEN (PAT) or complete "
            "OAuth login at /oauth/login"
        )
        logger.error(rt.last_error)
        return rt

    if not app_id:
        rt.status = "error"
        rt.last_error = "Missing DERIV_APP_ID"
        logger.error(rt.last_error)
        return rt

    api_mode = None if DERIV_API_MODE == "auto" else DERIV_API_MODE
    logger.info(
        "Starting bot mode=%s app_id=%s… api_mode=%s",
        rt.mode,
        str(app_id)[:10],
        api_mode or "auto",
    )
    client = DerivClient(
        str(app_id),
        str(token),
        rt.mode,
        api_mode=api_mode,
        account_id=DERIV_ACCOUNT_ID,
        api_base=DERIV_API_BASE,
    )
    rt.client = client

    ok = await client.connect()
    if not ok:
        rt.status = "error"
        rt.last_error = client.last_error or "Deriv authorize failed"
        await client.close()
        return rt

    client.subscribe_balance()
    balance = await client.refresh_balance()
    logger.info(
        "Balance %s %s loginid=%s",
        balance,
        client.get_currency(),
        (client.account or {}).get("loginid"),
    )

    fetcher = PriceFetcher(client)
    rt.fetcher = fetcher
    fetcher.subscribe_symbols(SYMBOLS)

    save_dir = str(ROOT / "data" / "historical") if SAVE_TICK_HISTORY else None
    sizes = await fetcher.bootstrap_history(
        SYMBOLS,
        count=TICK_HISTORY_COUNT,
        min_required=TICK_HISTORY_MIN,
        save_dir=save_dir,
    )
    rt.buffer_sizes = sizes
    logger.info("Tick bootstrap: %s", sizes)

    orch = TradingOrchestrator(client, rt.mode)
    rt.orchestrator = orch
    if balance is not None:
        orch.risk_manager.set_session_balance(balance)

    await orch.telegram.start_bot()
    logger.info("Risk status: %s", orch.risk_status())

    rt.status = "running"
    rt._task = asyncio.create_task(
        _trading_loop(rt, cycle_seconds=cycle_seconds), name="trading-loop"
    )
    logger.info("Trading loop started (cycle=%ss)", cycle_seconds)
    return rt


async def stop_bot() -> None:
    """Cancel trading loop and close connections."""
    rt = runtime
    if rt._stop:
        rt._stop.set()
    if rt._task and not rt._task.done():
        rt._task.cancel()
        try:
            await rt._task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
    rt._task = None

    if rt.orchestrator is not None:
        try:
            await rt.orchestrator.telegram.send_notification(
                "🛑 Deriv AI Bot shutting down.", force=True
            )
            await rt.orchestrator.telegram.stop_bot()
        except Exception as e:
            logger.debug("Telegram shutdown: %s", e)

    if rt.client is not None:
        try:
            await rt.client.close()
        except Exception:
            pass

    rt.status = "stopped"
    logger.info("Bot stopped")


async def run_bot_forever(mode: Optional[str] = None, *, cycle_seconds: int = 60) -> None:
    """CLI entry: start bot and block until cancelled."""
    await start_bot(mode, cycle_seconds=cycle_seconds)
    if runtime.status != "running" or runtime._task is None:
        return
    try:
        await runtime._task
    except asyncio.CancelledError:
        pass
    finally:
        await stop_bot()
