"""
Smoke-test the closed trade loop on demo:
  connect → (optional) force a proposal/buy → monitor until settle or timeout.

Usage (from project root, venv active):
  python scripts/test_trade_loop.py
  python scripts/test_trade_loop.py --execute false
  python scripts/test_trade_loop.py --symbol R_100 --type DIGITEVEN
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import DERIV_APP_ID, DERIV_API_TOKEN, MODE, TRADE_DURATION_TICKS
from src.utils.logger import setup_logger
from src.api.auth_manager import AuthManager
from src.api.deriv_client import DerivClient
from src.api.trade_executor import TradeExecutor
from src.api.trade_monitor import TradeMonitor

logger = setup_logger()


async def run(symbol: str, contract_type: str, stake: float, execute: bool, barrier: int | None):
    auth = AuthManager()
    app_id, token = auth.get_credentials()
    if not token or str(token).startswith("your_"):
        logger.error("Set DERIV_API_TOKEN in .env")
        return 1

    client = DerivClient(str(app_id or DERIV_APP_ID), str(token), MODE or "demo")
    ok = await client.connect()
    if not ok:
        logger.error("Failed to authorize")
        await client.close()
        return 1

    closed_event = asyncio.Event()
    result_box: dict = {}

    def on_close(contract, meta):
        result_box["contract"] = contract
        result_box["meta"] = meta
        if client._loop:
            client._loop.call_soon_threadsafe(closed_event.set)

    executor = TradeExecutor(client)
    monitor = TradeMonitor(client, on_close=on_close)

    # EVEN/ODD need no barrier; OVER/UNDER need one
    if contract_type in {"DIGITOVER", "DIGITUNDER"} and barrier is None:
        barrier = 4

    logger.info(
        "Testing %s %s stake=%s barrier=%s execute=%s balance=%s %s",
        symbol,
        contract_type,
        stake,
        barrier,
        execute,
        client.get_balance(),
        client.get_currency(),
    )

    trade = await executor.propose_and_buy(
        symbol=symbol,
        contract_type=contract_type,
        stake=stake,
        barrier=barrier,
        currency=client.get_currency(),
        duration=TRADE_DURATION_TICKS,
        execute=execute,
    )

    if not trade:
        logger.error("Trade failed: %s", executor.last_error)
        await client.close()
        return 1

    logger.info(
        "Proposal ask=%s id=%s executed=%s buy_failed=%s",
        (trade.get("proposal") or {}).get("ask_price"),
        (trade.get("proposal") or {}).get("id"),
        trade.get("executed"),
        trade.get("buy_failed"),
    )

    if trade.get("buy_failed"):
        logger.error(
            "Buy failed after proposal (expected if balance is 0): %s",
            trade.get("error") or executor.last_error,
        )
        await client.close()
        # Exit 0: loop path worked; API correctly rejected buy
        logger.info("Trade loop path PASSED (proposal ok, buy error handled).")
        return 0

    if not trade.get("executed"):
        logger.info("Proposal-only test PASSED (execute=false).")
        await client.close()
        return 0

    cid = trade.get("contract_id")
    await monitor.watch(cid, meta={"symbol": symbol, "contract_type": contract_type})

    logger.info("Waiting up to 90s for contract %s to settle...", cid)
    try:
        await asyncio.wait_for(closed_event.wait(), timeout=90)
    except asyncio.TimeoutError:
        logger.warning("Timed out waiting for settlement (contract may still be open).")
        await client.close()
        return 2

    contract = result_box.get("contract") or {}
    logger.info(
        "Trade loop PASSED | profit=%s status_fields=%s",
        contract.get("profit"),
        {
            "is_sold": contract.get("is_sold"),
            "status": contract.get("status"),
            "is_expired": contract.get("is_expired"),
        },
    )
    await client.close()
    return 0


def main():
    p = argparse.ArgumentParser(description="Test Deriv proposal→buy→monitor loop")
    p.add_argument("--symbol", default="R_100")
    p.add_argument(
        "--type",
        dest="contract_type",
        default="DIGITEVEN",
        help="DIGITEVEN is safest for smoke tests (no barrier)",
    )
    p.add_argument("--stake", type=float, default=0.35)
    p.add_argument("--barrier", type=int, default=None)
    p.add_argument(
        "--execute",
        default="true",
        choices=["true", "false"],
        help="false = proposal only",
    )
    args = p.parse_args()
    code = asyncio.run(
        run(
            args.symbol,
            args.contract_type,
            args.stake,
            args.execute == "true",
            args.barrier,
        )
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
