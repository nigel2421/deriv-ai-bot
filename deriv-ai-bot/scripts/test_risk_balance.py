"""
Verify live balance stream + risk gates against the real Deriv account.

  python scripts/test_risk_balance.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import (
    DERIV_APP_ID,
    MAX_OPEN_TRADES,
    MAX_STAKE_PCT,
    MIN_BALANCE,
    MIN_STAKE,
    MODE,
)
from src.utils.logger import setup_logger
from src.api.auth_manager import AuthManager
from src.api.deriv_client import DerivClient
from src.strategy.risk_manager import RiskManager

logger = setup_logger()


async def main() -> int:
    auth = AuthManager()
    app_id, token = auth.get_credentials()
    if not token or str(token).startswith("your_"):
        logger.error("Set DERIV_API_TOKEN in .env")
        return 1

    client = DerivClient(str(app_id or DERIV_APP_ID), str(token), MODE or "demo")
    ok = await client.connect()
    if not ok:
        logger.error("Authorize failed")
        await client.close()
        return 1

    client.subscribe_balance()
    bal = await client.refresh_balance()
    logger.info("Balance refresh: %s %s", bal, client.get_currency())

    # Wait a moment for stream updates
    await asyncio.sleep(1.5)
    bal2 = client.get_balance()
    logger.info("Balance after stream wait: %s", bal2)

    rm = RiskManager(
        min_balance=MIN_BALANCE,
        max_open_trades=MAX_OPEN_TRADES,
        max_stake_pct=MAX_STAKE_PCT,
        min_stake=MIN_STAKE,
    )
    if bal2 is not None:
        rm.set_session_balance(bal2)

    d0 = rm.can_trade(bal2, open_trades=0)
    d_open = rm.can_trade(bal2, open_trades=MAX_OPEN_TRADES)
    huge_stake = (bal2 or 0) * 0.5 + 1
    d_stake = rm.can_trade(bal2, open_trades=0, proposed_stake=huge_stake)
    clamped = rm.clamp_stake(huge_stake, bal2 or 0)

    logger.info("can_trade normal: %s (%s)", d0.allowed, d0.reason)
    logger.info("can_trade max open: %s (%s)", d_open.allowed, d_open.reason)
    logger.info("can_trade huge stake: %s (%s)", d_stake.allowed, d_stake.reason)
    logger.info("clamp huge stake → %s", clamped)

    # With 0 balance, trading must be blocked
    if bal2 is not None and bal2 < MIN_BALANCE:
        assert not d0.allowed, "Expected block on low balance"
        logger.info("Low-balance block OK")
    elif bal2 is not None and bal2 >= MIN_BALANCE:
        assert d0.allowed, "Expected allow with sufficient balance"
        logger.info("Sufficient-balance allow OK")

    assert not d_open.allowed
    assert not d_stake.allowed or huge_stake <= (bal2 or 0) * (MAX_STAKE_PCT / 100)

    await client.close()
    logger.info("Risk/balance test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
