import logging
from typing import Any, Dict, Optional

from src.api.deriv_client import DerivClient
from src.strategy.contract_types import (
    build_proposal_fields,
    normalize_contract_type,
    validate_contract,
)

logger = logging.getLogger(__name__)


class TradeExecutor:
    """
    Proposal + buy for Digits and Rise/Fall (CALL/PUT) via DerivClient.request().
    """

    def __init__(self, client: DerivClient):
        self.client = client
        self.open_trades: Dict[Any, Dict[str, Any]] = {}
        self.last_error: Optional[str] = None

    def build_proposal(
        self,
        symbol: str,
        contract_type: str,
        stake: float,
        barrier: Optional[int] = None,
        currency: str = "USD",
        duration: int = 5,
        duration_unit: str = "t",
    ) -> Dict[str, Any]:
        ok, reason, nb = validate_contract(contract_type, barrier)
        if not ok:
            raise ValueError(f"Invalid contract: {reason}")

        ct = normalize_contract_type(contract_type)
        assert ct is not None

        proposal: Dict[str, Any] = {
            "proposal": 1,
            "amount": float(stake),
            "basis": "stake",
            "contract_type": ct,
            "currency": currency,
            "duration": int(duration),
            "duration_unit": duration_unit,
        }
        # v2 Options WS rejects "symbol"; uses underlying_symbol instead.
        # Legacy classic WS still uses "symbol".
        api_mode = getattr(self.client, "api_mode", "legacy")
        if api_mode == "v2":
            proposal["underlying_symbol"] = symbol
        else:
            proposal["symbol"] = symbol
        # Digits: barrier when required; CALL/PUT: no barrier
        proposal.update(build_proposal_fields(ct, nb))
        return proposal

    async def send_proposal(
        self,
        symbol: str,
        contract_type: str,
        stake: float,
        barrier: Optional[int] = None,
        currency: Optional[str] = None,
        duration: int = 5,
        duration_unit: str = "t",
        timeout: float = 20.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Request a proposal and return the proposal object, or None on failure.
        duration_unit: t=ticks, m=minutes, s=seconds
        """
        currency = currency or self.client.get_currency()
        try:
            msg = self.build_proposal(
                symbol,
                contract_type,
                stake,
                barrier,
                currency,
                duration,
                duration_unit=duration_unit or "t",
            )
        except ValueError as e:
            self.last_error = str(e)
            logger.error("Invalid proposal params: %s", e)
            return None

        logger.info(
            "Proposal request: %s %s stake=%s barrier=%s dur=%s%s currency=%s",
            symbol,
            msg.get("contract_type"),
            stake,
            barrier,
            duration,
            duration_unit,
            currency,
        )
        try:
            data = await self.client.request(msg, timeout=timeout)
        except Exception as e:
            self.last_error = str(e)
            logger.error("Proposal request failed: %s", e)
            return None

        if data.get("error"):
            self.last_error = data["error"].get("message", str(data["error"]))
            logger.error("Proposal error: %s", data["error"])
            return None

        proposal = data.get("proposal")
        if not proposal or not proposal.get("id"):
            self.last_error = "Proposal response missing id"
            logger.error("Unexpected proposal payload: %s", data)
            return None

        logger.info(
            "Proposal ok id=%s ask=%s payout=%s",
            proposal.get("id"),
            proposal.get("ask_price"),
            proposal.get("payout"),
        )
        return proposal

    async def buy_contract(
        self,
        proposal_id: str,
        price: float,
        timeout: float = 20.0,
    ) -> Optional[Dict[str, Any]]:
        """Buy using a proposal id. Returns the buy object or None."""
        msg = {"buy": proposal_id, "price": float(price)}
        logger.info("Buy request: proposal_id=%s price=%s", proposal_id, price)
        try:
            data = await self.client.request(msg, timeout=timeout)
        except Exception as e:
            self.last_error = str(e)
            logger.error("Buy request failed: %s", e)
            return None

        if data.get("error"):
            self.last_error = data["error"].get("message", str(data["error"]))
            logger.error("Buy error: %s", data["error"])
            return None

        buy = data.get("buy")
        if not buy:
            self.last_error = "Buy response missing buy object"
            logger.error("Unexpected buy payload: %s", data)
            return None

        contract_id = buy.get("contract_id")
        if contract_id is not None:
            self.open_trades[contract_id] = buy
        logger.info(
            "Buy ok contract_id=%s buy_price=%s balance_after=%s",
            contract_id,
            buy.get("buy_price"),
            buy.get("balance_after"),
        )
        return buy

    @staticmethod
    def payout_metrics(proposal: Dict[str, Any], stake: float) -> Dict[str, float]:
        """
        Derive ask, payout, and net return from a proposal quote.
        net_return = (payout - ask) / ask  (profit per dollar risked).
        """
        try:
            ask = float(proposal.get("ask_price") or stake)
        except (TypeError, ValueError):
            ask = float(stake)
        if ask <= 0:
            ask = float(stake) if float(stake) > 0 else 1.0
        try:
            payout = float(proposal.get("payout") or 0)
        except (TypeError, ValueError):
            payout = 0.0
        net_return = (payout - ask) / ask if ask > 0 else 0.0
        return {"ask": ask, "payout": payout, "net_return": net_return}

    async def propose_and_buy(
        self,
        symbol: str,
        contract_type: str,
        stake: float,
        barrier: Optional[int] = None,
        currency: Optional[str] = None,
        duration: int = 5,
        duration_unit: str = "t",
        execute: bool = True,
        timeout: float = 20.0,
        min_net_return: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Full path: proposal → payout gate → (optional) buy.

        When execute=False, only requests a proposal (paper / dry-run).
        When min_net_return is set, skips buy if (payout-ask)/ask is below it.
        Returns a result dict with keys: proposal, buy (optional), contract_id.
        """
        self.last_error = None
        proposal = await self.send_proposal(
            symbol,
            contract_type,
            stake,
            barrier=barrier,
            currency=currency,
            duration=duration,
            duration_unit=duration_unit or "t",
            timeout=timeout,
        )
        if not proposal:
            return None

        metrics = self.payout_metrics(proposal, stake)
        result: Dict[str, Any] = {
            "proposal": proposal,
            "buy": None,
            "contract_id": None,
            "symbol": symbol,
            "contract_type": contract_type,
            "stake": stake,
            "barrier": barrier,
            "duration": duration,
            "duration_unit": duration_unit,
            "executed": False,
            "ask_price": metrics["ask"],
            "payout": metrics["payout"],
            "net_return": metrics["net_return"],
        }

        if not execute:
            logger.info("Dry-run: proposal only (execute=False)")
            return result

        # Skip junk odds (e.g. OVER@0 paying ~+$0.09 on $1)
        if min_net_return is not None and metrics["payout"] > 0:
            if metrics["net_return"] + 1e-9 < float(min_net_return):
                msg = (
                    f"Payout too low: net +{metrics['net_return']:.0%} "
                    f"(payout={metrics['payout']:.2f} on ask={metrics['ask']:.2f}) "
                    f"< required +{float(min_net_return):.0%}"
                )
                self.last_error = msg
                result["skipped_low_payout"] = True
                result["error"] = msg
                logger.warning(
                    "Skip buy %s %s barrier=%s: %s",
                    symbol,
                    contract_type,
                    barrier,
                    msg,
                )
                return result

        proposal_id = proposal["id"]
        ask = metrics["ask"]
        # Use max(ask, stake) so we cover the quoted price
        price = max(ask, float(stake))

        buy = await self.buy_contract(proposal_id, price, timeout=timeout)
        if not buy:
            # Proposal succeeded but buy failed (e.g. InsufficientBalance)
            result["buy_failed"] = True
            result["error"] = self.last_error
            return result

        result["buy"] = buy
        result["contract_id"] = buy.get("contract_id")
        result["executed"] = True

        # Refresh cached balance if present
        bal = buy.get("balance_after")
        if bal is not None:
            self.client.account["balance"] = bal

        return result

    def mark_closed(self, contract_id: Any) -> None:
        self.open_trades.pop(contract_id, None)

    def open_count(self) -> int:
        return len(self.open_trades)
