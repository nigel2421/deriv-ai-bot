import logging
from typing import Any, Callable, Dict, Optional

from src.api.deriv_client import DerivClient

logger = logging.getLogger(__name__)

CloseCallback = Callable[[Dict[str, Any], Dict[str, Any]], None]


class TradeMonitor:
    """
    Subscribes to proposal_open_contract streams and reports closed contracts.
    """

    def __init__(
        self,
        client: DerivClient,
        on_close: Optional[CloseCallback] = None,
    ):
        self.client = client
        self.on_close = on_close
        # contract_id -> metadata from when the trade was opened
        self.open_contracts: Dict[Any, Dict[str, Any]] = {}
        self._closed_seen: set = set()
        self.client.register_handler(
            "proposal_open_contract", self.handle_contract_update
        )

    async def watch(
        self,
        contract_id: Any,
        meta: Optional[Dict[str, Any]] = None,
        timeout: float = 20.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Subscribe to an open contract. Returns the first snapshot if available.
        """
        if contract_id is None:
            logger.error("Cannot watch contract: contract_id is None")
            return None

        self.open_contracts[contract_id] = dict(meta or {})
        msg = {
            "proposal_open_contract": 1,
            "contract_id": int(contract_id)
            if str(contract_id).isdigit()
            else contract_id,
            "subscribe": 1,
        }
        logger.info("Monitoring contract_id=%s", contract_id)
        try:
            data = await self.client.request(msg, timeout=timeout)
        except Exception as e:
            logger.error("Failed to subscribe to contract %s: %s", contract_id, e)
            return None

        if data.get("error"):
            logger.error(
                "proposal_open_contract error for %s: %s",
                contract_id,
                data["error"],
            )
            return None

        contract = data.get("proposal_open_contract")
        if contract:
            # Handle immediate settlement (very short digit contracts)
            self._maybe_close(contract)
        return contract

    def handle_contract_update(self, data: Dict[str, Any]) -> None:
        """WS stream handler (also used for first response if routed by type)."""
        if data.get("error"):
            logger.error("Open-contract stream error: %s", data["error"])
            return
        contract = data.get("proposal_open_contract")
        if not contract:
            return
        self._maybe_close(contract)

    def _is_closed(self, contract: Dict[str, Any]) -> bool:
        if contract.get("is_sold") in (1, True, "1"):
            return True
        status = str(contract.get("status") or "").lower()
        if status in {"sold", "closed", "won", "lost"}:
            return True
        # Digit contracts often settle with is_expired
        if contract.get("is_expired") in (1, True, "1") and contract.get(
            "is_settleable", 1
        ) in (1, True, "1"):
            # Prefer explicit profit presence after expiry
            if "profit" in contract:
                return True
        return False

    def _maybe_close(self, contract: Dict[str, Any]) -> None:
        if not self._is_closed(contract):
            return

        contract_id = contract.get("contract_id")
        if contract_id is None:
            return

        # Deduplicate stream updates after close
        if contract_id in self._closed_seen:
            return
        self._closed_seen.add(contract_id)

        meta = self.open_contracts.pop(contract_id, {})
        self.process_closed_contract(contract, meta)

    def process_closed_contract(
        self, contract: Dict[str, Any], meta: Optional[Dict[str, Any]] = None
    ) -> None:
        meta = meta or {}
        profit = contract.get("profit", 0)
        try:
            profit_f = float(profit)
        except (TypeError, ValueError):
            profit_f = 0.0

        status = "WIN" if profit_f > 0 else ("PUSH" if profit_f == 0 else "LOSS")
        logger.info(
            "Contract %s %s | profit=%s | %s %s",
            contract.get("contract_id"),
            status,
            profit_f,
            meta.get("symbol"),
            meta.get("contract_type"),
        )

        if self.on_close:
            try:
                self.on_close(contract, meta)
            except Exception as e:
                logger.exception("on_close callback failed: %s", e)

    def open_count(self) -> int:
        return len(self.open_contracts)
