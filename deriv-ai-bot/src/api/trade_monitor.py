import logging
from typing import Dict
from src.api.deriv_client import DerivClient

logger = logging.getLogger(__name__)

class TradeMonitor:
    """
    Monitors open contracts and processes results.
    """
    def __init__(self, client: DerivClient):
        self.client = client
        self.open_contracts = {}

    def subscribe_to_contract(self, contract_id: str):
        """Subscribe to open contract updates."""
        msg = {"proposal_open_contract": 1, "contract_id": contract_id}
        self.client.send_message(msg)
        logger.info(f"Monitoring contract: {contract_id}")

    def handle_contract_update(self, data: Dict):
        """Process contract status updates."""
        if 'proposal_open_contract' in data:
            contract = data['proposal_open_contract']
            contract_id = contract.get('contract_id')
            if contract.get('is_sold', False) or contract.get('status') == 'closed':
                self.process_closed_contract(contract)
                if contract_id in self.open_contracts:
                    del self.open_contracts[contract_id]

    def process_closed_contract(self, contract: Dict):
        """Handle win/loss and P&L."""
        profit = contract.get('profit', 0)
        status = "WIN" if profit > 0 else "LOSS"
        logger.info(f"Contract {contract.get('contract_id')} {status} | Profit: ${profit}")
        # TODO: Trigger Telegram notification
