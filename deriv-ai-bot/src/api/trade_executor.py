import json
import logging
from typing import Dict, Any, Optional
from src.api.deriv_client import DerivClient

logger = logging.getLogger(__name__)

class TradeExecutor:
    """
    Handles proposal and buy operations for Digits contracts.
    """
    def __init__(self, client: DerivClient):
        self.client = client
        self.open_trades = {}

    def send_proposal(self, symbol: str, contract_type: str, stake: float, barrier: Optional[int] = None) -> None:
        """Send proposal for a digits contract."""
        proposal = {
            "proposal": 1,
            "amount": stake,
            "basis": "stake",
            "contract_type": contract_type,
            "currency": "USD",
            "duration": 5,  # 5 ticks for digits
            "duration_unit": "t",
            "symbol": symbol,
        }
        if barrier is not None:
            proposal["barrier"] = str(barrier)  # For Over/Under
        
        logger.info(f"Sending proposal for {contract_type} on {symbol} with stake ${stake}")
        self.client.send_message(proposal)

    def buy_contract(self, proposal_id: str, price: float) -> None:
        """Execute buy after proposal."""
        buy_msg = {
            "buy": proposal_id,
            "price": price
        }
        logger.info(f"Buying contract with proposal ID: {proposal_id}")
        self.client.send_message(buy_msg)

    def handle_buy_response(self, data: Dict):
        """Process buy confirmation."""
        if 'buy' in data:
            contract_id = data['buy'].get('contract_id')
            if contract_id:
                self.open_trades[contract_id] = data['buy']
                logger.info(f"Trade opened successfully. Contract ID: {contract_id}")
