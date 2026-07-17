import logging
from datetime import datetime
from typing import Dict

logger = logging.getLogger(__name__)

class RiskManager:
    """Centralized risk management."""
    
    def __init__(self, max_daily_loss_pct: float = 5.0, max_consecutive_losses: int = 6):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.daily_loss = 0.0
        self.consecutive_losses = 0
        self.last_reset_date = datetime.now().date()
        self.paused_until = None

    def can_trade(self, account_balance: float) -> bool:
        """Pre-trade risk checks."""
        if self.paused_until and datetime.now() < self.paused_until:
            return False
        
        daily_max = account_balance * (self.max_daily_loss_pct / 100)
        if self.daily_loss > daily_max:
            logger.warning("Daily loss limit reached.")
            return False
        
        if self.consecutive_losses >= self.max_consecutive_losses:
            logger.warning("Max consecutive losses reached. Pausing.")
            # Pause for 60 minutes
            from datetime import timedelta
            self.paused_until = datetime.now() + timedelta(minutes=60)
            return False
        
        return account_balance > 10.0  # Minimum balance safety

    def record_trade_result(self, profit: float):
        if profit < 0:
            self.consecutive_losses += 1
            self.daily_loss += abs(profit)
        else:
            self.consecutive_losses = 0
            self.daily_loss = max(0, self.daily_loss + profit)  # Reset on good days if net positive
