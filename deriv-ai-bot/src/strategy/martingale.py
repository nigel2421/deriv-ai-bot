import logging
from typing import Dict

logger = logging.getLogger(__name__)

class MartingaleStrategy:
    """Martingale position sizing with strict risk controls."""
    
    def __init__(self, base_stake: float = 2.0, max_steps: int = 6):
        self.base_stake = base_stake
        self.max_steps = max_steps
        self.current_loss_streak = 0
        self.current_stake = base_stake

    def get_next_stake(self, is_win: bool = False) -> float:
        if is_win:
            self.reset()
            return self.base_stake
        else:
            self.current_loss_streak += 1
            if self.current_loss_streak > self.max_steps:
                logger.warning("Max loss streak reached. Stopping Martingale.")
                self.reset()
                return 0.0
            self.current_stake = self.base_stake * (2 ** self.current_loss_streak)
            return min(self.current_stake, self.base_stake * (2 ** self.max_steps))  # Cap

    def reset(self):
        self.current_loss_streak = 0
        self.current_stake = self.base_stake

    def is_safe_to_trade(self, account_balance: float, daily_loss_pct: float = 5.0) -> bool:
        """Basic risk check."""
        max_daily_loss = account_balance * (daily_loss_pct / 100)
        # TODO: Track daily loss
        return account_balance > self.base_stake * 2
