import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class MartingaleStrategy:
    """
    Martingale position sizing with a hard step cap.

    - First trade uses base_stake.
    - On loss: double stake (capped at base * 2**max_steps).
    - On win: reset to base_stake.
    - If losses exceed max_steps: deactivate (peek_stake -> 0) until reset().
    """

    def __init__(self, base_stake: float = 2.0, max_steps: int = 6):
        self.base_stake = float(base_stake)
        self.max_steps = int(max_steps)
        self.current_loss_streak = 0
        self.current_stake = self.base_stake
        self.active = True

    def peek_stake(self) -> float:
        """Stake to use for the next trade (does not advance state)."""
        if not self.active:
            return 0.0
        return float(self.current_stake)

    def on_result(self, is_win: bool) -> float:
        """
        Update state after a settled trade. Returns the next stake.
        """
        if is_win:
            self.reset()
            logger.info(
                "Martingale WIN → reset stake=%.2f streak=0", self.current_stake
            )
            return self.current_stake

        # Loss
        self.current_loss_streak += 1
        if self.current_loss_streak > self.max_steps:
            self.active = False
            self.current_stake = 0.0
            logger.warning(
                "Martingale max steps (%s) exceeded — deactivated until reset.",
                self.max_steps,
            )
            return 0.0

        uncapped = self.base_stake * (2**self.current_loss_streak)
        cap = self.base_stake * (2**self.max_steps)
        self.current_stake = min(uncapped, cap)
        logger.info(
            "Martingale LOSS → streak=%s next_stake=%.2f",
            self.current_loss_streak,
            self.current_stake,
        )
        return self.current_stake

    def reset(self) -> None:
        self.current_loss_streak = 0
        self.current_stake = self.base_stake
        self.active = True

    def is_safe_to_trade(self, account_balance: float) -> bool:
        if not self.active:
            return False
        return account_balance >= self.peek_stake() and self.peek_stake() > 0

    def snapshot(self) -> Dict[str, Any]:
        return {
            "base_stake": self.base_stake,
            "max_steps": self.max_steps,
            "current_stake": self.current_stake,
            "loss_streak": self.current_loss_streak,
            "active": self.active,
        }

    # Back-compat for older call sites / tests
    def get_next_stake(self, is_win: bool = False) -> float:
        """Deprecated: advances state. Prefer peek_stake + on_result."""
        if is_win:
            return self.on_result(True)
        # Historical tests call get_next_stake(False) to simulate a loss advance
        return self.on_result(False)
