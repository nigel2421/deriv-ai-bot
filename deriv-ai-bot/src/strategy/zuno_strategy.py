import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ZunoStrategy:
    """
    Zuno contract-type switcher.

    After each settled trade, switch contract type based on win/loss.
    Stake is not managed here (use base stake or martingale separately).
    """

    def __init__(
        self,
        switch_on_win: str = "DIGITOVER",
        switch_on_loss: str = "DIGITUNDER",
        initial_type: Optional[str] = None,
    ):
        self.switch_on_win = str(switch_on_win).upper()
        self.switch_on_loss = str(switch_on_loss).upper()
        self.current_type = str(initial_type or self.switch_on_win).upper()

    def peek_type(self) -> str:
        """Contract type for the next trade (does not advance state)."""
        return self.current_type

    def on_result(self, is_win: bool) -> str:
        """Update type after settlement. Returns the next type."""
        prev = self.current_type
        self.current_type = self.switch_on_win if is_win else self.switch_on_loss
        logger.info(
            "Zuno %s → type %s → %s",
            "WIN" if is_win else "LOSS",
            prev,
            self.current_type,
        )
        return self.current_type

    def set_type(self, contract_type: str) -> None:
        self.current_type = str(contract_type).upper()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "current_type": self.current_type,
            "switch_on_win": self.switch_on_win,
            "switch_on_loss": self.switch_on_loss,
        }

    # Back-compat
    def get_next_type(self, is_win: bool) -> str:
        return self.on_result(is_win)
