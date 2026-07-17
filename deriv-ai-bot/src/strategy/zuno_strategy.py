import logging

logger = logging.getLogger(__name__)

class ZunoStrategy:
    """Zuno: Switch on loss."""
    
    def __init__(self, base_type: str = "DIGITOVER"):
        self.current_type = base_type
        self.switch_on_loss = "DIGITUNDER"
        self.switch_on_win = base_type

    def get_next_type(self, is_win: bool) -> str:
        if is_win:
            self.current_type = self.switch_on_win
        else:
            self.current_type = self.switch_on_loss
        return self.current_type
