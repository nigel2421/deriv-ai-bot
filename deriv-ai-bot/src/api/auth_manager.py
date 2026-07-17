import logging
from typing import Optional, Tuple

from config.settings import DERIV_API_TOKEN, DERIV_APP_ID

logger = logging.getLogger(__name__)


class AuthManager:
    def __init__(self):
        self.app_id: Optional[str] = DERIV_APP_ID
        self.token: Optional[str] = DERIV_API_TOKEN
        # Token may be filled later via OAuth store; only warn if clearly placeholder
        if self.token and str(self.token).startswith("your_"):
            logger.error("DERIV_API_TOKEN looks like a placeholder.")
        if not self.app_id:
            logger.error("DERIV_APP_ID not set.")

    def get_credentials(self) -> Tuple[Optional[str], Optional[str]]:
        return self.app_id, self.token

