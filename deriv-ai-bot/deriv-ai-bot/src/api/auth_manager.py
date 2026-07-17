from config.settings import DERIV_API_TOKEN, DERIV_APP_ID
import logging

logger = logging.getLogger(__name__)

class AuthManager:
    def __init__(self):
        self.app_id = DERIV_APP_ID
        self.token = DERIV_API_TOKEN
        if not self.token:
            logger.error("DERIV_API_TOKEN not set in environment variables.")

    def get_credentials(self):
        return self.app_id, self.token
