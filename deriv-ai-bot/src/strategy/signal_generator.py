import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class SignalGenerator:
    """Converts AI predictions to trade signals."""
    
    def generate_signal(self, prediction: Dict, confidence: float, min_confidence: float = 0.75) -> Tuple[str, str, float]:
        if confidence < min_confidence:
            return None, None, 0.0
        
        # Example: Map prediction to contract type
        predicted_digit = prediction.get('digit', 5)
        if predicted_digit > 4:
            contract_type = "DIGITOVER"
        else:
            contract_type = "DIGITUNDER"
        
        barrier = predicted_digit
        logger.info(f"Generated signal: {contract_type} | Confidence: {confidence:.2%}")
        return contract_type, barrier, confidence
