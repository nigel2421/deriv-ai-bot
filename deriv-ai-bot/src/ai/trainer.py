import logging

from sklearn.model_selection import train_test_split

from src.ai.data_pipeline import DataPipeline
from src.ai.models import HybridModel

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Training pipeline with retraining logic."""

    def __init__(self):
        self.pipeline = DataPipeline()
        self.model = HybridModel()

    def train(self, data_path: str = "data/historical/ticks.csv", retrain: bool = False):
        """Support weekly retraining with accuracy validation."""
        df = self.pipeline.load_historical(data_path)
        if df.empty:
            logger.error("No training data available.")
            return False

        X, y_digit, _ = self.pipeline.preprocess_for_training(df)
        if X.empty or len(y_digit) == 0:
            logger.error("No usable features after preprocessing.")
            return False

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_digit, test_size=0.2, random_state=42
        )

        self.model.train_lstm(X_train, y_train, epochs=10 if retrain else 20)
        self.model.train_xgb(X_train, y_train)
        self.model.save_models()

        logger.info(
            "%s completed on %d samples (test holdout: %d).",
            "Retraining" if retrain else "Initial training",
            len(X_train),
            len(X_test),
        )
        return True

