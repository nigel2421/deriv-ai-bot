import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DataPipeline:
    """Handles data collection and preprocessing for AI."""

    def __init__(self):
        self.engineer = None  # Will init FeatureEngineer

    def load_historical(self, csv_path: str = "data/historical/ticks.csv") -> pd.DataFrame:
        try:
            path = Path(csv_path)
            if not path.is_file():
                logger.error("Training data not found: %s", path)
                return pd.DataFrame()
            df = pd.read_csv(path)
            logger.info("Loaded %d historical ticks from %s", len(df), path)
            return df
        except Exception as e:
            logger.error("Failed to load data: %s", e)
            return pd.DataFrame()

    def preprocess_for_training(self, df: pd.DataFrame) -> tuple:
        """Prepare features + target (next digit)."""
        from src.ai.feature_engineering import FeatureEngineer

        self.engineer = FeatureEngineer()
        features_df = self.engineer.create_features(df)
        if features_df.empty:
            logger.error("Feature engineering produced no rows (need more ticks).")
            return pd.DataFrame(), pd.Series(dtype=int), pd.Series(dtype=int)

        # Target: last digit of price
        features_df = features_df.copy()
        features_df["target_digit"] = (features_df["quote"] * 10).astype(int) % 10
        features_df["target_parity"] = features_df["target_digit"] % 2

        drop_cols = ["target_digit", "target_parity", "quote", "epoch"]
        X = features_df.drop(columns=[c for c in drop_cols if c in features_df.columns])
        # Keep only numeric feature columns for model training
        X = X.select_dtypes(include=["number"]).astype(float)
        y_digit = features_df["target_digit"].astype(int)
        y_parity = features_df["target_parity"].astype(int)

        return X, y_digit, y_parity

