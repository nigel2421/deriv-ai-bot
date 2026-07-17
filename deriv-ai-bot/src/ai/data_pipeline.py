import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from src.strategy.digit_contracts import extract_last_digit

logger = logging.getLogger(__name__)


class DataPipeline:
    """Handles data loading and preprocessing for AI train/infer."""

    def __init__(self):
        self.engineer = None
        self.feature_columns: Optional[List[str]] = None

    def load_historical(
        self, csv_path: str = "data/historical/ticks.csv"
    ) -> pd.DataFrame:
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

    def preprocess_for_training(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare features + target.

        Target is the **next** tick's last digit (shifted -1), not the current
        row's digit — avoids label leakage.
        """
        from src.ai.feature_engineering import FeatureEngineer

        self.engineer = FeatureEngineer()
        features_df = self.engineer.create_features(df)
        if features_df.empty:
            logger.error("Feature engineering produced no rows (need more ticks).")
            return pd.DataFrame(), pd.Series(dtype=int), pd.Series(dtype=int)

        features_df = features_df.copy()
        # Current last digit already in features; target = next tick digit
        features_df["next_digit"] = (
            features_df["quote"].map(extract_last_digit).astype("float").shift(-1)
        )
        features_df = features_df.dropna(subset=["next_digit"])
        features_df["target_digit"] = features_df["next_digit"].astype(int) % 10
        features_df["target_parity"] = features_df["target_digit"] % 2
        features_df = features_df.drop(columns=["next_digit"])

        X = self.engineer.model_matrix(features_df)
        y_digit = features_df.loc[X.index, "target_digit"].astype(int)
        y_parity = features_df.loc[X.index, "target_parity"].astype(int)

        self.feature_columns = list(X.columns)
        logger.info(
            "Training matrix: %d rows × %d features; digit dist sample=%s",
            len(X),
            X.shape[1],
            y_digit.value_counts().sort_index().to_dict(),
        )
        return X, y_digit, y_parity
