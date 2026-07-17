import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import ta

from src.strategy.digit_contracts import extract_last_digit

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering for tick price series (train + inference aligned)."""

    # Columns that must never be fed as model inputs
    EXCLUDE_FROM_MODEL = frozenset(
        {
            "quote",
            "epoch",
            "symbol",
            "id",
            "pip_size",
            "target_digit",
            "target_parity",
        }
    )

    def create_features(
        self,
        ticks_df: pd.DataFrame,
        *,
        min_rows: int = 50,
        quiet: bool = False,
    ) -> pd.DataFrame:
        """Generate technical features from raw ticks."""
        if ticks_df is None or len(ticks_df) < min_rows:
            if not quiet:
                logger.warning(
                    "Insufficient data for feature engineering (%s rows).",
                    0 if ticks_df is None else len(ticks_df),
                )
            return pd.DataFrame()

        df = ticks_df.copy()
        if "quote" not in df.columns:
            logger.error("ticks_df missing 'quote' column")
            return pd.DataFrame()

        df["quote"] = pd.to_numeric(df["quote"], errors="coerce")
        df = df.dropna(subset=["quote"])
        if len(df) < min_rows:
            return pd.DataFrame()

        # Last digit of current quote (strong feature for digit markets)
        df["last_digit"] = df["quote"].map(extract_last_digit).astype("float")
        df["last_digit_parity"] = df["last_digit"] % 2

        # Technical indicators
        df["rsi"] = ta.momentum.RSIIndicator(df["quote"], window=14).rsi()
        df["macd"] = ta.trend.MACD(df["quote"]).macd()
        bb = ta.volatility.BollingerBands(df["quote"], window=20, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_pct"] = (df["quote"] - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"]
        ).replace(0, np.nan)

        # Rolling stats
        df["volatility"] = df["quote"].rolling(20).std()
        df["returns"] = df["quote"].pct_change()
        df["sma_10"] = df["quote"].rolling(10).mean()
        df["sma_20"] = df["quote"].rolling(20).mean()
        df["ema_10"] = df["quote"].ewm(span=10, adjust=False).mean()

        # Time features
        if "epoch" in df.columns:
            ts = pd.to_datetime(df["epoch"], unit="s", errors="coerce")
            df["hour"] = ts.dt.hour
            df["day_of_week"] = ts.dt.dayofweek
            df["minute"] = ts.dt.minute

        # Lag features (last 10 quotes)
        for i in range(1, 11):
            df[f"lag_{i}"] = df["quote"].shift(i)
            df[f"digit_lag_{i}"] = df["last_digit"].shift(i)

        df = df.dropna()
        if not quiet:
            logger.debug(
                "Engineered %d feature rows × %d cols from ticks.",
                len(df),
                len(df.columns),
            )
        return df

    def model_matrix(
        self,
        features_df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Build a numeric feature matrix aligned to feature_columns when provided.
        """
        if features_df is None or features_df.empty:
            return pd.DataFrame()

        X = features_df.drop(
            columns=[c for c in self.EXCLUDE_FROM_MODEL if c in features_df.columns],
            errors="ignore",
        )
        X = X.select_dtypes(include=["number"]).astype(float)

        if feature_columns:
            # Add missing cols as 0; drop extras; preserve order
            for col in feature_columns:
                if col not in X.columns:
                    X[col] = 0.0
            X = X.reindex(columns=feature_columns, fill_value=0.0)

        return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
