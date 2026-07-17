import pandas as pd
import numpy as np
from typing import List, Dict
import logging
import ta  # Technical analysis library (add to requirements if needed)

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Feature engineering for time series price data."""
    
    def create_features(self, ticks_df: pd.DataFrame) -> pd.DataFrame:
        """Generate technical features from raw ticks."""
        if len(ticks_df) < 50:
            logger.warning("Insufficient data for feature engineering.")
            return pd.DataFrame()
        
        df = ticks_df.copy()
        df['quote'] = pd.to_numeric(df['quote'], errors='coerce')
        
        # Technical indicators
        df["rsi"] = ta.momentum.RSIIndicator(df["quote"], window=14).rsi()
        df["macd"] = ta.trend.MACD(df["quote"]).macd()
        bb = ta.volatility.BollingerBands(df["quote"], window=20, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()

        # Rolling stats
        df["volatility"] = df["quote"].rolling(20).std()
        df["returns"] = df["quote"].pct_change()

        # Time features
        if "epoch" in df.columns:
            ts = pd.to_datetime(df["epoch"], unit="s", errors="coerce")
            df["hour"] = ts.dt.hour
            df["day_of_week"] = ts.dt.dayofweek

        # Lag features (last 10 ticks)
        for i in range(1, 11):
            df[f"lag_{i}"] = df["quote"].shift(i)

        df = df.dropna()
        logger.info("Engineered %d features from %d ticks.", len(df.columns), len(df))
        return df

