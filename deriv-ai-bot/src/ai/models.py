import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import xgboost as xgb
import joblib
import logging
import os

logger = logging.getLogger(__name__)

class HybridModel:
    """LSTM for sequences + XGBoost for boosted predictions."""
    
    def __init__(self):
        self.lstm_model = None
        self.xgb_model = None
        self.scaler = None
    
    def build_lstm(self, input_shape: tuple):
        self.lstm_model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(10, activation='softmax')  # For digit prediction (0-9)
        ])
        self.lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        logger.info("LSTM model built.")
    
    def train_lstm(self, X_train, y_train, epochs=20):
        if self.lstm_model is None:
            self.build_lstm((X_train.shape[1], 1))
        # Reshape for LSTM
        X_lstm = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
        self.lstm_model.fit(X_lstm, y_train, epochs=epochs, batch_size=32, validation_split=0.2, verbose=1)
    
    def train_xgb(self, X_train, y_train):
        self.xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5)
        self.xgb_model.fit(X_train, y_train)
        logger.info("XGBoost trained.")
    
    def save_models(self, path: str = "src/models/"):
        os.makedirs(path, exist_ok=True)
        if self.lstm_model:
            self.lstm_model.save(f"{path}lstm_model.h5")
        if self.xgb_model:
            joblib.dump(self.xgb_model, f"{path}xgboost_model.pkl")
        logger.info("Models saved.")
