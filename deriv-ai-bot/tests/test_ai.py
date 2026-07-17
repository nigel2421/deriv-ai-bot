from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ai.predictor import Predictor
from src.ai.feature_engineering import FeatureEngineer
from src.ai.data_pipeline import DataPipeline
from src.ai.models import HybridModel
from src.strategy.digit_contracts import extract_last_digit


def _synthetic_ticks(n: int = 120) -> list:
    rng = np.random.default_rng(0)
    quotes = 500 + np.cumsum(rng.normal(0, 0.05, n))
    return [
        {"quote": float(q), "epoch": 1_700_000_000 + i}
        for i, q in enumerate(quotes)
    ]


def test_predictor_heuristic_shape():
    pred = Predictor(auto_load=False)
    result = pred.predict(_synthetic_ticks(60))
    assert "digit" in result
    assert 0 <= result["digit"] <= 9
    assert 0 <= result["confidence"] <= 1
    assert result["source"] in {"digit_freq", "sparse", "no_ticks"}


def test_feature_engineer_and_matrix():
    eng = FeatureEngineer()
    df = pd.DataFrame(_synthetic_ticks(80))
    feats = eng.create_features(df, quiet=True)
    assert not feats.empty
    assert "last_digit" in feats.columns
    X = eng.model_matrix(feats)
    assert X.shape[0] > 0
    assert "quote" not in X.columns


def test_pipeline_next_digit_target():
    pipe = DataPipeline()
    df = pd.DataFrame(_synthetic_ticks(150))
    X, y, yp = pipe.preprocess_for_training(df)
    assert len(X) == len(y)
    assert y.min() >= 0 and y.max() <= 9
    assert set(yp.unique()).issubset({0, 1})


def test_xgb_train_save_load_predict(tmp_path: Path):
    """End-to-end: train XGB on synthetic data, reload, predict."""
    ticks = _synthetic_ticks(400)
    df = pd.DataFrame(ticks)
    pipe = DataPipeline()
    X, y, _ = pipe.preprocess_for_training(df)
    if len(X) < 50:
        pytest.skip("not enough engineered rows")

    model = HybridModel(tmp_path)
    # Small XGB for speed
    model.train_xgb(X.iloc[:200], y.iloc[:200], n_estimators=30, max_depth=3)
    model.meta = {"test": True}
    model.save_models(tmp_path)

    assert (tmp_path / "xgboost_model.pkl").is_file()
    assert (tmp_path / "feature_columns.json").is_file()

    loaded = HybridModel(tmp_path)
    assert loaded.load_models(tmp_path)
    assert loaded.is_ready

    out = loaded.predict_digit(X.iloc[:5])
    assert 0 <= out["digit"] <= 9
    assert out["confidence"] > 0
    assert "xgb" in out["source"]

    # Predictor integration
    p = Predictor(model_dir=tmp_path, auto_load=True)
    assert p.models_loaded
    live = p.predict(ticks)
    assert 0 <= live["digit"] <= 9
    assert live["source"].startswith("model:") or "heuristic" in live["source"]


def test_extract_digit_used_in_features():
    assert extract_last_digit(123.456) == 6
