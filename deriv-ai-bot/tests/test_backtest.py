from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine


def _synth_ticks(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    quotes = 500 + np.cumsum(rng.normal(0, 0.08, n))
    quotes = np.round(quotes, 2)
    return pd.DataFrame(
        {
            "epoch": np.arange(1_700_000_000, 1_700_000_000 + n),
            "quote": quotes,
        }
    )


def test_backtest_runs_and_produces_metrics():
    df = _synth_ticks(400)
    eng = BacktestEngine(
        symbol="R_100",
        initial_balance=1000.0,
        min_confidence=0.45,
        warmup=50,
        duration_ticks=5,
        use_model=False,  # pure heuristic — no TF/XGB required
        max_trades=30,
    )
    result = eng.run(df)
    assert result.trades >= 0
    assert result.final_balance > 0
    assert 0 <= result.win_rate <= 1
    assert len(result.equity_curve) >= 1
    s = result.summary()
    assert "R_100" in s


def test_backtest_export_shape(tmp_path: Path):
    df = _synth_ticks(250)
    eng = BacktestEngine(
        symbol="R_100",
        min_confidence=0.4,
        use_model=False,
        max_trades=10,
        warmup=50,
    )
    result = eng.run(df)
    if result.trade_log:
        rows = [t.__dict__ for t in result.trade_log]
        out = tmp_path / "trades.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        loaded = pd.read_csv(out)
        assert "is_win" in loaded.columns
        assert "stake" in loaded.columns
