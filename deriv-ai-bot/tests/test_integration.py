"""Integration-style tests (no live network)."""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pandas as pd

from src.api.deriv_client import DerivClient
from src.orchestrator import TradingOrchestrator
from src.strategy.xml_parser import XMLStrategyParser
from src.strategy.strategy_engine import StrategyEngine
from src.backtest.engine import BacktestEngine


def test_xml_and_strategy_engine_load():
    parser = XMLStrategyParser("config/strategy.xml")
    assert "global" in parser.config
    assert "R_100" in parser.config.get("markets", {})
    engine = StrategyEngine(parser)
    assert "R_100" in engine.runtimes
    intent = engine.apply_signal("R_100", "DIGITOVER", 4, 0.9)
    assert intent is not None
    assert intent["stake"] > 0


def test_orchestrator_wires_and_respects_pause():
    client = DerivClient("test_app", "test_token", mode="demo")
    # Seed fake ticks so scan can run without network
    ticks = [
        {"symbol": "R_100", "quote": 100.0 + i * 0.01, "epoch": 1_700_000_000 + i}
        for i in range(80)
    ]
    client.seed_tick_buffer("R_100", ticks)

    orch = TradingOrchestrator(client, mode="demo")
    orch.telegram.pause_trading("test")
    result = asyncio.run(orch.execute_trade_cycle())
    assert result is None
    orch.telegram.resume_trading("test")
    assert orch.telegram.trading_enabled is True
    status = orch.risk_status()
    assert "balance" in status
    assert "telegram_trading" in status


def test_client_history_parse_and_seed():
    client = DerivClient("a", "b")
    raw = {
        "history": {
            "prices": [1.1, 1.2, 1.3],
            "times": [10, 11, 12],
        }
    }
    ticks = DerivClient.parse_history_response(raw, "R_100")
    assert len(ticks) == 3
    n = client.seed_tick_buffer("R_100", ticks)
    assert n == 3
    assert client.buffer_size("R_100") == 3


def test_backtest_pipeline_smoke():
    rng = np.random.default_rng(1)
    quotes = np.round(500 + np.cumsum(rng.normal(0, 0.05, 200)), 2)
    df = pd.DataFrame({"epoch": range(200), "quote": quotes})
    eng = BacktestEngine(
        symbol="R_100",
        use_model=False,
        min_confidence=0.4,
        warmup=50,
        max_trades=5,
    )
    res = eng.run(df)
    assert res.symbol == "R_100"
    assert res.trades <= 5


def test_project_layout_no_nested_duplicate():
    """Nested scaffold should not be required for the app to run."""
    root = Path(__file__).resolve().parents[1]
    assert (root / "src" / "main.py").is_file()
    assert (root / "config" / "strategy.xml").is_file()
    assert (root / "docker-compose.yml").is_file()
    assert (root / "src" / "dashboard" / "app.py").is_file()
