"""
Unit tests for DeepSeekAdvisor — 1000-trade history analysis.

Tests:
  1. Default MAX_HISTORY_TRADES is 1000.
  2. _load_symbol_history respects the 1000-trade cap.
  3. DEEPSEEK_MAX_TRADES env var overrides the history limit.
  4. Payload builds correctly with a 1000-trade history.
  5. snapshot() exposes max_history_trades.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from src.strategy.deepseek_advisor import (
    DEFAULT_MAX_HISTORY_TRADES,
    DEFAULT_MAX_GLOBAL_TRADES,
    DeepSeekAdvisor,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_advisor(tmp_path: Path, **env_overrides) -> DeepSeekAdvisor:
    """Create a DeepSeekAdvisor pointing at tmp_path, with optional env overrides."""
    old = {}
    for k, v in env_overrides.items():
        old[k] = os.environ.get(k)
        os.environ[k] = str(v)
    advisor = DeepSeekAdvisor(
        history_path=tmp_path / "trade_history.jsonl",
        report_path=tmp_path / "deepseek_report.json",
        state_path=tmp_path / "deepseek_state.json",
    )
    # Restore env
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return advisor


def _write_trades(path: Path, symbol: str, n: int, extra_symbols: int = 0) -> None:
    """Write `n` fake trades for `symbol`, plus optional noise rows for other symbols."""
    rows = []
    for i in range(n):
        rows.append(json.dumps({
            "symbol": symbol,
            "contract_type": "DIGITOVER",
            "is_win": i % 2 == 0,
            "profit": 1.0 if i % 2 == 0 else -1.0,
            "confidence": 0.82 + (i % 10) * 0.01,
            "ev": 0.05,
            "ts": f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}Z",
        }))
    for j in range(extra_symbols):
        rows.append(json.dumps({
            "symbol": f"OTHER_{j}",
            "contract_type": "CALL",
            "is_win": True,
            "profit": 1.0,
        }))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestDeepSeekConstants:
    def test_default_history_limit_is_1000(self):
        assert DEFAULT_MAX_HISTORY_TRADES == 1000

    def test_default_global_scan_is_10000(self):
        assert DEFAULT_MAX_GLOBAL_TRADES == 10000


class TestLoadSymbolHistory:
    def test_loads_up_to_1000_trades(self, tmp_path: Path):
        history = tmp_path / "trade_history.jsonl"
        _write_trades(history, "R_100", n=1200)

        advisor = _make_advisor(tmp_path)
        trades = advisor._load_symbol_history("R_100")

        # Should cap at max_history_trades (default 1000)
        assert len(trades) == 1000

    def test_loads_fewer_than_limit_when_not_enough(self, tmp_path: Path):
        history = tmp_path / "trade_history.jsonl"
        _write_trades(history, "R_100", n=250)

        advisor = _make_advisor(tmp_path)
        trades = advisor._load_symbol_history("R_100")

        assert len(trades) == 250

    def test_filters_by_symbol(self, tmp_path: Path):
        history = tmp_path / "trade_history.jsonl"
        _write_trades(history, "R_50", n=300, extra_symbols=500)

        advisor = _make_advisor(tmp_path)
        trades = advisor._load_symbol_history("R_50")

        assert all(t["symbol"] == "R_50" for t in trades)
        assert len(trades) == 300

    def test_env_override_deepseek_max_trades(self, tmp_path: Path):
        history = tmp_path / "trade_history.jsonl"
        _write_trades(history, "R_75", n=800)

        # Override to only use 500
        old = os.environ.get("DEEPSEEK_MAX_TRADES")
        os.environ["DEEPSEEK_MAX_TRADES"] = "500"
        try:
            advisor = DeepSeekAdvisor(
                history_path=history,
                report_path=tmp_path / "r.json",
                state_path=tmp_path / "s.json",
            )
        finally:
            if old is None:
                os.environ.pop("DEEPSEEK_MAX_TRADES", None)
            else:
                os.environ["DEEPSEEK_MAX_TRADES"] = old

        assert advisor.max_history_trades == 500
        trades = advisor._load_symbol_history("R_75")
        assert len(trades) == 500

    def test_returns_most_recent_trades_when_capped(self, tmp_path: Path):
        """Ensure we get the LAST 1000 trades (most recent), not the first."""
        history = tmp_path / "trade_history.jsonl"
        rows = []
        for i in range(1100):
            rows.append(json.dumps({
                "symbol": "R_100",
                "profit": float(i),   # unique marker
                "is_win": True,
                "contract_type": "DIGITOVER",
            }))
        history.write_text("\n".join(rows) + "\n", encoding="utf-8")

        advisor = _make_advisor(tmp_path)
        trades = advisor._load_symbol_history("R_100")

        # The first trade returned should have profit=100 (the oldest of the last 1000)
        assert float(trades[0]["profit"]) == pytest.approx(100.0)
        assert float(trades[-1]["profit"]) == pytest.approx(1099.0)


class TestPayloadWith1000Trades:
    def test_payload_reflects_1000_trades(self, tmp_path: Path):
        history = tmp_path / "trade_history.jsonl"
        _write_trades(history, "R_100", n=1000)

        advisor = _make_advisor(tmp_path)
        trades = advisor._load_symbol_history("R_100")
        payload = advisor._build_analysis_payload("R_100", trades)

        assert payload["total_trades_analyzed"] == 1000
        assert payload["symbol"] == "R_100"
        assert "contract_type_breakdown" in payload
        assert "confidence_bucket_analysis" in payload
        assert "ev_stats" in payload


class TestSnapshot:
    def test_snapshot_exposes_max_history_trades(self, tmp_path: Path):
        advisor = _make_advisor(tmp_path)
        snap = advisor.snapshot()
        assert "max_history_trades" in snap
        assert snap["max_history_trades"] == 1000
