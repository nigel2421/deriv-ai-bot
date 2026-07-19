"""Market scanner gates + priority book."""
from pathlib import Path

from src.analytics.market_scanner import (
    MarketPriorityBook,
    scanner_gates_pass,
)


def test_scanner_gates_strict():
    g = scanner_gates_pass(
        edge_score=85,
        pattern_clarity=80,
        hpp=78,
        momentum_persistence=72,
        ev=0.05,
        cold_start=False,
    )
    assert g["allow"] is True

    g2 = scanner_gates_pass(
        edge_score=70,
        pattern_clarity=80,
        hpp=78,
        momentum_persistence=72,
        ev=0.05,
        cold_start=False,
    )
    assert g2["allow"] is False
    assert g2["checks"]["edge"] is False


def test_scanner_gates_cold_soft():
    g = scanner_gates_pass(
        edge_score=72,
        pattern_clarity=66,
        hpp=62,
        momentum_persistence=58,
        ev=0.01,
        cold_start=True,
    )
    assert g["allow"] is True


def test_priority_book_boost_and_reduce(tmp_path: Path):
    book = MarketPriorityBook(path=tmp_path / "pri.json")
    # Winner market
    for _ in range(10):
        book.record_trade("R_75", is_win=True, profit=1.0, hpp_velocity=3.0, clarity=80)
    # Loser market
    for _ in range(10):
        book.record_trade("JD100", is_win=False, profit=-1.0, hpp_velocity=-4.0, clarity=40)
    assert book.priority("R_75") > book.priority("JD100")
    ordered = book.ordered_symbols(["JD100", "R_75", "R_25"])
    assert ordered[0] == "R_75"


def test_report_every_500(tmp_path: Path):
    book = MarketPriorityBook(path=tmp_path / "pri2.json")
    report = None
    for i in range(500):
        sym = "R_50" if i % 2 == 0 else "R_10"
        report = book.record_trade(
            sym, is_win=(i % 3 != 0), profit=1.0 if i % 3 != 0 else -1.0
        )
    assert report is not None
    assert "Top Markets" in report.get("display", "")
    assert book.total_trades == 500
