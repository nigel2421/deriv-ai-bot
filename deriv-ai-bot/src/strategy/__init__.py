"""Strategy package: risk, signals, martingale, zuno, XML config."""

from src.strategy.martingale import MartingaleStrategy
from src.strategy.zuno_strategy import ZunoStrategy
from src.strategy.xml_parser import XMLStrategyParser
from src.strategy.strategy_engine import StrategyEngine, MarketRuntime
from src.strategy.risk_manager import RiskManager, RiskDecision
from src.strategy.signal_generator import SignalGenerator
from src.strategy.digit_contracts import (
    extract_last_digit,
    validate_digit_contract,
    would_win,
)

__all__ = [
    "MartingaleStrategy",
    "ZunoStrategy",
    "XMLStrategyParser",
    "StrategyEngine",
    "MarketRuntime",
    "RiskManager",
    "RiskDecision",
    "SignalGenerator",
    "extract_last_digit",
    "validate_digit_contract",
    "would_win",
]
