import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _elem_text(parent: ET.Element, tag: str) -> Optional[str]:
    """Return stripped text for a child element, or None if missing/empty."""
    child = parent.find(tag)
    if child is None or child.text is None:
        return None
    text = child.text.strip()
    return text if text else None


def _require_float(
    parent: ET.Element, tag: str, default: float, *, required: bool = True
) -> float:
    text = _elem_text(parent, tag)
    if text is None:
        if required:
            logger.warning("Missing <%s>; using default %s", tag, default)
        return default
    return float(text)


def _require_int(
    parent: ET.Element, tag: str, default: int, *, required: bool = True
) -> int:
    text = _elem_text(parent, tag)
    if text is None:
        if required:
            logger.warning("Missing <%s>; using default %s", tag, default)
        return default
    return int(text)


def _require_bool(
    parent: ET.Element, tag: str, default: bool, *, required: bool = False
) -> bool:
    text = _elem_text(parent, tag)
    if text is None:
        if required:
            logger.warning("Missing <%s>; using default %s", tag, default)
        return default
    return text.strip().lower() in {"1", "true", "yes", "on"}


def _parse_contract_types(strategy_elem: ET.Element) -> List[str]:
    container = strategy_elem.find("contract_types")
    if container is None:
        return []
    types: List[str] = []
    for child in container.findall("type"):
        if child.text and child.text.strip():
            types.append(child.text.strip().upper())
    return types


def _parse_strategy_elem(strategy_elem: ET.Element) -> Dict[str, Any]:
    stype = (strategy_elem.get("type") or "flat").strip().lower()
    cfg: Dict[str, Any] = {
        "type": stype,
        "base_stake": _require_float(strategy_elem, "base_stake", 1.0),
        "max_steps": _require_int(strategy_elem, "max_steps", 6, required=False),
        "duration": _require_int(strategy_elem, "duration", 5, required=False),
        "duration_unit": (_elem_text(strategy_elem, "duration_unit") or "t").lower(),
        "contract_types": _parse_contract_types(strategy_elem),
        "preferred_condition": _elem_text(strategy_elem, "preferred_condition"),
        "enable_correlated_pairs": _require_bool(
            strategy_elem, "enable_correlated_pairs", False
        ),
        "switch_on_loss": (
            (_elem_text(strategy_elem, "switch_on_loss") or "DIGITUNDER").upper()
        ),
        "switch_on_win": (
            (_elem_text(strategy_elem, "switch_on_win") or "DIGITOVER").upper()
        ),
        "initial_type": (
            (_elem_text(strategy_elem, "initial_type") or "").upper() or None
        ),
        "default_barrier": _require_int(
            strategy_elem, "default_barrier", 4, required=False
        ),
        # Fallback barriers when barrier_mode=fixed (adaptive is default)
        "over_barrier": _require_int(
            strategy_elem, "over_barrier", 6, required=False
        ),
        "under_barrier": _require_int(
            strategy_elem, "under_barrier", 4, required=False
        ),
        # adaptive | fixed | random
        "barrier_mode": (
            (_elem_text(strategy_elem, "barrier_mode") or "adaptive").strip().lower()
        ),
    }
    # Optional multiplier for martingale (default classic 2x)
    cfg["multiplier"] = _require_float(
        strategy_elem, "multiplier", 2.0, required=False
    )
    return cfg


class XMLStrategyParser:
    """Parses strategy.xml for global risk + per-market strategy configuration."""

    def __init__(self, config_path: str = "config/strategy.xml"):
        self.config_path = str(config_path)
        self.config = self._parse()

    def _parse(self) -> Dict[str, Any]:
        path = Path(self.config_path)
        if not path.is_file():
            logger.error("strategy.xml not found: %s", path)
            return {"global": {}, "markets": {}}

        try:
            tree = ET.parse(path)
            root = tree.getroot()
            config: Dict[str, Any] = {}

            global_elem = root.find("global")
            if global_elem is not None:
                config["global"] = {
                    "min_confidence": _require_float(
                        global_elem, "min_confidence", 0.75
                    ),
                    "max_daily_loss_pct": _require_float(
                        global_elem, "max_daily_loss_pct", 5.0
                    ),
                    "max_consecutive_losses": _require_int(
                        global_elem, "max_consecutive_losses", 6
                    ),
                    "trade_pause_minutes": _require_int(
                        global_elem, "trade_pause_minutes", 60
                    ),
                    "stake_mode": (
                        (_elem_text(global_elem, "stake_mode") or "flat")
                        .strip()
                        .lower()
                    ),
                }
            else:
                config["global"] = {}

            config["markets"] = {}
            for market in root.findall("market"):
                symbol = market.get("symbol")
                if not symbol:
                    continue
                strategy_elem = market.find("strategy")
                if strategy_elem is None:
                    continue
                market_cfg = _parse_strategy_elem(strategy_elem)
                market_cfg["symbol"] = symbol
                config["markets"][symbol] = market_cfg
                logger.info(
                    "Loaded strategy for %s: type=%s stake=%.2f steps=%s types=%s",
                    symbol,
                    market_cfg["type"],
                    market_cfg["base_stake"],
                    market_cfg["max_steps"],
                    market_cfg["contract_types"] or "(any)",
                )

            return config
        except Exception as e:
            logger.error("Failed to parse strategy.xml: %s", e)
            return {"global": {}, "markets": {}}

    def get_strategy(self, symbol: str) -> Dict[str, Any]:
        markets = self.config.get("markets", {})
        if symbol in markets:
            return dict(markets[symbol])
        # Fallback: first configured market or empty defaults
        if "R_100" in markets:
            return dict(markets["R_100"])
        if markets:
            return dict(next(iter(markets.values())))
        return {
            "type": "flat",
            "base_stake": 1.0,
            "max_steps": 6,
            "contract_types": [],
            "switch_on_loss": "DIGITUNDER",
            "switch_on_win": "DIGITOVER",
        }

    def market_symbols(self) -> List[str]:
        return list(self.config.get("markets", {}).keys())
