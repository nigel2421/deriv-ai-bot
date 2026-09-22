"""
Market Discovery Manager — Dynamic Deriv API Asset Discovery & Lifecycle System.

Queries Deriv WebSocket API (`active_symbols`) to dynamically discover available synthetic
and forex markets, classify asset families, inspect contract availability, and manage market
promotion lifecycles:

  DISCOVERED ──> VALIDATED ──> PAPER_TRADING ──> ELIGIBLE ──> ACTIVE
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Default known synthetic & forex portfolio baseline if WS API query is unavailable
DEFAULT_SYMBOLS_PORTFOLIO = [
    # Volatility Indices
    "R_10", "R_25", "R_50", "R_75", "R_100",
    "1HZ10V", "1HZ25V", "1HZ50V", "1HZ75V", "1HZ100V",
    # Jump Indices
    "JD10", "JD25", "JD50", "JD75", "JD100",
    # Step & Skew Step Indices
    "STPIDX", "STEP10", "STEP25", "SKEWSTEP",
    # Boom / Crash Indices
    "BOOM300", "BOOM500", "BOOM1000", "CRASH300", "CRASH500", "CRASH1000",
    # Switch Indices
    "VOLSWITCH", "DRIFTSWITCH",
    # Forex Majors
    "frxEURUSD", "frxGBPUSD", "frxUSDJPY", "frxAUDUSD",
]


class DiscoveredMarket:
    """Holds metadata and lifecycle state for one discovered asset."""

    def __init__(
        self,
        symbol: str,
        display_name: str = "",
        market: str = "synthetic_index",
        submarket: str = "random_index",
        asset_family: str = "volatility",
        status: str = "PAPER_TRADING",  # DISCOVERED | VALIDATED | PAPER_TRADING | ELIGIBLE | ACTIVE
        allowed_contracts: Optional[List[str]] = None,
        is_active_on_deriv: bool = True,
    ):
        self.symbol = symbol
        self.display_name = display_name or symbol
        self.market = market
        self.submarket = submarket
        self.asset_family = asset_family
        self.status = status
        self.allowed_contracts = allowed_contracts or ["DIGITOVER", "DIGITUNDER", "DIGITEVEN", "DIGITODD", "CALL", "PUT"]
        self.is_active_on_deriv = is_active_on_deriv
        self.paper_trades_count: int = 0
        self.paper_wins: int = 0
        self.live_trades_count: int = 0
        self.live_wins: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "display_name": self.display_name,
            "market": self.market,
            "submarket": self.submarket,
            "asset_family": self.asset_family,
            "status": self.status,
            "allowed_contracts": self.allowed_contracts,
            "is_active_on_deriv": self.is_active_on_deriv,
            "paper_trades_count": self.paper_trades_count,
            "paper_wins": self.paper_wins,
            "live_trades_count": self.live_trades_count,
            "live_wins": self.live_wins,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiscoveredMarket":
        m = cls(
            symbol=data["symbol"],
            display_name=data.get("display_name", data["symbol"]),
            market=data.get("market", "synthetic_index"),
            submarket=data.get("submarket", "random_index"),
            asset_family=data.get("asset_family", "volatility"),
            status=data.get("status", "PAPER_TRADING"),
            allowed_contracts=data.get("allowed_contracts"),
            is_active_on_deriv=data.get("is_active_on_deriv", True),
        )
        m.paper_trades_count = data.get("paper_trades_count", 0)
        m.paper_wins = data.get("paper_wins", 0)
        m.live_trades_count = data.get("live_trades_count", 0)
        m.live_wins = data.get("live_wins", 0)
        return m


class MarketDiscoveryManager:
    """Manages dynamic symbol discovery via Deriv API & asset lifecycle pipeline."""

    def __init__(self, state_path: Path = Path("data/market_discovery_state.json")):
        self.state_path = state_path
        self.markets: Dict[str, DiscoveredMarket] = {}
        self._load_state()

    def _classify_asset_family(self, symbol: str, display_name: str = "") -> str:
        s = symbol.upper()
        if s.startswith("BOOM") or s.startswith("CRASH"):
            return "boom_crash"
        if s.startswith("JD") or "JUMP" in s or "JUMP" in display_name.upper():
            return "jump"
        if "SKEW" in s or "SKEW" in display_name.upper():
            return "skew_step"
        if "STEP" in s or "STPIDX" in s or "STEP" in display_name.upper():
            return "step"
        if "SWITCH" in s or "DRIFT" in s:
            return "switch"
        if s.startswith("FRX") or s.startswith("FOREX"):
            return "forex"
        if s.startswith("R_") or "1HZ" in s or "VOLATILITY" in display_name.upper():
            return "volatility"
        return "synthetic_other"

    def _load_state(self) -> None:
        if self.state_path.is_file():
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                for sym, data in raw.items():
                    self.markets[sym] = DiscoveredMarket.from_dict(data)
                logger.info("Loaded %d discovered markets from state file.", len(self.markets))
            except Exception as e:
                logger.error("Failed to load market discovery state: %s", e)
                self._seed_default_markets()
        else:
            self._seed_default_markets()

    def _seed_default_markets(self) -> None:
        for sym in DEFAULT_SYMBOLS_PORTFOLIO:
            fam = self._classify_asset_family(sym)
            status = "ACTIVE" if sym in ("R_10", "R_25", "R_50", "R_75", "R_100", "1HZ10V", "1HZ25V", "1HZ50V", "1HZ75V", "1HZ100V", "JD10", "JD25", "JD50", "STPIDX", "BOOM500", "BOOM1000", "CRASH500", "CRASH1000", "frxEURUSD", "frxGBPUSD") else "PAPER_TRADING"
            self.markets[sym] = DiscoveredMarket(
                symbol=sym,
                display_name=sym,
                asset_family=fam,
                status=status,
            )
        self.save_state()

    def save_state(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {sym: m.to_dict() for sym, m in self.markets.items()}
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("Failed to save market discovery state: %s", e)

    async def discover_markets_from_api(self, client: Any) -> List[str]:
        """
        Query Deriv API for active_symbols and update market discovery registry.
        """
        if not client or not getattr(client, "connected", False):
            logger.warning("DerivClient not connected — skipping dynamic API market discovery.")
            return self.get_active_symbols()

        try:
            req = {"active_symbols": "brief", "product_type": "basic"}
            resp = await client.request(req, timeout=15.0)
            symbols_list = resp.get("active_symbols") or []
            if not isinstance(symbols_list, list):
                return self.get_active_symbols()

            discovered_count = 0
            for item in symbols_list:
                if not isinstance(item, dict):
                    continue
                sym = item.get("symbol")
                if not sym:
                    continue
                is_active = bool(item.get("is_trading_suspended") == 0)
                display_name = str(item.get("display_name") or sym)
                market_type = str(item.get("market") or "synthetic_index")
                submarket_type = str(item.get("submarket") or "random_index")

                fam = self._classify_asset_family(sym, display_name)

                if sym not in self.markets:
                    # New market discovered! Starts in DISCOVERED / PAPER_TRADING
                    self.markets[sym] = DiscoveredMarket(
                        symbol=sym,
                        display_name=display_name,
                        market=market_type,
                        submarket=submarket_type,
                        asset_family=fam,
                        status="PAPER_TRADING",
                        is_active_on_deriv=is_active,
                    )
                    discovered_count += 1
                else:
                    self.markets[sym].is_active_on_deriv = is_active
                    self.markets[sym].display_name = display_name

            if discovered_count > 0:
                logger.info("Discovered %d new markets from Deriv API!", discovered_count)
                self.save_state()

        except Exception as e:
            logger.error("Failed to discover markets from Deriv API: %s", e)

        return self.get_active_symbols()

    def get_active_symbols(self) -> List[str]:
        """Return symbols that are currently ACTIVE or ELIGIBLE for live scanning."""
        return [
            sym for sym, m in self.markets.items()
            if m.status in ("ACTIVE", "ELIGIBLE") and m.is_active_on_deriv
        ]

    def get_paper_symbols(self) -> List[str]:
        """Return symbols currently in PAPER_TRADING state."""
        return [
            sym for sym, m in self.markets.items()
            if m.status in ("PAPER_TRADING", "VALIDATED", "DISCOVERED") and m.is_active_on_deriv
        ]

    def record_paper_trade(self, symbol: str, is_win: bool) -> None:
        """Record paper trading result and evaluate lifecycle promotion."""
        if symbol in self.markets:
            m = self.markets[symbol]
            m.paper_trades_count += 1
            if is_win:
                m.paper_wins += 1

            # Promotion rule: 20+ paper trades with win rate >= 52% promotes to ELIGIBLE / ACTIVE
            if m.status == "PAPER_TRADING" and m.paper_trades_count >= 20:
                win_rate = m.paper_wins / max(1, m.paper_trades_count)
                if win_rate >= 0.52:
                    m.status = "ACTIVE"
                    logger.info("PROMOTED market %s to ACTIVE! Paper win rate: %.1f%% (%d trades)", symbol, win_rate * 100, m.paper_trades_count)
                    self.save_state()

    def get_discovery_stats(self) -> Dict[str, Any]:
        """Return summary stats of discovered markets for dashboard UI."""
        total = len(self.markets)
        active = sum(1 for m in self.markets.values() if m.status in ("ACTIVE", "ELIGIBLE"))
        paper = sum(1 for m in self.markets.values() if m.status == "PAPER_TRADING")
        by_family: Dict[str, int] = {}
        for m in self.markets.values():
            by_family[m.asset_family] = by_family.get(m.asset_family, 0) + 1

        return {
            "total_discovered": total,
            "active_count": active,
            "paper_count": paper,
            "by_family": by_family,
        }
