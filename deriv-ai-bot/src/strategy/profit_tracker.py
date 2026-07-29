"""
Profit Tracker — Phase 1, 2, 3

Tracks Gross Profit and Gross Loss to compute Profit Factor.
Computes Market Profitability Score (MPS) and manages Market Retirement.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class ProfitTracker:
    def __init__(self, path: Optional[Path] = None):
        self.path = path or Path("data/profit_state.json")
        self.state = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.path.is_file():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("ProfitTracker load failed: %s", e)
        return {"markets": {}}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        except Exception as e:
            logger.debug("ProfitTracker save failed: %s", e)

    def record_trade(self, symbol: str, contract_type: str, profit: float) -> None:
        """Called on every settled trade to update gross profit/loss."""
        markets = self.state.setdefault("markets", {})
        mkt = markets.setdefault(symbol, {"gross_profit": 0.0, "gross_loss": 0.0, "trades": 0, "contracts": {}})
        ct = mkt["contracts"].setdefault(contract_type, {"gross_profit": 0.0, "gross_loss": 0.0, "trades": 0})
        
        mkt["trades"] += 1
        ct["trades"] += 1
        
        if profit > 0:
            mkt["gross_profit"] += profit
            ct["gross_profit"] += profit
        elif profit < 0:
            mkt["gross_loss"] += abs(profit)
            ct["gross_loss"] += abs(profit)
            
        self._save()

    def get_profit_factor(self, symbol: str, contract_type: Optional[str] = None) -> float:
        """
        Calculate Profit Factor (Gross Profit / Gross Loss).
        Returns 1.0 (breakeven) if no losses yet or no data.
        """
        markets = self.state.get("markets", {})
        mkt = markets.get(symbol)
        if not mkt:
            return 1.0
            
        target = mkt
        if contract_type:
            target = mkt.get("contracts", {}).get(contract_type)
            if not target:
                return 1.0
                
        gp = float(target.get("gross_profit", 0.0))
        gl = float(target.get("gross_loss", 0.0))
        
        if gl == 0:
            # Discovery allowance or perfect win streak
            return max(gp, 1.0)
            
        return gp / gl

    def get_trade_count(self, symbol: str, contract_type: Optional[str] = None) -> int:
        """Get the number of trades for a symbol (and optionally contract_type)."""
        markets = self.state.get("markets", {})
        mkt = markets.get(symbol)
        if not mkt:
            return 0
            
        if contract_type:
            ct = mkt.get("contracts", {}).get(contract_type)
            if not ct:
                return 0
            return ct.get("trades", 0)
            
        return mkt.get("trades", 0)

    def get_mps(self, symbol: str, contract_type: Optional[str] = None) -> float:
        """
        Calculate Max Profit Score (MPS) based on empirical historical profit.
        Returns 0-100 score based on Profit Factor.
        """
        pf = self.get_profit_factor(symbol, contract_type)
        return min(100.0, max(0.0, pf * 50.0))

    def top_n_markets(self, active_symbols: set, n: int = 5) -> set:
        """
        Rank active symbols by their aggregate Max Profit Score (MPS)
        and return the top N. Used for Market Capital Allocation.
        """
        ranked = []
        for sym in active_symbols:
            mps = self.get_mps(sym)
            # Edge Discovery Mode: Boost new markets so they get a chance to reach 50 trades
            if self.get_trade_count(sym) < 50:
                mps = max(mps, 85.0)  
            ranked.append((sym, mps))
            
        ranked.sort(key=lambda x: x[1], reverse=True)
        top_n = [x[0] for x in ranked[:n]]
        return set(top_n)

    def compute_mps(self, symbol: str, pf: float, ev: float, hpp: float, hpp_velocity: float, trade_quality: float) -> float:
        """
        Phase 2: Market Profitability Score (MPS)
        MPS = 40% PF + 25% EV + 15% HPP + 10% HPP Velocity + 10% Trade Quality
        Assumes inputs are normalized (or normalizes them here).
        """
        # Normalize PF: e.g., PF=1.0 -> 50, PF=2.0 -> 100
        pf_score = min(100.0, max(0.0, pf * 50.0))
        
        # Normalize EV: e.g., EV=0.0 -> 50, EV=0.5 -> 100 (for 0 to 100 scale)
        ev_score = min(100.0, max(0.0, (ev * 100.0) + 50.0))
        
        # HPP is naturally 0-100. Velocity is -100 to +100 (shift to 0-100).
        vel_score = min(100.0, max(0.0, (hpp_velocity + 50.0)))
        
        mps = (0.40 * pf_score) + (0.25 * ev_score) + (0.15 * hpp) + (0.10 * vel_score) + (0.10 * trade_quality)
        return mps

    def check_retirement(self, symbol: str, pf: float, ev: float, hpp_velocity: float) -> Tuple[bool, Optional[str]]:
        """
        Phase 3: Market Retirement System
        Ban market for 24h if trades >= 200 AND PF < 1 AND EV < 0 AND HPP Velocity < 0.
        """
        markets = self.state.get("markets", {})
        mkt = markets.get(symbol)
        if not mkt:
            return False, None
            
        now = time.time()
        retired_until = mkt.get("retired_until", 0)
        
        # Check active retirement
        if retired_until > now:
            remaining = int((retired_until - now) / 60)
            return True, f"Retired for {remaining}m ({mkt.get('retire_reason', 'poor performance')})"
            
        # Check for new retirement
        trades = mkt.get("trades", 0)
        if trades >= 200 and pf < 1.0 and ev < 0.0 and hpp_velocity < 0.0:
            mkt["retired_until"] = now + 86400  # 24 hours
            reason = f"PF={pf:.2f} EV={ev:.3f} Vel={hpp_velocity:.1f}"
            mkt["retire_reason"] = reason
            self._save()
            logger.warning("Market %s RETIRED for 24h: %s", symbol, reason)
            return True, f"Retired for 24h ({reason})"
            
        return False, None

    def snapshot(self) -> Dict[str, Any]:
        """Return full state for dashboard."""
        return self.state
