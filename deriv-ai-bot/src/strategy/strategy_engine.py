"""
Per-symbol strategy runtime: applies Martingale stake and/or Zuno type
rules on top of AI signals using config from strategy.xml.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from src.strategy.barrier_picker import adaptive_barrier
from src.strategy.contract_types import (
    is_rise_fall,
    normalize_contract_type,
    requires_barrier,
    validate_contract,
)
from src.strategy.digit_contracts import (
    DEFAULT_OVER_BARRIER,
    DEFAULT_UNDER_BARRIER,
    normalize_barrier,
)
from src.strategy.martingale import MartingaleStrategy
from src.strategy.zuno_strategy import ZunoStrategy
from src.strategy.xml_parser import XMLStrategyParser

logger = logging.getLogger(__name__)


class MarketRuntime:
    """Live state for one market symbol."""

    def __init__(self, symbol: str, cfg: Dict[str, Any]):
        self.symbol = symbol
        self.cfg = cfg
        self.stype = (cfg.get("type") or "flat").lower()
        self.base_stake = float(cfg.get("base_stake", 1.0))
        self.allowed_types: List[str] = list(cfg.get("contract_types") or [])
        self.default_barrier = int(cfg.get("default_barrier", 4))
        # Fallback / fixed-mode barriers (adaptive is default)
        self.over_barrier = int(cfg.get("over_barrier", DEFAULT_OVER_BARRIER))
        self.under_barrier = int(cfg.get("under_barrier", DEFAULT_UNDER_BARRIER))
        # adaptive | fixed | random — adaptive picks barrier from ticks + prediction
        self.barrier_mode = str(cfg.get("barrier_mode") or "adaptive").strip().lower()
        self.duration = int(cfg.get("duration", 5))

        self.martingale: Optional[MartingaleStrategy] = None
        self.zuno: Optional[ZunoStrategy] = None

        if self.stype in {"martingale", "martingale_zuno", "hybrid"}:
            self.martingale = MartingaleStrategy(
                base_stake=self.base_stake,
                max_steps=int(cfg.get("max_steps", 6)),
            )

        if self.stype in {"zuno", "martingale_zuno", "hybrid"}:
            win_t = cfg.get("switch_on_win") or "DIGITOVER"
            loss_t = cfg.get("switch_on_loss") or "DIGITUNDER"
            initial = cfg.get("initial_type") or win_t
            self.zuno = ZunoStrategy(
                switch_on_win=win_t,
                switch_on_loss=loss_t,
                initial_type=initial,
            )

        logger.info(
            "MarketRuntime %s type=%s martingale=%s zuno=%s",
            symbol,
            self.stype,
            bool(self.martingale),
            bool(self.zuno),
        )

    def is_tradeable(self) -> bool:
        if self.martingale and not self.martingale.active:
            return False
        return True

    def resolve_stake(self) -> float:
        if self.martingale:
            return self.martingale.peek_stake()
        return self.base_stake

    def resolve_contract(
        self,
        signal_type: Optional[str],
        signal_barrier: Optional[int],
        *,
        predicted_digit: Optional[int] = None,
        ticks: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Optional[str], Optional[int]]:
        """
        Choose contract type + barrier for the next trade.

        Barriers for OVER/UNDER use adaptive picker by default (recent digits +
        prediction), not fixed 6/4. EVEN/ODD and CALL/PUT have no barrier.
        """
        contract_type: Optional[str] = None
        barrier: Optional[int] = signal_barrier

        if self.zuno:
            contract_type = self.zuno.peek_type()
        elif signal_type:
            contract_type = normalize_contract_type(signal_type)
        elif self.allowed_types:
            contract_type = normalize_contract_type(self.allowed_types[0])

        if not contract_type:
            return None, None

        contract_type = normalize_contract_type(contract_type)
        if not contract_type:
            return None, None

        # Filter against allow-list when present
        allowed_norm = [
            t for t in (normalize_contract_type(x) for x in self.allowed_types) if t
        ]
        if allowed_norm and contract_type not in allowed_norm:
            sig = normalize_contract_type(signal_type) if signal_type else None
            if sig and sig in allowed_norm:
                contract_type = sig
            elif not self.zuno:
                contract_type = allowed_norm[0]
            else:
                logger.warning(
                    "%s: type %s not in allow-list %s — keeping strategy type",
                    self.symbol,
                    contract_type,
                    allowed_norm,
                )

        barrier_meta: Dict[str, Any] = {}
        if is_rise_fall(contract_type):
            barrier = None
        elif requires_barrier(contract_type):
            if contract_type in {"DIGITOVER", "DIGITUNDER"}:
                # Prefer adaptive barrier; signal_barrier is a soft hint only
                barrier, barrier_meta = adaptive_barrier(
                    contract_type,
                    predicted_digit=predicted_digit,
                    ticks=ticks,
                    mode=self.barrier_mode,
                    fixed_over=self.over_barrier,
                    fixed_under=self.under_barrier,
                )
                # If adaptive failed, fall back to signal then fixed
                if barrier is None and signal_barrier is not None:
                    barrier = normalize_barrier(contract_type, signal_barrier)
                if barrier is None:
                    barrier = (
                        self.over_barrier
                        if contract_type == "DIGITOVER"
                        else self.under_barrier
                    )
            elif barrier is None:
                barrier = self.default_barrier
            barrier = normalize_barrier(
                contract_type,
                barrier,
                default_over=self.over_barrier,
                default_under=self.under_barrier,
            )
            if barrier_meta:
                logger.info(
                    "%s barrier %s@%s mode=%s emp_wr=%s pred=%s",
                    self.symbol,
                    contract_type,
                    barrier,
                    barrier_meta.get("mode"),
                    barrier_meta.get("win_rate"),
                    barrier_meta.get("pred"),
                )
        else:
            barrier = None

        ok, reason, barrier = validate_contract(contract_type, barrier)
        if not ok:
            logger.error(
                "%s invalid contract after resolve: %s %s (%s)",
                self.symbol,
                contract_type,
                barrier,
                reason,
            )
            return None, None

        return contract_type, barrier

    def on_trade_result(self, is_win: bool, profit: float = 0.0) -> None:
        if self.martingale:
            self.martingale.on_result(is_win)
        if self.zuno:
            self.zuno.on_result(is_win)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "type": self.stype,
            "base_stake": self.base_stake,
            "tradeable": self.is_tradeable(),
            "next_stake": self.resolve_stake(),
            "next_type": self.zuno.peek_type() if self.zuno else None,
            "martingale": self.martingale.snapshot() if self.martingale else None,
            "zuno": self.zuno.snapshot() if self.zuno else None,
            "allowed_types": self.allowed_types,
            "over_barrier": self.over_barrier,
            "under_barrier": self.under_barrier,
            "barrier_mode": self.barrier_mode,
        }


class StrategyEngine:
    """Holds one MarketRuntime per configured symbol."""

    def __init__(self, parser: Optional[XMLStrategyParser] = None):
        self.parser = parser or XMLStrategyParser()
        self.runtimes: Dict[str, MarketRuntime] = {}
        for symbol, cfg in self.parser.config.get("markets", {}).items():
            self.runtimes[symbol] = MarketRuntime(symbol, cfg)

    def get(self, symbol: str) -> MarketRuntime:
        if symbol not in self.runtimes:
            cfg = self.parser.get_strategy(symbol)
            cfg = dict(cfg)
            cfg["symbol"] = symbol
            self.runtimes[symbol] = MarketRuntime(symbol, cfg)
            logger.info("Created on-demand runtime for %s: %s", symbol, cfg.get("type"))
        return self.runtimes[symbol]

    def apply_signal(
        self,
        symbol: str,
        signal_type: Optional[str],
        signal_barrier: Optional[int],
        confidence: float,
        *,
        predicted_digit: Optional[int] = None,
        ticks: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Merge AI signal with strategy rules → trade intent dict, or None to skip.
        """
        runtime = self.get(symbol)
        if not runtime.is_tradeable():
            logger.warning(
                "%s strategy not tradeable (e.g. martingale deactivated)", symbol
            )
            return None

        contract_type, barrier = runtime.resolve_contract(
            signal_type,
            signal_barrier,
            predicted_digit=predicted_digit,
            ticks=ticks,
        )
        if not contract_type:
            return None

        stake = runtime.resolve_stake()
        if stake <= 0:
            logger.warning("%s stake is 0 — skip", symbol)
            return None

        return {
            "symbol": symbol,
            "contract_type": contract_type,
            "barrier": barrier,
            "confidence": confidence,
            "strategy": runtime.stype,
            "base_stake": runtime.base_stake,
            "stake": stake,
            "duration": runtime.duration,
            "predicted_digit": predicted_digit,
            "barrier_mode": runtime.barrier_mode,
            "strategy_snapshot": runtime.snapshot(),
        }

    def on_trade_result(self, symbol: str, is_win: bool, profit: float = 0.0) -> None:
        if symbol in self.runtimes or symbol:
            self.get(symbol).on_trade_result(is_win, profit)

    def snapshots(self) -> Dict[str, Any]:
        return {s: r.snapshot() for s, r in self.runtimes.items()}
