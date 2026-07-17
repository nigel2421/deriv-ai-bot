"""
Tick-replay backtester for digit strategies.

Walks historical ticks, generates signals (heuristic or Predictor),
applies Martingale/Zuno via StrategyEngine, and settles with would_win()
using the last digit at trade expiry.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.ai.predictor import Predictor
from src.strategy.digit_contracts import extract_last_digit, would_win
from src.strategy.signal_generator import SignalGenerator
from src.strategy.strategy_engine import StrategyEngine
from src.strategy.xml_parser import XMLStrategyParser

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    index: int
    symbol: str
    contract_type: str
    barrier: Optional[int]
    stake: float
    confidence: float
    strategy: str
    settlement_digit: int
    is_win: bool
    profit: float
    balance_after: float
    epoch: Optional[int] = None


@dataclass
class BacktestResult:
    symbol: str
    initial_balance: float
    final_balance: float
    trades: int
    wins: int
    losses: int
    pushes: int
    win_rate: float
    total_profit: float
    max_drawdown: float
    max_drawdown_pct: float
    profit_factor: float
    avg_stake: float
    trade_log: List[TradeRecord] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["trade_log"] = [asdict(t) for t in self.trade_log]
        return d

    def summary(self) -> str:
        return (
            f"{self.symbol}: trades={self.trades} W/L={self.wins}/{self.losses} "
            f"WR={self.win_rate:.1%} PnL={self.total_profit:.2f} "
            f"bal {self.initial_balance:.2f}→{self.final_balance:.2f} "
            f"maxDD={self.max_drawdown:.2f} ({self.max_drawdown_pct:.1%}) "
            f"PF={self.profit_factor:.2f}"
        )


class BacktestEngine:
    """Replay digit strategy on a historical tick series."""

    def __init__(
        self,
        *,
        symbol: str = "R_100",
        initial_balance: float = 1000.0,
        min_confidence: float = 0.55,
        duration_ticks: int = 5,
        win_payout: float = 0.95,
        warmup: int = 50,
        step: int = 1,
        use_model: bool = True,
        strategy_xml: str = "config/strategy.xml",
        max_trades: Optional[int] = None,
    ):
        self.symbol = symbol
        self.initial_balance = float(initial_balance)
        self.min_confidence = float(min_confidence)
        self.duration_ticks = int(duration_ticks)
        self.win_payout = float(win_payout)
        self.warmup = int(warmup)
        self.step = max(1, int(step))
        self.use_model = use_model
        self.max_trades = max_trades

        self.parser = XMLStrategyParser(strategy_xml)
        self.engine = StrategyEngine(self.parser)
        self.signal_gen = SignalGenerator()
        self.predictor = Predictor(auto_load=use_model) if use_model else None
        # Fresh strategy runtime for this symbol
        cfg = self.parser.get_strategy(symbol)
        cfg = dict(cfg)
        cfg["symbol"] = symbol
        from src.strategy.strategy_engine import MarketRuntime

        self.runtime = MarketRuntime(symbol, cfg)
        self.engine.runtimes[symbol] = self.runtime

    def _ticks_from_df(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        rows = []
        for _, r in df.iterrows():
            quote = r.get("quote")
            if quote is None or (isinstance(quote, float) and np.isnan(quote)):
                continue
            ep = r.get("epoch")
            try:
                epoch = int(ep) if ep is not None and not pd.isna(ep) else None
            except (TypeError, ValueError):
                epoch = None
            rows.append(
                {
                    "symbol": self.symbol,
                    "quote": float(quote),
                    "epoch": epoch or 0,
                }
            )
        return rows

    def run(self, ticks_df: pd.DataFrame) -> BacktestResult:
        ticks = self._ticks_from_df(ticks_df)
        n = len(ticks)
        if n < self.warmup + self.duration_ticks + 1:
            logger.error(
                "Not enough ticks (%d) for warmup=%d duration=%d",
                n,
                self.warmup,
                self.duration_ticks,
            )
            return BacktestResult(
                symbol=self.symbol,
                initial_balance=self.initial_balance,
                final_balance=self.initial_balance,
                trades=0,
                wins=0,
                losses=0,
                pushes=0,
                win_rate=0.0,
                total_profit=0.0,
                max_drawdown=0.0,
                max_drawdown_pct=0.0,
                profit_factor=0.0,
                avg_stake=0.0,
            )

        # Reset strategy state
        if self.runtime.martingale:
            self.runtime.martingale.reset()
        if self.runtime.zuno:
            # re-init type
            cfg = self.runtime.cfg
            self.runtime.zuno.current_type = (
                cfg.get("initial_type") or cfg.get("switch_on_win") or "DIGITOVER"
            )

        balance = self.initial_balance
        peak = balance
        max_dd = 0.0
        equity = [balance]
        log: List[TradeRecord] = []
        wins = losses = pushes = 0
        gross_profit = 0.0
        gross_loss = 0.0
        stakes: List[float] = []

        allowed = self.runtime.allowed_types or None
        i = self.warmup
        while i < n - self.duration_ticks:
            if self.max_trades is not None and len(log) >= self.max_trades:
                break
            if not self.runtime.is_tradeable():
                logger.info("Strategy inactive at i=%s — stop backtest for symbol", i)
                break

            window = ticks[: i + 1]
            # Prediction
            if self.predictor is not None:
                pred = self.predictor.predict(window[-max(self.warmup, 60) :])
            else:
                from src.strategy.digit_contracts import last_digits_from_ticks
                from collections import Counter

                digits = last_digits_from_ticks(window, n=50)
                if not digits:
                    i += self.step
                    continue
                mode, cnt = Counter(digits).most_common(1)[0]
                pred = {
                    "digit": mode,
                    "confidence": min(0.9, 0.4 + cnt / len(digits)),
                    "parity": mode % 2 == 0,
                    "source": "bt_freq",
                }

            conf = float(pred.get("confidence", 0.5))
            sig_type, sig_barrier, conf_out = self.signal_gen.generate_signal(
                {**pred, "recent_ticks": window},
                conf,
                self.min_confidence,
                allowed_types=allowed,
            )

            if not sig_type and not self.runtime.zuno:
                i += self.step
                continue
            if not sig_type and self.runtime.zuno:
                if conf < self.min_confidence:
                    i += self.step
                    continue
                conf_out = conf

            intent = self.engine.apply_signal(
                self.symbol,
                sig_type,
                sig_barrier,
                conf_out or conf,
            )
            if not intent:
                i += self.step
                continue

            stake = float(intent["stake"])
            if stake <= 0 or stake > balance:
                i += self.step
                continue

            settle_idx = i + self.duration_ticks
            settle_digit = extract_last_digit(ticks[settle_idx]["quote"])
            if settle_digit is None:
                i += self.step
                continue

            ct = intent["contract_type"]
            barrier = intent.get("barrier")
            won = would_win(ct, barrier, settle_digit)
            # Push: treat as no win/loss for DIGIT contracts rarely exact push —
            # we only have win/loss here
            if won:
                profit = stake * self.win_payout
                wins += 1
                gross_profit += profit
            else:
                profit = -stake
                losses += 1
                gross_loss += stake

            balance += profit
            stakes.append(stake)
            peak = max(peak, balance)
            dd = peak - balance
            max_dd = max(max_dd, dd)
            equity.append(balance)

            self.runtime.on_trade_result(is_win=won, profit=profit)

            log.append(
                TradeRecord(
                    index=i,
                    symbol=self.symbol,
                    contract_type=ct,
                    barrier=barrier if barrier is None else int(barrier),
                    stake=stake,
                    confidence=float(intent.get("confidence") or conf),
                    strategy=str(intent.get("strategy") or self.runtime.stype),
                    settlement_digit=int(settle_digit),
                    is_win=won,
                    profit=float(profit),
                    balance_after=float(balance),
                    epoch=ticks[i].get("epoch"),
                )
            )

            # Jump past settlement to avoid overlapping contracts (simple model)
            i = settle_idx + self.step

        trades = len(log)
        wr = wins / trades if trades else 0.0
        pf = (gross_profit / gross_loss) if gross_loss > 0 else (
            float("inf") if gross_profit > 0 else 0.0
        )
        max_dd_pct = (max_dd / peak) if peak > 0 else 0.0

        result = BacktestResult(
            symbol=self.symbol,
            initial_balance=self.initial_balance,
            final_balance=balance,
            trades=trades,
            wins=wins,
            losses=losses,
            pushes=pushes,
            win_rate=wr,
            total_profit=balance - self.initial_balance,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            profit_factor=float(pf) if pf != float("inf") else 999.0,
            avg_stake=float(np.mean(stakes)) if stakes else 0.0,
            trade_log=log,
            equity_curve=equity,
            extras={
                "warmup": self.warmup,
                "duration_ticks": self.duration_ticks,
                "min_confidence": self.min_confidence,
                "win_payout": self.win_payout,
                "use_model": self.use_model,
                "n_ticks": n,
            },
        )
        logger.info(result.summary())
        return result


def load_ticks_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "quote" not in df.columns:
        raise ValueError(f"CSV missing 'quote' column: {path}")
    return df
