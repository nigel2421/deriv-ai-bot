"""
Decision Audit, Shadow Trading, and Gate Effectiveness Subsystem.

- DecisionAuditor: Records full decision traces for every evaluated market opportunity.
- ShadowTrader: Tracks hypothetical exit outcomes for rejected opportunities (NEVER executes real trades).
- GateAnalytics: Computes statistical gate effectiveness metrics (Accepted WR vs Shadow Rejected WR).
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Minimum sample size required before declaring a gate POSITIVE or NEGATIVE
MIN_SAMPLE_FOR_GATE_VERDICT = 30


@dataclass
class DecisionTrace:
    symbol: str
    market_regime: str = "NORMAL"
    proposed_contract_type: Optional[str] = None
    proposed_duration: int = 5
    proposed_duration_unit: str = "t"
    original_agent_confidence: float = 0.0
    individual_specialist_votes: List[Dict[str, Any]] = field(default_factory=list)
    agent_reputation_weights: Dict[str, float] = field(default_factory=dict)
    consensus_score: float = 0.0
    quorum_result: bool = True
    htf_alignment_result: bool = True
    atr_volatility_value: float = 1.0
    adaptive_expiry_selected: int = 5
    boom_crash_cooldown_state: bool = False
    live_payout: float = 0.87
    estimated_probability: float = 0.0
    expected_value: float = 0.0
    portfolio_manager_decision: bool = True
    final_decision: str = "REJECTED_SHADOW"  # EXECUTED | REJECTED_SHADOW | REJECTED_STOPPED
    rejection_reason: Optional[str] = None
    trade_id: Optional[int] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    audit_id: Optional[int] = None


@dataclass
class ShadowTrade:
    shadow_id: str
    audit_id: Optional[int]
    symbol: str
    contract_type: str
    rejection_gate: str
    shadow_entry_price: float = 0.0
    shadow_duration: int = 5
    shadow_duration_unit: str = "t"
    shadow_exit_price: Optional[float] = None
    shadow_result: Optional[str] = None  # WIN | LOSS
    shadow_profit_loss: Optional[float] = None
    opened_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    closed_at: Optional[str] = None
    ticks_tracked: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "pending"  # pending | completed

    def __init__(
        self,
        shadow_id: str,
        audit_id: Optional[int] = None,
        symbol: str = "",
        contract_type: str = "",
        rejection_gate: str = "",
        shadow_entry_price: float = 0.0,
        shadow_duration: int = 5,
        shadow_duration_unit: str = "t",
        shadow_exit_price: Optional[float] = None,
        shadow_result: Optional[str] = None,
        shadow_profit_loss: Optional[float] = None,
        opened_at: Optional[str] = None,
        closed_at: Optional[str] = None,
        ticks_tracked: Optional[List[Dict[str, Any]]] = None,
        status: str = "pending",
        # Backward-compatibility / alias kwargs for unit tests
        entry_price: Optional[float] = None,
        target_expiry_ticks: Optional[int] = None,
        duration: Optional[int] = None,
        duration_unit: Optional[str] = None,
        outcome: Optional[str] = None,
        profit: Optional[float] = None,
    ):
        self.shadow_id = shadow_id
        self.audit_id = audit_id
        self.symbol = symbol
        self.contract_type = contract_type
        self.rejection_gate = rejection_gate
        self.shadow_entry_price = entry_price if entry_price is not None else shadow_entry_price
        self.shadow_duration = (
            duration if duration is not None
            else (target_expiry_ticks if target_expiry_ticks is not None else shadow_duration)
        )
        self.shadow_duration_unit = duration_unit if duration_unit is not None else shadow_duration_unit
        self.shadow_exit_price = shadow_exit_price
        self.shadow_result = outcome if outcome is not None else shadow_result
        self.shadow_profit_loss = profit if profit is not None else shadow_profit_loss
        self.opened_at = opened_at or datetime.now(timezone.utc).isoformat()
        self.closed_at = closed_at
        self.ticks_tracked = ticks_tracked or []
        self.status = status.lower() if status in ("SETTLED", "completed") else status

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DecisionAuditor:
    """Logs full decision traces to PostgreSQL and local append-only JSONL storage."""

    def __init__(self, jsonl_path: Optional[Path] = None):
        self.jsonl_path = jsonl_path or Path("data/decision_audit.jsonl")
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._memory_traces: List[DecisionTrace] = []

    @property
    def traces(self) -> List[DecisionTrace]:
        return self._memory_traces

    def record_trace(self, trace: DecisionTrace) -> int:
        """Record decision trace to memory and JSONL log."""
        self._memory_traces.append(trace)
        if len(self._memory_traces) > 500:
            self._memory_traces = self._memory_traces[-500:]

        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(trace)) + "\n")
        except Exception as e:
            logger.warning("Failed to append decision trace to JSONL: %s", e)

        # Non-blocking DB log attempt if loop is running
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._persist_db(trace))
        except RuntimeError:
            pass

        return len(self._memory_traces)

    async def _persist_db(self, trace: DecisionTrace) -> None:
        try:
            from src.database.db import get_db
            db = await get_db()
            if db and hasattr(db, "execute"):
                query = """
                    INSERT INTO decision_audit (
                        symbol, market_regime, proposed_contract_type, proposed_duration, proposed_duration_unit,
                        original_agent_confidence, individual_specialist_votes, agent_reputation_weights,
                        consensus_score, quorum_result, htf_alignment_result, atr_volatility_value,
                        adaptive_expiry_selected, boom_crash_cooldown_state, live_payout, estimated_probability,
                        expected_value, portfolio_manager_decision, final_decision, rejection_reason, trade_id
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21
                    ) RETURNING id;
                """
                res = await db.fetchrow(
                    query,
                    trace.symbol,
                    trace.market_regime,
                    trace.proposed_contract_type,
                    trace.proposed_duration,
                    trace.proposed_duration_unit,
                    trace.original_agent_confidence,
                    json.dumps(trace.individual_specialist_votes),
                    json.dumps(trace.agent_reputation_weights),
                    trace.consensus_score,
                    trace.quorum_result,
                    trace.htf_alignment_result,
                    trace.atr_volatility_value,
                    trace.adaptive_expiry_selected,
                    trace.boom_crash_cooldown_state,
                    trace.live_payout,
                    trace.estimated_probability,
                    trace.expected_value,
                    trace.portfolio_manager_decision,
                    trace.final_decision,
                    trace.rejection_reason,
                    trace.trade_id,
                )
                if res and "id" in res:
                    trace.audit_id = res["id"]
        except Exception as e:
            logger.debug("DB decision_audit persist skipped/failed: %s", e)

    def rejection_funnel_counts(self) -> Dict[str, Any]:
        funnel = {"total_scanned": len(self._memory_traces), "executed": 0, "by_gate": {}}
        for t in self._memory_traces:
            if t.final_decision == "EXECUTED":
                funnel["executed"] += 1
            else:
                reason = t.rejection_reason or "other"
                funnel["by_gate"][reason] = funnel["by_gate"].get(reason, 0) + 1
        return funnel

    def recent_traces(self, limit: int = 50) -> List[Dict[str, Any]]:
        return [asdict(t) for t in self._memory_traces[-limit:]]


class ShadowTrader:
    """
    Shadow Trading Subsystem.
    Tracks hypothetical trade exit outcomes for opportunities rejected by gates.
    CRITICAL: NEVER executes real API calls or live trades on Deriv.
    """

    def __init__(self, jsonl_path: Optional[Path] = None):
        self.jsonl_path = jsonl_path or Path("data/shadow_trades.jsonl")
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.pending_shadows: Dict[str, ShadowTrade] = {}
        self.completed_shadows: List[ShadowTrade] = []
        self._load_history()

    @property
    def shadow_trades(self) -> Dict[str, ShadowTrade]:
        all_st = {**self.pending_shadows}
        for s in self.completed_shadows:
            all_st[s.shadow_id] = s
        return all_st

    def _load_history(self) -> None:
        if self.jsonl_path.exists():
            try:
                with open(self.jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        st = ShadowTrade(**data)
                        if st.status in ("completed", "settled"):
                            self.completed_shadows.append(st)
                        elif st.status == "pending":
                            self.pending_shadows[st.shadow_id] = st
            except Exception as e:
                logger.warning("ShadowTrader history load error: %s", e)

    def spawn_shadow_trade(
        self,
        symbol: str,
        contract_type: str,
        rejection_gate: str,
        entry_price: float,
        duration: int = 5,
        duration_unit: str = "t",
        audit_id: Optional[int] = None,
    ) -> str:
        """Create a new pending shadow trade and return shadow_id."""
        shadow_id = f"SHADOW_{symbol}_{datetime.now(timezone.utc).strftime('%H%M%S%f')}"
        st = ShadowTrade(
            shadow_id=shadow_id,
            audit_id=audit_id,
            symbol=symbol,
            contract_type=contract_type,
            rejection_gate=rejection_gate,
            shadow_entry_price=entry_price,
            shadow_duration=duration,
            shadow_duration_unit=duration_unit,
            ticks_tracked=[{"quote": entry_price, "epoch": datetime.now(timezone.utc).timestamp()}],
        )
        self.pending_shadows[shadow_id] = st
        logger.info(
            "Spawned SHADOW TRADE %s (symbol=%s ct=%s gate=%s entry=%.4f dur=%s%s)",
            shadow_id, symbol, contract_type, rejection_gate, entry_price, duration, duration_unit
        )
        return shadow_id

    def on_tick(self, symbol: str, quote: float, epoch: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Feed new tick update to active shadow trades for symbol.
        Evaluates completion when shadow_duration is reached.
        """
        closed_this_tick: List[Dict[str, Any]] = []
        now_epoch = epoch or datetime.now(timezone.utc).timestamp()

        for sid, st in list(self.pending_shadows.items()):
            if st.symbol != symbol:
                continue

            st.ticks_tracked.append({"quote": quote, "epoch": now_epoch})
            unit = str(st.shadow_duration_unit).lower()

            # Check if duration expired
            expired = False
            if unit == "t":
                if len(st.ticks_tracked) >= (st.shadow_duration + 1):
                    expired = True
            elif unit in ("m", "s"):
                start_ep = st.ticks_tracked[0].get("epoch", now_epoch)
                target_sec = st.shadow_duration * 60.0 if unit == "m" else float(st.shadow_duration)
                if (now_epoch - start_ep) >= target_sec:
                    expired = True

            if expired:
                st.shadow_exit_price = quote
                st.closed_at = datetime.now(timezone.utc).isoformat()
                st.status = "SETTLED"
                
                # Evaluate win/loss result
                won, pnl = self._evaluate_outcome(st.contract_type, st.shadow_entry_price, quote)
                st.shadow_result = "WIN" if won else "LOSS"
                st.shadow_profit_loss = pnl

                self.pending_shadows.pop(sid, None)
                self.completed_shadows.append(st)
                closed_this_tick.append({
                    "shadow_id": st.shadow_id,
                    "symbol": st.symbol,
                    "contract_type": st.contract_type,
                    "status": "SETTLED",
                    "outcome": st.shadow_result,
                    "profit": pnl,
                })

                self._persist_shadow(st)
                logger.info(
                    "SHADOW TRADE COMPLETED %s: symbol=%s ct=%s gate=%s entry=%.4f exit=%.4f -> %s (pnl=%.2f)",
                    st.shadow_id, st.symbol, st.contract_type, st.rejection_gate, st.shadow_entry_price, quote, st.shadow_result, pnl
                )

        return closed_this_tick

    def _evaluate_outcome(self, contract_type: str, entry: float, exit_price: float) -> Tuple[bool, float]:
        ct = contract_type.upper()
        payout = 0.87  # Standard 87% net return on win
        
        if ct in ("CALL", "RISE", "HIGHER"):
            won = exit_price > entry
        elif ct in ("PUT", "FALL", "LOWER"):
            won = exit_price < entry
        elif ct in ("DIGITEVEN", "EVEN"):
            last_digit = int(str(exit_price).rstrip("0")[-1]) if "." in str(exit_price) else int(str(exit_price)[-1])
            won = (last_digit % 2 == 0)
        elif ct in ("DIGITODD", "ODD"):
            last_digit = int(str(exit_price).rstrip("0")[-1]) if "." in str(exit_price) else int(str(exit_price)[-1])
            won = (last_digit % 2 != 0)
        elif ct.startswith("DIGITOVER"):
            barrier = int(ct[-1]) if ct[-1].isdigit() else 5
            last_digit = int(str(exit_price).rstrip("0")[-1]) if "." in str(exit_price) else int(str(exit_price)[-1])
            won = last_digit > barrier
        elif ct.startswith("DIGITUNDER"):
            barrier = int(ct[-1]) if ct[-1].isdigit() else 5
            last_digit = int(str(exit_price).rstrip("0")[-1]) if "." in str(exit_price) else int(str(exit_price)[-1])
            won = last_digit < barrier
        else:
            won = exit_price > entry

        pnl = payout if won else -1.0
        return won, pnl

    def _persist_shadow(self, st: ShadowTrade) -> None:
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(st)) + "\n")
        except Exception as e:
            logger.warning("Failed to persist shadow trade: %s", e)

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._persist_db(st))
        except RuntimeError:
            pass

    async def _persist_db(self, st: ShadowTrade) -> None:
        try:
            from src.database.db import get_db
            db = await get_db()
            if db and hasattr(db, "execute"):
                query = """
                    INSERT INTO shadow_trades (
                        decision_audit_id, symbol, contract_type, rejection_gate, shadow_entry_price,
                        shadow_exit_price, shadow_result, shadow_profit_loss, shadow_duration, shadow_duration_unit,
                        opened_at, closed_at, ticks_tracked, status
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
                    );
                """
                await db.execute(
                    query,
                    st.audit_id,
                    st.symbol,
                    st.contract_type,
                    st.rejection_gate,
                    st.shadow_entry_price,
                    st.shadow_exit_price,
                    st.shadow_result,
                    st.shadow_profit_loss,
                    st.shadow_duration,
                    st.shadow_duration_unit,
                    st.opened_at,
                    st.closed_at,
                    json.dumps(st.ticks_tracked),
                    st.status,
                )
        except Exception as e:
            logger.debug("DB shadow_trades persist skipped: %s", e)

    def get_active_and_settled_summary(self) -> Dict[str, Any]:
        return {
            "pending_count": len(self.pending_shadows),
            "completed_count": len(self.completed_shadows),
            "active_shadows": [asdict(s) for s in list(self.pending_shadows.values())[-20:]],
            "recent_completed": [asdict(s) for s in self.completed_shadows[-20:]],
        }

    def summary(self) -> Dict[str, Any]:
        return self.get_active_and_settled_summary()


class GateAnalytics:
    """Computes statistical Gate Effectiveness metrics (Accepted WR vs Shadow Rejected WR)."""

    def __init__(self, shadow_trades: Optional[Any] = None):
        if shadow_trades is None:
            self.shadow_trades_list = []
        elif isinstance(shadow_trades, dict):
            self.shadow_trades_list = list(shadow_trades.values())
        else:
            self.shadow_trades_list = list(shadow_trades)

    def summary(self) -> Dict[str, Any]:
        gates_dict = {}
        by_gate: Dict[str, List[ShadowTrade]] = {}
        for s in self.shadow_trades_list:
            by_gate.setdefault(s.rejection_gate, []).append(s)

        for gate_name, s_list in by_gate.items():
            completed = [s for s in s_list if s.status in ("completed", "settled", "SETTLED")]
            n_shadow = len(completed)
            shadow_wins = sum(1 for s in completed if s.shadow_result in ("WIN", "win"))
            shadow_wr = (shadow_wins / n_shadow * 100.0) if n_shadow > 0 else 0.0

            if n_shadow < MIN_SAMPLE_FOR_GATE_VERDICT:
                verdict = "NEEDS_MORE_DATA"
                rec = f"Accumulating shadow samples ({n_shadow}/{MIN_SAMPLE_FOR_GATE_VERDICT})"
            elif shadow_wr < 48.0:
                verdict = "POSITIVE"  # Low shadow WR means gate correctly blocks lossy trades
                rec = f"Gate successfully filters low-quality trades ({shadow_wr:.1f}%)"
            elif shadow_wr > 60.0:
                verdict = "NEGATIVE"  # High shadow WR means gate incorrectly rejects winning trades
                rec = f"Gate improperly rejects winning trades ({shadow_wr:.1f}%)"
            else:
                verdict = "NEUTRAL"
                rec = f"Gate performance neutral ({shadow_wr:.1f}%)"

            gates_dict[gate_name] = {
                "shadow_samples": n_shadow,
                "shadow_win_rate": round(shadow_wr, 1),
                "executed_samples": 0,
                "executed_win_rate": 0.0,
                "verdict": verdict,
                "recommendation": rec,
            }

        return {"gates": gates_dict}

    @staticmethod
    def evaluate_gate_effectiveness(
        gate_name: str,
        accepted_trades: List[Dict[str, Any]],
        shadow_trades: List[ShadowTrade],
    ) -> Dict[str, Any]:
        accepted = [t for t in accepted_trades if str(t.get("status")).lower() in ("win", "loss")]
        rejected = [s for s in shadow_trades if s.rejection_gate == gate_name and s.status in ("completed", "SETTLED")]

        acc_count = len(accepted)
        rej_count = len(rejected)

        acc_wins = sum(1 for t in accepted if str(t.get("status")).lower() == "win")
        acc_wr = (acc_wins / acc_count * 100.0) if acc_count > 0 else 0.0

        rej_wins = sum(1 for s in rejected if s.shadow_result in ("WIN", "win"))
        rej_wr = (rej_wins / rej_count * 100.0) if rej_count > 0 else 0.0

        sample_size = acc_count + rej_count

        if sample_size < MIN_SAMPLE_FOR_GATE_VERDICT:
            status = "INSUFFICIENT_DATA"
        elif acc_wr > (rej_wr + 5.0):
            status = "POSITIVE"  # Gate successfully blocked lower-quality trades
        elif rej_wr > (acc_wr + 5.0):
            status = "NEGATIVE"  # Gate incorrectly rejected winning trades
        else:
            status = "NEUTRAL"

        return {
            "gate_name": gate_name,
            "accepted_opportunities": acc_count,
            "rejected_opportunities": rej_count,
            "accepted_win_rate": round(acc_wr, 1),
            "shadow_rejected_win_rate": round(rej_wr, 1),
            "sample_size": sample_size,
            "status": status,
        }
