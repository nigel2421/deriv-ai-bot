import asyncio
import json
import logging
from typing import Any, Dict, Optional

from config.settings import (
    EXECUTE_TRADES,
    LEARNING_ALWAYS,
    LEARNING_PATH,
    MAX_OPEN_TRADES,
    MAX_STAKE,
    MAX_STAKE_PCT,
    MIN_BALANCE,
    MIN_NET_RETURN,
    MIN_STAKE,
    SYMBOLS,
    TRADE_DURATION_TICKS,
)
from pathlib import Path
from src.api.deriv_client import DerivClient
from src.api.price_fetcher import PriceFetcher
from src.api.trade_executor import TradeExecutor
from src.api.trade_monitor import TradeMonitor
from src.strategy.xml_parser import XMLStrategyParser
from src.strategy.strategy_engine import StrategyEngine
from src.strategy.risk_manager import RiskManager
from src.strategy.trade_selector import TradeSelector
from src.strategy.signal_generator import SignalGenerator
from src.strategy.trend_analyzer import analyze_trend
from src.strategy.adaptive_learner import AdaptiveLearner
from src.strategy.anti_spiral import AntiSpiral
from src.strategy.regime_filter import should_skip_digits, should_skip_rise_fall
from src.strategy.digit_queue import queue_signal
from src.strategy.minute_engine import analyze_minute
from src.strategy.contract_types import is_digit_contract, is_rise_fall, normalize_contract_type
from src.strategy.ev_engine import ev_rank, compute_ev, DEFAULT_PAYOUT_RATE
from src.strategy.mor_tracker import MORTracker, normalize_score
from src.strategy.transition_matrix import TransitionMatrix
from src.strategy.calibration_tracker import CalibrationTracker
from src.strategy.correlation_filter import CorrelationFilter
from src.strategy.ai_auditor import AIAuditor
from src.ai.predictor import Predictor
from src.utils.telegram_bot import TelegramBot
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class TradingOrchestrator:
    """Main coordinator for multi-market scanning and the trade lifecycle."""

    def __init__(self, client: DerivClient, mode: str = "demo"):
        self.client = client
        self.mode = mode
        self.fetcher = PriceFetcher(client)
        self.executor = TradeExecutor(client)
        self.parser = XMLStrategyParser()
        self.strategy_engine = StrategyEngine(self.parser)
        global_cfg = self.parser.config.get("global", {})
        # Confidence gate comes from strategy.xml only (soft safety clamp 50–95%)
        raw_min = float(global_cfg.get("min_confidence", 0.80))
        self.min_confidence = max(0.50, min(0.95, raw_min))

        min_balance = MIN_BALANCE
        if mode == "real" and min_balance < 10:
            min_balance = 10.0

        self.risk_manager = RiskManager(
            max_daily_loss_pct=global_cfg.get("max_daily_loss_pct", 5.0),
            max_consecutive_losses=global_cfg.get("max_consecutive_losses", 6),
            trade_pause_minutes=global_cfg.get("trade_pause_minutes", 60),
            min_balance=min_balance,
            max_open_trades=MAX_OPEN_TRADES,
            max_stake_pct=MAX_STAKE_PCT,
            min_stake=MIN_STAKE,
            max_stake=MAX_STAKE,
        )
        self.selector = TradeSelector()
        self.signal_gen = SignalGenerator(prefer_parity=True)
        self.predictor = Predictor()
        self.learner = AdaptiveLearner(
            path=Path(LEARNING_PATH),
            always_on=LEARNING_ALWAYS,
            min_samples=2,
            cold_streak_skip=2,  # ban setup after 2 losses (was 4)
        )
        self.anti_spiral = AntiSpiral()
        # Flat stake by default — martingale is the main "loss pit" driver
        self.stake_mode = str(
            global_cfg.get("stake_mode")
            or __import__("os").getenv("STAKE_MODE", "flat")
        ).strip().lower()
        self.telegram = TelegramBot()
        self.telegram.set_status_provider(self.risk_status)
        self.telegram.set_stats_provider(self.stats_snapshot)
        # Telegram /resume also clears risk cooldown
        self.telegram.on_resume_hook = lambda: self.force_resume("telegram:/resume")
        self.telegram.on_pause_hook = lambda: self.force_pause("telegram:/pause", 60)

        # ---- New engines (Recs #3, #4, #5, #7, #8, #9, #10) ----
        self.transition_matrix = TransitionMatrix(
            path=Path("data/transition_matrix.json")
        )
        self.calibration = CalibrationTracker(
            path=Path("data/calibration_state.json")
        )
        self.mor_tracker = MORTracker(path=Path("data/mor_state.json"))
        self.correlation_filter = CorrelationFilter()
        self.ai_auditor = AIAuditor(
            history_path=Path("data/trade_history.jsonl"),
            report_path=Path("data/auditor_report.json"),
        )
        # Persistent trade history file (append-only JSONL) — Recs #5, #8, #10
        self._trade_history_path = Path("data/trade_history.jsonl")
        self._trade_history_path.parent.mkdir(parents=True, exist_ok=True)

        # Prefer env SYMBOLS; ensure strategy.xml markets are covered when listed
        xml_syms = self.parser.market_symbols()
        if SYMBOLS:
            # Keep order from env; drop unknown only if xml has a strict list
            self.active_symbols = list(SYMBOLS)
            for s in xml_syms:
                if s not in self.active_symbols:
                    self.active_symbols.append(s)
        else:
            self.active_symbols = xml_syms or list(SYMBOLS)
        self.max_open_trades = MAX_OPEN_TRADES

        self.execute_trades = bool(EXECUTE_TRADES)
        if mode == "real" and not EXECUTE_TRADES:
            self.execute_trades = False

        self.monitor = TradeMonitor(client, on_close=self._on_contract_closed)
        self.closed_trades: list = []
        # Structured trade log for dashboard (newest last, cap 50)
        self.trade_log: list = []
        # Durable open-trade view for dashboard (monitor can drop short-lived contracts)
        self.open_trade_meta: Dict[Any, Dict[str, Any]] = {}
        self.min_net_return = float(MIN_NET_RETURN)
        self.enable_minute = str(
            global_cfg.get("enable_minute_engine")
            or __import__("os").getenv("ENABLE_MINUTE_ENGINE", "true")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.minute_duration = int(
            global_cfg.get("minute_duration")
            or __import__("os").getenv("MINUTE_DURATION", "2")
        )
        self.minute_min_conf = float(
            global_cfg.get("minute_min_confidence")
            or __import__("os").getenv("MINUTE_MIN_CONFIDENCE", "0.78")
        )

        bal = client.get_balance()
        if bal is not None:
            self.risk_manager.set_session_balance(bal)

        logger.info(
            "Orchestrator ready mode=%s execute_trades=%s symbols=%s "
            "strategies=%s min_conf=%.2f min_balance=%s max_open=%s telegram=%s",
            mode,
            self.execute_trades,
            self.active_symbols,
            {s: r.stype for s, r in self.strategy_engine.runtimes.items()},
            self.min_confidence,
            min_balance,
            MAX_OPEN_TRADES,
            self.telegram.is_configured(),
        )

    def open_trade_count(self) -> int:
        mon = set(self.monitor.open_contracts.keys())
        exe = set(self.executor.open_trades.keys())
        local = set(self.open_trade_meta.keys())
        return len(mon | exe | local)

    def _reap_stale_open_trades(self, max_age_seconds: Optional[float] = None) -> int:
        """
        Drop local/executor open-trade entries that are older than duration + buffer.
        Prevents a missed settle callback from permanently blocking trading
        (especially with MAX_OPEN_TRADES=1).
        """
        now = datetime.now(timezone.utc)
        reaped = 0
        for cid, meta in list((self.open_trade_meta or {}).items()):
            age_limit = max_age_seconds
            if age_limit is None:
                dur = meta.get("duration")
                unit = str(meta.get("duration_unit") or "t").lower()
                try:
                    d = float(dur) if dur is not None else 5.0
                except (TypeError, ValueError):
                    d = 5.0
                if unit == "m":
                    age_limit = d * 60.0 + 120.0  # duration + 2 min buffer
                elif unit == "s":
                    age_limit = d + 120.0
                else:
                    # ticks: ~2s worst case each + generous buffer
                    age_limit = max(90.0, d * 3.0 + 60.0)
            opened = meta.get("opened_at")
            if not opened:
                continue
            try:
                if isinstance(opened, str):
                    ot = datetime.fromisoformat(opened.replace("Z", "+00:00"))
                else:
                    ot = opened
                if ot.tzinfo is None:
                    ot = ot.replace(tzinfo=timezone.utc)
            except Exception:
                continue
            age = (now - ot).total_seconds()
            if age < age_limit:
                continue
            logger.warning(
                "Reaping stale open trade contract_id=%s age=%.0fs limit=%.0fs %s %s",
                cid,
                age,
                age_limit,
                meta.get("symbol"),
                meta.get("contract_type"),
            )
            self.open_trade_meta.pop(cid, None)
            self.executor.mark_closed(cid)
            self.monitor.open_contracts.pop(cid, None)
            self._log_trade_event(
                {
                    "status": "stale_reaped",
                    "contract_id": cid,
                    "symbol": meta.get("symbol"),
                    "contract_type": meta.get("contract_type"),
                    "stake": meta.get("stake"),
                    "opened_at": meta.get("opened_at"),
                    "age_seconds": round(age, 1),
                }
            )
            reaped += 1
        # Executor-only orphans with no meta (age unknown) — drop if monitor empty
        for cid in list((self.executor.open_trades or {}).keys()):
            if cid in self.open_trade_meta or cid in self.monitor.open_contracts:
                continue
            # No meta and not monitored → ghost; drop after grace
            logger.warning("Reaping orphan executor open trade contract_id=%s", cid)
            self.executor.mark_closed(cid)
            reaped += 1
        return reaped

    async def _handle_auto_resume(self, event: Dict[str, Any]) -> None:
        """Cool-down finished: clear soft bans and notify operator."""
        prev = event.get("previous_streak")
        reason = event.get("reason") or "cooldown"
        logger.info(
            "AUTO-RESUME after cooldown (prev_streak=%s reason=%s count=%s)",
            prev,
            reason,
            event.get("count"),
        )
        try:
            self.anti_spiral.clear_cooldowns()
        except Exception as e:
            logger.debug("anti_spiral clear on auto-resume failed: %s", e)
        # Ensure Telegram master switch is on after *risk* cooldown (not manual /pause)
        # Manual pause sets telegram is_active=False AND risk pause — if user used
        # /pause we leave telegram alone. Risk-only pauses never flip telegram.
        bal = self.client.get_balance()
        msg = self.telegram.format_system(
            "▶️ AUTO-RESUME after cooldown",
            [
                f"Previous loss streak: <code>{prev}</code>",
                f"Reason: <code>{TelegramBot._esc(str(reason))}</code>",
                f"Pause # auto-resumes: <code>{event.get('count')}</code>",
                "Loss streak reset to 0 — trading continues.",
            ],
            balance=bal,
            currency=self.client.get_currency(),
        )
        await self.telegram.send_notification(msg)

    async def _live_balance(self, refresh: bool = False) -> Optional[float]:
        if refresh or self.client.get_balance() is None:
            bal = await self.client.refresh_balance()
        else:
            bal = self.client.get_balance()
        if bal is not None:
            self.risk_manager.update_balance(bal)
            if self.risk_manager.session_start_balance is None:
                self.risk_manager.set_session_balance(bal)
        return bal

    async def scan_markets(self) -> Optional[Dict[str, Any]]:
        """
        Scan all markets for Digits + Rise/Fall candidates.
        Apply adaptive learning, enforce min_confidence (>=80%), pick best market.
        """
        signals = []
        min_conf = self.min_confidence

        for symbol in self.active_symbols:
            ticks = self.fetcher.get_recent_data(symbol, 120)
            if not ticks:
                logger.debug("No ticks yet for %s", symbol)
                continue

            runtime = self.strategy_engine.get(symbol)
            if not runtime.is_tradeable():
                logger.info(
                    "Skip %s: strategy inactive (e.g. martingale max steps)", symbol
                )
                continue

            allowed_raw = runtime.allowed_types or []
            allowed = [
                t
                for t in (normalize_contract_type(x) for x in allowed_raw)
                if t
            ]
            digit_allowed = [t for t in allowed if is_digit_contract(t)]
            rf_allowed = [t for t in allowed if is_rise_fall(t)]
            # If allow-list empty, permit digits (incl. even/odd) + rise/fall
            if not allowed:
                digit_allowed = [
                    "DIGITOVER",
                    "DIGITUNDER",
                    "DIGITEVEN",
                    "DIGITODD",
                ]
                rf_allowed = ["CALL", "PUT"]

            candidates: list = []

            # ---- Early regime skip (choppy / whipsaw) ----
            skip_d, dig_reason, dig_reg = should_skip_digits(ticks)
            skip_rf, rf_reason, rf_reg = should_skip_rise_fall(ticks)
            if skip_d and skip_rf:
                logger.info(
                    "Skip %s both families: digits=%s rf=%s chop=%.2f",
                    symbol,
                    dig_reason,
                    rf_reason,
                    dig_reg.get("chop_score", 0),
                )
                continue

            # ---- Digits path (AI / heuristic last-digit + even/odd) ----
            if digit_allowed and not skip_d:
                pred = self.predictor.predict(ticks)
                pred = {**pred, "recent_ticks": ticks}
                # Parity streak confidence for EVEN/ODD (often clearer than single digit)
                ct_stats, parity_conf = self.signal_gen.parity_confidence(ticks)
                raw_conf = float(pred.get("confidence", 0.5))
                # Prefer parity path when even/odd allowed and parity is strong
                if (
                    ("DIGITEVEN" in digit_allowed or "DIGITODD" in digit_allowed)
                    and parity_conf >= 0.62
                ):
                    pred = {
                        **pred,
                        "preferred_type": (
                            "DIGITEVEN" if ct_stats.get("even") else "DIGITODD"
                        ),
                        "confidence": max(raw_conf, parity_conf),
                    }
                    raw_conf = float(pred["confidence"])
                # Community-style digit queue (high/low runs → over/under)
                q = queue_signal(ticks)
                if q and q.get("preferred_type") in (digit_allowed or []):
                    pred = {
                        **pred,
                        "preferred_type": q["preferred_type"],
                        "confidence": min(
                            0.95,
                            max(raw_conf, 0.72) + float(q.get("confidence_boost") or 0),
                        ),
                        "queue_reason": q.get("reason"),
                    }
                    if q.get("hint_barrier") is not None:
                        pred["barrier"] = q["hint_barrier"]
                    raw_conf = float(pred["confidence"])
                    logger.info(
                        "%s digit_queue %s boost → conf=%.2f",
                        symbol,
                        q.get("reason"),
                        raw_conf,
                    )
                signal_type, signal_barrier, conf = self.signal_gen.generate_signal(
                    pred,
                    raw_conf,
                    min_confidence=0.0,  # gate after learning adjust
                    allowed_types=digit_allowed,
                )
                if signal_type:
                    adj = self.learner.adjust_confidence(
                        symbol, signal_type, conf or raw_conf
                    )
                    if adj >= min_conf:
                        # Rec #6: check pattern decay before adding candidate
                        block_decay, decay_reason = self.learner.should_block_for_decay(
                            symbol, signal_type, current_strength=conf or raw_conf
                        )
                        if block_decay:
                            logger.info(
                                "PatternDecay block %s %s: %s",
                                symbol, signal_type, decay_reason,
                            )
                            continue
                        # Rec #6: record scan-time strength for decay tracking
                        self.learner.record_pattern_strength(
                            symbol, signal_type, conf or raw_conf
                        )
                        try:
                            pred_digit = int(pred.get("digit")) if pred.get("digit") is not None else None
                        except (TypeError, ValueError):
                            pred_digit = None
                        intent = self.strategy_engine.apply_signal(
                            symbol=symbol,
                            signal_type=signal_type,
                            signal_barrier=signal_barrier,
                            confidence=adj,
                            predicted_digit=pred_digit,
                            ticks=ticks,
                        )
                        if intent:
                            intent["family"] = "digits"
                            intent["raw_confidence"] = conf or raw_conf
                            intent["regime"] = dig_reg
                            intent["learn_bonus"] = self.learner.selection_bonus(
                                symbol, intent["contract_type"]
                            )
                            intent["trend_strength"] = 0.0
                            # Rec #9: payout_rate placeholder (updated after proposal)
                            intent["payout_rate"] = DEFAULT_PAYOUT_RATE
                            # Rec #4: MOR velocity bonus
                            intent["velocity_bonus"] = self.mor_tracker.get_velocity_bonus(symbol)
                            # Rec #1: confidence level metadata
                            intent["confidence_level"] = self.learner.confidence_level(
                                symbol, intent["contract_type"]
                            )
                            intent["historical_support"] = self.learner.historical_support(
                                symbol, intent["contract_type"]
                            )
                            candidates.append(intent)
            elif skip_d:
                logger.debug("Skip digits %s: %s", symbol, dig_reason)

            # ---- Rise/Fall path (trend + chart tools; skip chop) ----
            if rf_allowed and not skip_rf:
                trend = analyze_trend(ticks)
                rf_type = trend.get("contract_type")
                rf_conf = float(trend.get("confidence") or 0.0)
                # Penalize mild chop even if tools pass
                chop = float(rf_reg.get("chop_score") or 0)
                if chop > 0.45:
                    rf_conf *= 1.0 - (chop - 0.45) * 0.8
                if rf_type and rf_type in rf_allowed:
                    adj = self.learner.adjust_confidence(symbol, rf_type, rf_conf)
                    if adj >= min_conf:
                        # Rec #6: check pattern decay for rise/fall
                        block_decay, decay_reason = self.learner.should_block_for_decay(
                            symbol, rf_type, current_strength=rf_conf
                        )
                        if block_decay:
                            logger.info(
                                "PatternDecay block %s %s: %s",
                                symbol, rf_type, decay_reason,
                            )
                        else:
                            self.learner.record_pattern_strength(symbol, rf_type, rf_conf)
                            intent = self.strategy_engine.apply_signal(
                                symbol=symbol,
                                signal_type=rf_type,
                                signal_barrier=None,
                                confidence=adj,
                            )
                            if intent:
                                intent["family"] = "rise_fall"
                                intent["raw_confidence"] = rf_conf
                                intent["trend"] = trend
                                intent["regime"] = rf_reg
                                intent["trend_strength"] = float(
                                    trend.get("strength") or 0.0
                                )
                                intent["learn_bonus"] = self.learner.selection_bonus(
                                    symbol, intent["contract_type"]
                                )
                                # Rec #3: persistence adjustment from TransitionMatrix
                                intent["persistence_adjustment"] = (
                                    self.transition_matrix.persistence_score_adjustment(
                                        symbol, rf_type
                                    )
                                )
                                intent["persistence_conf"] = (
                                    self.transition_matrix.persistence_confidence(
                                        symbol, rf_type
                                    )
                                )
                                # Rec #9: payout_rate placeholder
                                intent["payout_rate"] = DEFAULT_PAYOUT_RATE
                                # Rec #4: MOR velocity bonus
                                intent["velocity_bonus"] = self.mor_tracker.get_velocity_bonus(symbol)
                                # Rec #1: confidence level metadata
                                intent["confidence_level"] = self.learner.confidence_level(
                                    symbol, intent["contract_type"]
                                )
                                intent["historical_support"] = self.learner.historical_support(
                                    symbol, intent["contract_type"]
                                )
                                candidates.append(intent)
                    else:
                        logger.debug(
                            "%s trend %s conf=%.2f adj=%.2f < min=%.2f",
                            symbol,
                            rf_type,
                            rf_conf,
                            adj,
                            min_conf,
                        )
            elif skip_rf:
                logger.info(
                    "Skip rise/fall %s: %s (chop=%.2f eff=%.2f)",
                    symbol,
                    rf_reason,
                    rf_reg.get("chop_score", 0),
                    rf_reg.get("efficiency", 0),
                )

            # ---- Minute Rise/Fall (candles + EMA/RSI) ----
            if self.enable_minute and rf_allowed and not skip_rf:
                need = max(self.minute_min_conf, min_conf * 0.95)
                msig = analyze_minute(
                    ticks,
                    period_sec=60,
                    duration_minutes=self.minute_duration,
                    min_confidence=need,
                )
                if msig and msig["contract_type"] in rf_allowed:
                    adj = self.learner.adjust_confidence(
                        symbol, msig["contract_type"], float(msig["confidence"])
                    )
                    if adj >= need:
                        intent = self.strategy_engine.apply_signal(
                            symbol=symbol,
                            signal_type=msig["contract_type"],
                            signal_barrier=None,
                            confidence=adj,
                        )
                        if intent:
                            intent["family"] = "minute_rise_fall"
                            intent["horizon"] = "minute"
                            intent["duration"] = msig["duration"]
                            intent["duration_unit"] = "m"
                            intent["raw_confidence"] = msig["confidence"]
                            intent["minute_details"] = msig.get("details")
                            intent["learn_bonus"] = self.learner.selection_bonus(
                                symbol, intent["contract_type"]
                            )
                            intent["trend_strength"] = float(
                                (msig.get("details") or {}).get("call_pts")
                                or (msig.get("details") or {}).get("put_pts")
                                or 0
                            ) / 5.0
                            candidates.append(intent)
                            logger.info(
                                "Minute candidate %s %s conf=%.2f dur=%sm notes=%s",
                                symbol,
                                msig["contract_type"],
                                adj,
                                msig["duration"],
                                (msig.get("details") or {}).get("notes"),
                            )

            for intent in candidates:
                # Rec #4: Update MOR tracker for this symbol (scored candidate)
                raw_score = (
                    float(intent.get("confidence") or 0.0)
                    + float(intent.get("learn_bonus") or 0.0)
                    + 0.05 * float(intent.get("trend_strength") or 0.0)
                )
                mor_score = self.mor_tracker.update_score(intent["symbol"], raw_score)
                intent["mor_score"] = mor_score

                # Flat stake mode: never martingale-double (stops loss pits)
                if self.stake_mode == "flat":
                    intent["stake"] = float(intent.get("base_stake") or intent["stake"])
                ok_as, why = self.anti_spiral.allow(
                    symbol,
                    str(intent["contract_type"]),
                    float(intent.get("confidence") or 0),
                )
                if not ok_as:
                    logger.info(
                        "AntiSpiral block %s %s: %s",
                        symbol,
                        intent["contract_type"],
                        why,
                    )
                    continue
                signals.append(intent)
                logger.info(
                    "Candidate %s: type=%s stake=%.2f barrier=%s conf=%.2f "
                    "(raw=%.2f) family=%s strat=%s mor=%.1f level=%s",
                    symbol,
                    intent["contract_type"],
                    intent["stake"],
                    intent.get("barrier"),
                        intent["confidence"],
                    intent.get("raw_confidence"),
                    intent.get("family"),
                    intent["strategy"],
                    intent.get("mor_score", 0),
                    intent.get("confidence_level", "?"),
                )

        if not signals:
            logger.info(
                "No signals ≥ %.0f%% confidence across %s markets "
                "(anti_spiral=%s)",
                min_conf * 100,
                len(self.active_symbols),
                self.anti_spiral.snapshot(),
            )
            return None

        # ---- Rec #9: EV ranking (annotate + sort by EV, filter negative EV) ----
        signals = ev_rank(signals, allow_negative=False)
        if not signals:
            logger.info("All candidates filtered by negative EV")
            return None

        # ---- Rec #7: Correlation filter (within-group, keep highest EV) ----
        signals_before_corr = list(signals)
        signals = self.correlation_filter.filter_candidates(signals)
        if not signals:
            logger.info("All candidates filtered by correlation filter")
            return None
        # Store last correlation snapshot for dashboard
        self._last_corr_snapshot = self.correlation_filter.snapshot(
            signals_before_corr, signals
        )

        best = self.selector.select_best_trade(signals)
        if best:
            self.anti_spiral.note_selected(
                str(best["symbol"]), str(best["contract_type"])
            )
        return best

    async def execute_trade_cycle(self) -> Optional[Dict[str, Any]]:
        """Full cycle: risk → scan/strategy → stake clamp → proposal → buy → monitor."""
        # Reap ghost opens so max_open_trades never permanently blocks the bot
        self._reap_stale_open_trades()

        # Auto-resume after 30m cooldown (risk manager clears streak on expiry)
        auto = self.risk_manager.consume_auto_resume()
        if auto:
            await self._handle_auto_resume(auto)

        # Telegram /pause master switch (manual only — not the risk cooldown)
        if not self.telegram.trading_enabled:
            logger.info("Trading paused via Telegram (/pause). Skipping cycle.")
            return None

        balance = await self._live_balance(refresh=False)
        open_count = self.open_trade_count()

        decision = self.risk_manager.can_trade(balance, open_trades=open_count)
        # consume again in case can_trade just expired the pause mid-check
        auto2 = self.risk_manager.consume_auto_resume()
        if auto2:
            await self._handle_auto_resume(auto2)
            decision = self.risk_manager.can_trade(balance, open_trades=open_count)

        if not decision:
            logger.warning(
                "Risk block: %s | balance=%s open=%s/%s consecutive=%s paused_until=%s",
                decision.reason,
                balance,
                open_count,
                self.max_open_trades,
                self.risk_manager.consecutive_losses,
                self.risk_manager.paused_until,
            )
            return None

        best = await self.scan_markets()
        if not best:
            logger.info(
                "No qualifying signals | balance=%s %s open=%s strategies=%s",
                balance,
                self.client.get_currency(),
                open_count,
                self.strategy_engine.snapshots(),
            )
            return None

        # Prefer strategy-computed stake; fall back to base
        raw_stake = float(best.get("stake", best.get("base_stake", MIN_STAKE)))
        assert balance is not None
        stake = self.risk_manager.clamp_stake(raw_stake, balance)
        if stake <= 0:
            logger.warning(
                "Stake %.2f cannot be placed (balance=%.2f risk caps)",
                raw_stake,
                balance,
            )
            return None

        if stake < raw_stake:
            logger.info(
                "Stake clamped %.2f → %.2f (balance=%.2f)",
                raw_stake,
                stake,
                balance,
            )

        stake_decision = self.risk_manager.can_trade(
            balance, open_trades=open_count, proposed_stake=stake
        )
        if not stake_decision:
            logger.warning("Risk block on stake: %s", stake_decision.reason)
            return None

        duration = int(best.get("duration") or TRADE_DURATION_TICKS)
        duration_unit = str(best.get("duration_unit") or "t")
        # Minute family always uses minutes
        if best.get("family") == "minute_rise_fall" or best.get("horizon") == "minute":
            duration_unit = "m"
            duration = int(best.get("duration") or self.minute_duration)
        # Always define horizon before any logging / event dict uses it
        horizon = str(
            best.get("horizon")
            or ("minute" if duration_unit == "m" else "tick")
        )

        logger.info(
            "Trade candidate: %s %s stake=%s barrier=%s conf=%.2f "
            "dur=%s%s family=%s horizon=%s min_net=%.0f%% execute=%s balance=%.2f open=%s",
            best["symbol"],
            best["contract_type"],
            stake,
            best.get("barrier"),
            best.get("confidence", 0),
            duration,
            duration_unit,
            best.get("family"),
            horizon,
            self.min_net_return * 100,
            self.execute_trades,
            balance,
            open_count,
        )

        result = await self.executor.propose_and_buy(
            symbol=best["symbol"],
            contract_type=best["contract_type"],
            stake=stake,
            barrier=best.get("barrier"),
            currency=self.client.get_currency(),
            duration=duration,
            duration_unit=duration_unit,
            execute=self.execute_trades,
            min_net_return=self.min_net_return if self.execute_trades else None,
        )

        if not result:
            err = self.executor.last_error or "unknown"
            logger.error("Trade path failed: %s", err)
            self._log_trade_event(
                {
                    "status": "failed",
                    "symbol": best["symbol"],
                    "contract_type": best["contract_type"],
                    "stake": stake,
                    "barrier": best.get("barrier"),
                    "confidence": best.get("confidence"),
                    "family": best.get("family"),
                    "horizon": horizon,
                    "duration": duration,
                    "duration_unit": duration_unit,
                    "error": err,
                }
            )
            bal_now = self.client.get_balance()
            await self.telegram.send_notification(
                self.telegram.format_trade_error(
                    title="Trade failed",
                    error=err,
                    balance=bal_now,
                    currency=self.client.get_currency(),
                    symbol=best["symbol"],
                    contract_type=best["contract_type"],
                    stake=stake,
                )
            )
            return None

        if result.get("skipped_low_payout"):
            err = result.get("error") or self.executor.last_error or "payout too low"
            net = result.get("net_return")
            self._log_trade_event(
                {
                    "status": "skipped_low_payout",
                    "symbol": best["symbol"],
                    "contract_type": best["contract_type"],
                    "stake": stake,
                    "barrier": best.get("barrier"),
                    "confidence": best.get("confidence"),
                    "family": best.get("family"),
                    "horizon": horizon,
                    "duration": duration,
                    "duration_unit": duration_unit,
                    "ask_price": result.get("ask_price"),
                    "payout": result.get("payout"),
                    "net_return": net,
                    "error": err,
                }
            )
            # Soft notice only — not a failure; cool down this setup lightly
            self.anti_spiral.note_selected(
                str(best["symbol"]), str(best["contract_type"])
            )
            logger.info(
                "Skipped low-payout quote: %s %s barrier=%s net=%s",
                best["symbol"],
                best["contract_type"],
                best.get("barrier"),
                f"{float(net):+.0%}" if net is not None else "?",
            )
            return result

        if result.get("buy_failed"):
            err = result.get("error") or self.executor.last_error or "buy failed"
            logger.error("Buy failed after proposal: %s", err)
            self._log_trade_event(
                {
                    "status": "buy_failed",
                    "symbol": best["symbol"],
                    "contract_type": best["contract_type"],
                    "stake": stake,
                    "error": err,
                    "family": best.get("family"),
                    "horizon": horizon,
                }
            )
            await self._live_balance(refresh=True)
            await self.telegram.send_notification(
                self.telegram.format_trade_error(
                    title="Buy failed",
                    error=err,
                    balance=self.client.get_balance(),
                    currency=self.client.get_currency(),
                    symbol=best["symbol"],
                    contract_type=best["contract_type"],
                    stake=stake,
                )
            )
            return result

        proposal = result.get("proposal") or {}
        buy = result.get("buy") or {}

        if buy.get("balance_after") is not None:
            try:
                self.client.set_balance(float(buy["balance_after"]))
                self.risk_manager.update_balance(float(buy["balance_after"]))
            except (TypeError, ValueError):
                pass

        bal_after = buy.get("balance_after")
        if bal_after is None:
            bal_after = self.client.get_balance()
        await self.telegram.send_notification(
            self.telegram.format_trade_opened(
                symbol=best["symbol"],
                contract_type=best["contract_type"],
                stake=stake,
                balance=bal_after,
                currency=self.client.get_currency(),
                confidence=best.get("confidence"),
                barrier=best.get("barrier"),
                duration=duration,
                duration_unit=duration_unit,
                family=str(best.get("family") or ""),
                contract_id=result.get("contract_id"),
                ask_price=proposal.get("ask_price"),
                executed=bool(result.get("executed")),
            )
        )

        contract_id = result.get("contract_id")
        if result.get("executed") and contract_id is not None:
            opened_at = datetime.now(timezone.utc).isoformat()
            meta = {
                **best,
                "symbol": best["symbol"],
                "contract_type": best["contract_type"],
                "stake": stake,
                "barrier": best.get("barrier"),
                "confidence": best.get("confidence"),
                "family": best.get("family"),
                "horizon": horizon,
                "duration": duration,
                "duration_unit": duration_unit,
                "proposal_id": proposal.get("id"),
                "opened_at": opened_at,
                "ask_price": proposal.get("ask_price") or result.get("ask_price"),
                "payout": proposal.get("payout") or result.get("payout"),
                "net_return": result.get("net_return"),
                "status": "open",
                "contract_id": contract_id,
            }
            # Track locally first so dashboard always sees the open trade
            self.open_trade_meta[contract_id] = dict(meta)
            await self.monitor.watch(contract_id, meta=meta)
            self._log_trade_event(
                {
                    "status": "open",
                    "contract_id": contract_id,
                    "symbol": best["symbol"],
                    "contract_type": best["contract_type"],
                    "stake": stake,
                    "barrier": best.get("barrier"),
                    "confidence": best.get("confidence"),
                    "family": best.get("family"),
                    "horizon": horizon,
                    "duration": duration,
                    "duration_unit": duration_unit,
                    "ask_price": meta.get("ask_price"),
                    "payout": meta.get("payout"),
                    "net_return": meta.get("net_return"),
                    "opened_at": opened_at,
                }
            )
            logger.info(
                "Trade opened contract_id=%s family=%s horizon=%s open_now=%s",
                contract_id,
                best.get("family"),
                horizon,
                self.open_trade_count(),
            )
        elif not self.execute_trades:
            logger.info("Proposal-only cycle complete (EXECUTE_TRADES=false)")

        return result

    def _log_trade_event(self, event: Dict[str, Any]) -> None:
        event = {
            **event,
            "ts": event.get("ts") or datetime.now(timezone.utc).isoformat(),
        }
        self.trade_log.append(event)
        if len(self.trade_log) > 50:
            self.trade_log = self.trade_log[-50:]

    def _on_contract_closed(
        self, contract: Dict[str, Any], meta: Dict[str, Any]
    ) -> None:
        """Called from WS thread via monitor when a contract settles."""
        try:
            profit = float(contract.get("profit") or 0)
        except (TypeError, ValueError):
            profit = 0.0

        is_win = profit > 0
        contract_id = contract.get("contract_id")
        symbol = meta.get("symbol")
        contract_type = meta.get("contract_type") or ""

        self.executor.mark_closed(contract_id)
        if contract_id is not None:
            self.open_trade_meta.pop(contract_id, None)
        self.risk_manager.record_trade_result(profit)

        # Advance Martingale / Zuno for this symbol
        if symbol:
            self.strategy_engine.on_trade_result(symbol, is_win=is_win, profit=profit)
            # Online learning: re-weight markets / contract families
            if contract_type:
                try:
                    conf_in = float(meta.get("confidence") or 0)
                except (TypeError, ValueError):
                    conf_in = None
                self.learner.record(
                    symbol,
                    str(contract_type),
                    is_win,
                    profit,
                    confidence=conf_in,
                    family=meta.get("family"),
                )
                self.anti_spiral.record(symbol, str(contract_type), is_win)

        self.closed_trades.append(
            {"contract": contract, "meta": meta, "profit": profit}
        )

        for key in ("balance_after", "account_balance"):
            if contract.get(key) is not None:
                try:
                    bal = float(contract[key])
                    self.client.set_balance(bal)
                    self.risk_manager.update_balance(bal)
                except (TypeError, ValueError):
                    pass
                break

        status = "WIN" if is_win else ("PUSH" if profit == 0 else "LOSS")
        strat_snap = (
            self.strategy_engine.get(symbol).snapshot() if symbol else {}
        )
        snap = self.risk_manager.snapshot()
        # Update trade log entry / append close
        self._log_trade_event(
            {
                "status": status.lower(),
                "contract_id": contract_id,
                "symbol": symbol,
                "contract_type": meta.get("contract_type"),
                "stake": meta.get("stake"),
                "barrier": meta.get("barrier"),
                "confidence": meta.get("confidence"),
                "family": meta.get("family"),
                "horizon": "min"
                if meta.get("duration_unit") == "m"
                else "tick",
                "duration": meta.get("duration"),
                "duration_unit": meta.get("duration_unit"),
                "profit": profit,
                "closed_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        # ---- Rec #8: Calibration tracking ----
        conf_in = None
        try:
            conf_in = float(meta.get("confidence") or 0)
        except (TypeError, ValueError):
            pass
        if conf_in and conf_in > 0:
            self.calibration.record(conf_in, is_win)

        # ---- Rec #3: Transition matrix (rise/fall only) ----
        if symbol and contract_type and is_rise_fall(contract_type):
            self.transition_matrix.record_outcome(symbol, contract_type, is_win)

        # ---- Rec #5: MOR outcome tracking ----
        mor_at_trade = float(meta.get("mor_score") or 0)
        if symbol and contract_type:
            self.mor_tracker.record_outcome(
                symbol, mor_at_trade, is_win, contract_type
            )

        # ---- Persistent trade history (JSONL) ---- Recs #5, #8, #10 ----
        self._append_trade_history({
            "ts": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "contract_type": str(contract_type),
            "barrier": meta.get("barrier"),
            "stake": meta.get("stake"),
            "confidence": conf_in,
            "payout_rate": float(meta.get("payout_rate") or DEFAULT_PAYOUT_RATE),
            "ev": float(meta.get("ev") or 0),
            "mor_score": mor_at_trade,
            "family": meta.get("family"),
            "is_win": is_win,
            "profit": profit,
            "trend_strength": float(meta.get("trend_strength") or 0),
            "learn_bonus": float(meta.get("learn_bonus") or 0),
            "parity_conf": float(meta.get("raw_confidence") or 0),
            "persistence_conf": float(meta.get("persistence_conf") or 0),
            "chop_score": float(
                (meta.get("regime") or {}).get("chop_score") or 0
            ),
            "status": status.lower(),
        })

        # ---- Rec #10: AI Auditor ----
        self.ai_auditor.check_and_run(self.learner.total_recorded)
        bal_now = self.client.get_balance()
        msg = self.telegram.format_trade_closed(
            status=status,
            symbol=str(symbol or "?"),
            contract_type=str(meta.get("contract_type") or "?"),
            profit=profit,
            balance=bal_now,
            currency=self.client.get_currency(),
            stake=meta.get("stake"),
            barrier=meta.get("barrier"),
            contract_id=contract_id,
            daily_pnl=snap.get("daily_pnl"),
            consecutive_losses=snap.get("consecutive_losses"),
            family=str(meta.get("family") or ""),
            duration=meta.get("duration"),
            duration_unit=str(meta.get("duration_unit") or "t"),
        )
        logger.info(
            "🏁 %s %s %s P&L=%s bal=%s daily=%s",
            status,
            symbol,
            meta.get("contract_type"),
            profit,
            bal_now,
            snap.get("daily_pnl"),
        )
        self._notify_async(msg)

    def _notify_async(self, message: str) -> None:
        """Schedule telegram send from WS/thread callbacks onto the asyncio loop."""
        loop = getattr(self.client, "_loop", None)
        if loop and loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(
                    self.telegram.send_notification(message),
                    loop,
                )
                return
            except Exception as e:
                logger.debug("notify schedule failed: %s", e)
        # Best-effort log if no loop
        logger.info("TELEGRAM(deferred): %s", message)

    def _append_trade_history(self, record: Dict[str, Any]) -> None:
        """Append a settled trade record to the persistent JSONL history file."""
        try:
            with self._trade_history_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.debug("Trade history append failed: %s", e)

    def stats_snapshot(self) -> Dict[str, Any]:
        snap = self.risk_manager.snapshot()
        return {
            **snap,
            "closed_count": len(self.closed_trades),
            "open_trades": self.open_trade_count(),
            "balance": self.client.get_balance(),
            "currency": self.client.get_currency(),
            "learning": self.learner.snapshot(),
            "min_confidence": self.min_confidence,
        }

    def risk_status(self) -> Dict[str, Any]:
        return {
            **self.risk_manager.snapshot(),
            "balance": self.client.get_balance(),
            "currency": self.client.get_currency(),
            "open_trades": self.open_trade_count(),
            "max_open_trades": self.max_open_trades,
            "execute_trades": self.execute_trades,
            "mode": self.mode,
            "telegram_trading": self.telegram.trading_enabled,
            "strategies": self.strategy_engine.snapshots(),
            "symbols": list(self.active_symbols),
            "closed_trades": len(self.closed_trades),
            "min_confidence": self.min_confidence,
            "learning": self.learner.snapshot(),
            "anti_spiral": self.anti_spiral.snapshot(),
            "stake_mode": self.stake_mode,
            "enable_minute": self.enable_minute,
            "minute_duration": self.minute_duration,
            "min_net_return": self.min_net_return,
            "recent_trades": list(reversed(self.trade_log[-20:])),
            "open_trade_details": self._open_trade_details(),
            # New engine snapshots
            "transition_matrix": self.transition_matrix.snapshot(),
            "calibration": self.calibration.snapshot(),
            "mor": self.mor_tracker.snapshot(),
            "correlation": getattr(self, "_last_corr_snapshot", {}),
            "ai_auditor": self.ai_auditor.latest_report(),
        }

    def _open_trade_details(self) -> list:
        """Merge monitor + local meta + executor so the dashboard card stays filled."""
        by_id: Dict[Any, Dict[str, Any]] = {}

        def _merge(cid: Any, row: Dict[str, Any]) -> None:
            if cid is None:
                return
            prev = by_id.get(cid) or {}
            merged = {**prev, **{k: v for k, v in row.items() if v is not None}}
            merged["contract_id"] = cid
            merged["status"] = "open"
            by_id[cid] = merged

        for cid, meta in (self.open_trade_meta or {}).items():
            _merge(
                cid,
                {
                    "symbol": meta.get("symbol"),
                    "contract_type": meta.get("contract_type"),
                    "stake": meta.get("stake"),
                    "barrier": meta.get("barrier"),
                    "confidence": meta.get("confidence"),
                    "family": meta.get("family"),
                    "horizon": meta.get("horizon"),
                    "duration": meta.get("duration"),
                    "duration_unit": meta.get("duration_unit"),
                    "opened_at": meta.get("opened_at"),
                    "ask_price": meta.get("ask_price"),
                    "payout": meta.get("payout"),
                },
            )
        for cid, meta in (self.monitor.open_contracts or {}).items():
            _merge(
                cid,
                {
                    "symbol": meta.get("symbol"),
                    "contract_type": meta.get("contract_type"),
                    "stake": meta.get("stake"),
                    "barrier": meta.get("barrier"),
                    "confidence": meta.get("confidence"),
                    "family": meta.get("family"),
                    "horizon": meta.get("horizon"),
                    "duration": meta.get("duration"),
                    "duration_unit": meta.get("duration_unit"),
                    "opened_at": meta.get("opened_at"),
                },
            )
        for cid, buy in (self.executor.open_trades or {}).items():
            local = (self.open_trade_meta or {}).get(cid) or {}
            _merge(
                cid,
                {
                    "symbol": local.get("symbol") or buy.get("symbol") or buy.get("shortcode"),
                    "contract_type": local.get("contract_type"),
                    "stake": local.get("stake") or buy.get("buy_price"),
                    "barrier": local.get("barrier"),
                    "confidence": local.get("confidence"),
                    "family": local.get("family"),
                    "horizon": local.get("horizon"),
                    "duration": local.get("duration"),
                    "duration_unit": local.get("duration_unit"),
                    "opened_at": local.get("opened_at"),
                },
            )
        # Fallback: recent trade_log opens not yet closed (race with settle stream)
        closed_ids = {
            e.get("contract_id")
            for e in self.trade_log
            if str(e.get("status") or "").lower() in {"win", "loss", "push"}
        }
        for e in reversed(self.trade_log):
            if str(e.get("status") or "").lower() != "open":
                continue
            cid = e.get("contract_id")
            if cid is None or cid in closed_ids or cid in by_id:
                continue
            _merge(cid, dict(e))

        return list(by_id.values())

    def force_resume(self, source: str = "control") -> Dict[str, Any]:
        """
        Clear risk cooldown + Telegram pause so trading continues immediately.
        Does not reset daily PnL or learning stats.
        Does NOT reset martingale ladders (that re-opens loss pits).
        """
        self.risk_manager.resume(reset_streak=True)
        self.telegram.resume_trading(source)
        self.anti_spiral.clear_cooldowns()
        # Keep martingale inactive if it hit max steps — force flat until natural win path
        for sym, rt in self.strategy_engine.runtimes.items():
            if rt.martingale and not rt.martingale.active:
                # Reactivate at BASE stake only (no ladder recovery)
                rt.martingale.reset()
                logger.info(
                    "Martingale re-armed at base only for %s via %s", sym, source
                )
        bal = self.client.get_balance()
        msg = self.telegram.format_system(
            "▶️ Trading RESUMED",
            [
                f"Source: <code>{source}</code>",
                f"Stake mode: <code>{self.stake_mode}</code>",
                "Cooldownoldown + anti-spiral bans cleared.",
            ],
            balance=bal,
            currency=self.client.get_currency(),
        )
        logger.info("Force RESUME via %s bal=%s", source, bal)
        self._notify_async(msg)
        return self.risk_status()

    def force_pause(self, source: str = "control", minutes: int = 60) -> Dict[str, Any]:
        """Stop new trades (Telegram switch + risk pause)."""
        self.telegram.pause_trading(source)
        self.risk_manager.pause(minutes=minutes, reason=source)
        bal = self.client.get_balance()
        msg = self.telegram.format_system(
            "⏸ Trading PAUSED",
            [
                f"Source: <code>{source}</code>",
                f"Risk pause: <code>{minutes}m</code>",
            ],
            balance=bal,
            currency=self.client.get_currency(),
        )
        logger.info("Force PAUSE via %s bal=%s", source, bal)
        self._notify_async(msg)
        return self.risk_status()
