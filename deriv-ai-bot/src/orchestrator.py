import asyncio
import logging
from typing import Any, Dict, Optional

from config.settings import (
    BASE_STAKE,
    DEEPSEEK_ANALYZE_EVERY,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_ENABLED,
    DEEPSEEK_MODEL,
    DEEPSEEK_TIMEOUT_SEC,
    EXECUTE_TRADES,
    LEARNING_ALWAYS,
    LEARNING_PATH,
    MAX_OPEN_TRADES,
    MAX_STAKE,
    MAX_STAKE_PCT,
    MIN_BALANCE,
    MIN_NET_RETURN,
    MIN_STAKE,
    SESSION_STOP_LOSS_PCT,
    SESSION_STOP_LOSS_PCT_MAX,
    SESSION_STOP_LOSS_PCT_MIN,
    SESSION_STOP_ON_TARGET,
    SESSION_TARGET_RR,
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
from src.strategy.pro_trend import analyze_pro_trend
from src.strategy.adaptive_learner import AdaptiveLearner
from src.strategy.anti_spiral import AntiSpiral
from src.strategy.market_offer_gate import (
    MarketOfferGate,
    REASON_DURATION,
    REASON_MARKET_CLOSED,
    REASON_UNAVAILABLE,
    classify_offer_error,
    duration_fallbacks,
)
from src.strategy.regime_filter import should_skip_digits, should_skip_rise_fall
from src.strategy.digit_queue import queue_signal
from src.strategy.minute_engine import analyze_minute
from src.strategy.contract_types import is_digit_contract, is_rise_fall, normalize_contract_type
from src.ai.predictor import Predictor
from src.ai.deepseek_advisor import DeepSeekAdvisor
from src.analytics.trade_filter import evaluate_setup
from src.analytics.edge_scanner import scan_markets
from src.analytics.digit_analysis import digit_snapshot
from src.analytics.probability_engine import probability_table
from src.analytics.session_analytics import SessionAnalytics
from src.analytics.strategy_builder import StrategyBuilder
from src.analytics.adaptive_stake import adaptive_risk_pct, stake_from_risk
from src.analytics.martingale_safety import survival_probability
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

        # Session stop-loss band 5–10%; align daily loss with session stop by default
        stop_pct = float(
            global_cfg.get("session_stop_loss_pct")
            or global_cfg.get("max_daily_loss_pct")
            or SESSION_STOP_LOSS_PCT
        )
        # Smart stop: default cool-down 10 minutes (was 30/60)
        pause_mins = int(global_cfg.get("trade_pause_minutes") or 10)
        self.risk_manager = RiskManager(
            max_daily_loss_pct=stop_pct,
            max_consecutive_losses=global_cfg.get("max_consecutive_losses", 6),
            trade_pause_minutes=pause_mins,
            min_balance=min_balance,
            max_open_trades=MAX_OPEN_TRADES,
            max_stake_pct=MAX_STAKE_PCT,
            min_stake=MIN_STAKE,
            max_stake=MAX_STAKE,
            session_stop_loss_pct=stop_pct,
            session_stop_loss_pct_min=SESSION_STOP_LOSS_PCT_MIN,
            session_stop_loss_pct_max=SESSION_STOP_LOSS_PCT_MAX,
            session_target_rr=float(
                global_cfg.get("session_target_rr") or SESSION_TARGET_RR
            ),
            session_stop_on_target=SESSION_STOP_ON_TARGET,
            base_stake=BASE_STAKE if BASE_STAKE > 0 else None,
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
        self.deepseek = DeepSeekAdvisor(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            model=DEEPSEEK_MODEL,
            enabled=DEEPSEEK_ENABLED,
            timeout_sec=DEEPSEEK_TIMEOUT_SEC,
            analyze_every=DEEPSEEK_ANALYZE_EVERY,
        )
        self.anti_spiral = AntiSpiral()
        # Skip duration/market-closed offers so we scan open markets instead
        self.offer_gate = MarketOfferGate()
        self._last_scan_signals: list = []
        self.session_analytics = SessionAnalytics(path=Path("data/session_analytics.json"))
        self.strategy_builder = StrategyBuilder(directory=Path("data/strategies"))
        # Seed example marketplace strategy if empty
        try:
            if not self.strategy_builder.list_strategies():
                self.strategy_builder.create_example_cold_digit()
        except Exception:
            pass
        # Analytics gate: Pattern≥75, LiveEdge≥80, Quality≥80 (softened when n small)
        self.analytics_gate = str(
            __import__("os").getenv("ANALYTICS_GATE", "true")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.last_filter: Optional[Dict[str, Any]] = None
        self.last_scan: Optional[Dict[str, Any]] = None
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
        # Env SYMBOLS is the scan universe (multi-market exploration).
        # strategy.xml still supplies per-market templates; on-demand runtime for extras.
        xml_syms = self.parser.market_symbols()
        if SYMBOLS:
            self.active_symbols = list(SYMBOLS)
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
        # Same floor as global min_confidence (unified across digits / RF / minute)
        self.minute_min_conf = float(
            global_cfg.get("minute_min_confidence")
            or __import__("os").getenv("MINUTE_MIN_CONFIDENCE")
            or self.min_confidence
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

    def _history_rows(self, symbol: str, contract_type: str) -> list:
        """Closed trade rows for edge scoring (symbol|type)."""
        rows = []
        ct = str(contract_type or "").upper()
        for e in self.trade_log:
            st = str(e.get("status") or "").lower()
            if st not in {"win", "loss", "push"}:
                continue
            if str(e.get("symbol") or "") != symbol:
                continue
            if ct and str(e.get("contract_type") or "").upper() != ct:
                continue
            rows.append(
                {
                    "profit": e.get("profit"),
                    "status": st,
                    "symbol": symbol,
                    "contract_type": e.get("contract_type"),
                }
            )
        for row in self.closed_trades[-200:]:
            meta = row.get("meta") or {}
            if str(meta.get("symbol") or "") != symbol:
                continue
            if ct and str(meta.get("contract_type") or "").upper() != ct:
                continue
            rows.append(
                {
                    "profit": row.get("profit"),
                    "symbol": symbol,
                    "contract_type": meta.get("contract_type"),
                }
            )
        return rows[-500:]

    def _apply_analytics_gate(
        self,
        intent: Dict[str, Any],
        ticks: list,
        *,
        family: str,
    ) -> Optional[Dict[str, Any]]:
        """
        AI trade filter + quality/edge gates. Returns enriched intent or None to skip.
        """
        if not self.analytics_gate:
            return intent
        symbol = str(intent.get("symbol") or "")
        ct = str(intent.get("contract_type") or "")
        hist = self._history_rows(symbol, ct)
        # Hour / symbol analytics soft skip
        skip_h, why_h = self.session_analytics.skip_hour()
        if skip_h:
            logger.info("Analytics skip hour: %s", why_h)
            # Do not hard-block forever — only reduce via filter confidence
        skip_s, why_s = self.session_analytics.skip_symbol(symbol)
        if skip_s:
            logger.info("Analytics weak symbol: %s", why_s)

        filt = evaluate_setup(
            ticks,
            symbol=symbol,
            contract_type=ct,
            family=family,
            history_rows=hist,
            recent_rows=hist[-100:],
            signal_confidence=float(intent.get("confidence") or 0),
            global_samples=self.learner.global_samples(),
        )
        self.last_filter = filt
        intent["live_edge"] = float(filt["live_edge"]["live_edge"])
        intent["quality_score"] = float(filt["quality"]["quality_score"])
        intent["pattern_strength"] = float(
            filt["pattern_strength"]["pattern_strength"]
        )
        intent["pattern_clarity"] = float(
            (filt.get("pattern_clarity") or {}).get("pattern_clarity") or 0
        )
        intent["edge_score"] = float(
            (filt.get("historical_edge") or {}).get("edge_score") or 0
        )
        intent["sample_size"] = int(filt.get("sample_size") or 0)
        # No-trade / EV decision fields
        nt = filt.get("no_trade") or {}
        intent["decision_quality"] = filt.get("decision_quality")
        intent["ev"] = filt.get("ev")
        intent["p_win"] = filt.get("p_win")
        intent["risk_pct"] = filt.get("risk_pct")
        intent["regime"] = filt.get("regime")
        intent["hpp"] = filt.get("hpp")
        intent["hpp_velocity"] = filt.get("hpp_velocity")
        intent["edge_decay_pct"] = filt.get("edge_decay_pct")
        intent["momentum_persistence"] = filt.get("momentum_persistence")
        intent["mp_score"] = filt.get("mp_score")
        intent["no_trade"] = {
            "status": nt.get("status"),
            "reason": nt.get("reason"),
            "allow": nt.get("allow"),
            "ev": nt.get("ev"),
            "risk_pct": nt.get("risk_pct"),
            "regime": nt.get("regime"),
            "ensemble": nt.get("ensemble"),
            "trade_quality": (nt.get("trade_quality") or {}).get("trade_quality"),
            "blocks": nt.get("blocks"),
        }
        intent["filter"] = {
            "recommendation": filt.get("recommendation"),
            "market_condition": filt.get("market_condition"),
            "expected_edge": filt.get("expected_edge"),
            "reasons": filt.get("reasons"),
            "copilot": filt.get("copilot"),
            "allow": filt.get("allow"),
            "pattern_clarity": intent["pattern_clarity"],
            "edge_score": intent["edge_score"],
            "sample_size": intent["sample_size"],
            "metrics": (filt.get("pattern_clarity") or {}).get("metrics")
            or ((filt.get("pattern_clarity") or {}).get("contract_profile") or {}).get(
                "metrics"
            ),
            "profile_recommendation": (filt.get("pattern_clarity") or {}).get(
                "recommendation"
            ),
            "no_trade": intent["no_trade"],
            "ev": intent.get("ev"),
            "decision_quality": intent.get("decision_quality"),
            "regime": intent.get("regime"),
        }
        # Persist profile metrics on intent for learning at trade close
        intent["profile_metrics"] = intent["filter"].get("metrics")
        if skip_s:
            intent["live_edge"] = max(0.0, intent["live_edge"] - 12)
            intent["quality_score"] = max(0.0, intent["quality_score"] - 10)
            clarity = float(
                (filt.get("pattern_clarity") or {}).get("pattern_clarity") or 0
            )
            edge_sc = float(
                (filt.get("historical_edge") or {}).get("edge_score") or 0
            )
            n_samp = int(filt.get("sample_size") or 0)
            # Weak symbol: still require no-trade allow when production sample
            classic_ok = (
                intent["pattern_strength"] >= 75
                and clarity >= 80
                and edge_sc >= 80
                and intent["live_edge"] >= 80
                and intent["quality_score"] >= 80
                and n_samp >= 500
            )
            filt["allow"] = classic_ok and bool(nt.get("allow", True))
            intent["pattern_clarity"] = clarity
        if not filt.get("allow"):
            reason = (nt.get("reason") if nt and not nt.get("allow") else None) or (
                filt.get("copilot") or ""
            )[:120]
            logger.info(
                "AI filter SKIP %s %s: status=%s condition=%s edge=%s live=%.0f "
                "pattern=%.0f quality=%.0f dq=%s EV=%s regime=%s — %s",
                symbol,
                ct,
                nt.get("status") or filt.get("action"),
                filt.get("market_condition"),
                filt.get("expected_edge"),
                intent["live_edge"],
                intent["pattern_strength"],
                intent["quality_score"],
                intent.get("decision_quality"),
                intent.get("ev"),
                intent.get("regime"),
                reason,
            )
            return None
        logger.info(
            "AI filter TRADE %s %s live=%.0f pattern=%.0f quality=%.0f "
            "decision_q=%s EV=%s risk%%=%.2f regime=%s",
            symbol,
            ct,
            intent["live_edge"],
            intent["pattern_strength"],
            intent["quality_score"],
            intent.get("decision_quality"),
            intent.get("ev"),
            float(intent.get("risk_pct") or 0),
            intent.get("regime"),
        )
        return intent

    async def scan_markets(self) -> Optional[Dict[str, Any]]:
        """
        Scan all markets for Digits + Rise/Fall candidates.
        Apply adaptive learning, enforce min_confidence (>=80%), pick best market.
        """
        signals = []
        min_conf_base = self.min_confidence
        phase = self.learner.cold_start_phase()

        # Self-optimizing order: priority book first, DeepSeek-preferred markets next
        try:
            from src.analytics.market_scanner import get_priority_book

            scan_symbols = get_priority_book().ordered_symbols(
                list(self.active_symbols or [])
            )
        except Exception:
            scan_symbols = list(self.active_symbols or [])
        try:
            ds_pref = self.deepseek.preferred_symbols()
            if ds_pref:
                pref_set = {s.upper() for s in ds_pref}
                head = [s for s in scan_symbols if str(s).upper() in pref_set]
                tail = [s for s in scan_symbols if str(s).upper() not in pref_set]
                scan_symbols = head + tail
        except Exception:
            pass

        for symbol in scan_symbols:
            if self.offer_gate.is_symbol_blocked(symbol):
                logger.info(
                    "Skip %s: market offer blocked (closed/unavailable)", symbol
                )
                continue
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
            # Category-aware contracts: e.g. forex/crypto never scan digits
            try:
                from src.strategy.market_categories import (
                    filter_allowed_for_symbol,
                    market_profile,
                )

                mprof = market_profile(symbol)
                if not allowed:
                    allowed = list(mprof.get("allowed_contracts") or ["CALL", "PUT"])
                else:
                    allowed = filter_allowed_for_symbol(symbol, allowed) or allowed
            except Exception:
                mprof = {"category": "unknown", "scoring_path": "directional"}
            digit_allowed = [t for t in allowed if is_digit_contract(t)]
            rf_allowed = [t for t in allowed if is_rise_fall(t)]
            # If allow-list empty after filter, fall back to category defaults
            if not allowed:
                digit_allowed = [
                    "DIGITOVER",
                    "DIGITUNDER",
                    "DIGITEVEN",
                    "DIGITODD",
                ]
                rf_allowed = ["CALL", "PUT"]
                try:
                    from src.strategy.market_categories import (
                        SYNTHETIC_VOL,
                        classify_market,
                        allowed_contracts,
                    )

                    if classify_market(symbol) != SYNTHETIC_VOL:
                        digit_allowed = []
                        rf_allowed = sorted(allowed_contracts(classify_market(symbol)))
                except Exception:
                    pass

            # DeepSeek hard bans — remove contract types from allow-lists
            digit_allowed = [
                t
                for t in digit_allowed
                if not self.deepseek.is_banned(symbol, t)
            ]
            rf_allowed = [
                t for t in rf_allowed if not self.deepseek.is_banned(symbol, t)
            ]

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

            # ---- Digits path (isolated: errors must not kill RF path) ----
            if digit_allowed and not skip_d:
              try:
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
                    if self.deepseek.is_banned(symbol, signal_type):
                        logger.info(
                            "DeepSeek ban skip %s %s", symbol, signal_type
                        )
                        signal_type = None
                if signal_type:
                    adj = self.learner.adjust_confidence(
                        symbol, signal_type, conf or raw_conf
                    )
                    ds_mult = self.deepseek.confidence_multiplier(
                        symbol, signal_type
                    )
                    adj = float(max(0.0, min(0.99, adj * ds_mult)))
                    # "reduce" verdict → slightly higher bar
                    if ds_mult < 0.9:
                        need_extra = 0.03 * (0.9 - ds_mult) / 0.4
                    else:
                        need_extra = 0.0
                    need = self.learner.effective_min_confidence(
                        min_conf_base,
                        family="digits",
                        contract_type=signal_type,
                    ) + need_extra
                    if adj >= need:
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
                            intent["market_category"] = (mprof or {}).get("category")
                            intent["scoring_path"] = (mprof or {}).get("scoring_path")
                            intent["raw_confidence"] = conf or raw_conf
                            intent["regime"] = dig_reg
                            intent["learn_bonus"] = self.learner.selection_bonus(
                                symbol, intent["contract_type"]
                            )
                            intent["trend_strength"] = 0.0
                            intent["deepseek_mult"] = ds_mult
                            intent["deepseek_boost"] = self.deepseek.selection_boost(
                                symbol, signal_type
                            )
                            intent["min_conf_used"] = need
                            gated = self._apply_analytics_gate(
                                intent, ticks, family="digits"
                            )
                            if gated:
                                candidates.append(gated)
              except Exception as e:
                logger.exception("Digits path error on %s (RF continues): %s", symbol, e)
            elif skip_d:
                logger.debug("Skip digits %s: %s", symbol, dig_reason)

            # ---- Rise/Fall path (isolated: errors must not kill digits) ----
            if rf_allowed and not skip_rf:
              try:
                rf_need = self.learner.effective_min_confidence(
                    min_conf_base, family="rise_fall", contract_type="CALL"
                )
                trend = analyze_trend(ticks)
                pro = analyze_pro_trend(
                    ticks, symbol=symbol, min_confidence=rf_need
                )
                # Prefer pro-trend when it agrees or is stronger; require
                # agreement with classic tools when both fire different sides
                mom_type = trend.get("contract_type")
                pro_type = pro.get("contract_type")
                mom_conf = float(trend.get("confidence") or 0.0)
                pro_conf = float(pro.get("confidence") or 0.0)

                if pro_type and mom_type and pro_type == mom_type:
                    rf_type = pro_type
                    rf_conf = min(0.98, 0.45 * mom_conf + 0.55 * pro_conf + 0.04)
                elif pro_type and pro_conf >= max(rf_need, mom_conf):
                    rf_type = pro_type
                    rf_conf = pro_conf * 0.97
                elif mom_type and mom_conf >= rf_need and not pro_type:
                    rf_type = mom_type
                    rf_conf = mom_conf * 0.92
                elif mom_type and pro_type and mom_type != pro_type:
                    # Conflict → no trade (protect win rate)
                    rf_type = None
                    rf_conf = min(mom_conf, pro_conf) * 0.55
                else:
                    rf_type = pro_type or mom_type
                    rf_conf = max(pro_conf, mom_conf) * 0.85

                # Penalize mild chop even if tools pass
                chop = float(rf_reg.get("chop_score") or 0)
                if chop > 0.45:
                    rf_conf *= 1.0 - (chop - 0.45) * 0.8
                if rf_type and rf_type in rf_allowed:
                    if self.deepseek.is_banned(symbol, rf_type):
                        logger.info("DeepSeek ban skip %s %s", symbol, rf_type)
                        rf_type = None
                if rf_type and rf_type in rf_allowed:
                    adj = self.learner.adjust_confidence(symbol, rf_type, rf_conf)
                    # DeepSeek per-type learning multiplier
                    ds_mult = self.deepseek.confidence_multiplier(symbol, rf_type)
                    adj = float(max(0.0, min(0.99, adj * ds_mult)))
                    if ds_mult < 0.9:
                        need_extra = 0.03 * (0.9 - ds_mult) / 0.4
                    else:
                        need_extra = 0.0
                    need = self.learner.effective_min_confidence(
                        min_conf_base,
                        family="rise_fall",
                        contract_type=rf_type,
                    ) + need_extra
                    if adj >= need:
                        intent = self.strategy_engine.apply_signal(
                            symbol=symbol,
                            signal_type=rf_type,
                            signal_barrier=None,
                            confidence=adj,
                        )
                        if intent:
                            intent["family"] = "rise_fall"
                            intent["market_category"] = (mprof or {}).get("category")
                            intent["scoring_path"] = (mprof or {}).get("scoring_path")
                            intent["raw_confidence"] = rf_conf
                            intent["trend"] = trend
                            intent["pro_trend"] = {
                                k: pro.get(k)
                                for k in (
                                    "notes",
                                    "call_pts",
                                    "put_pts",
                                    "ema50",
                                    "ema200",
                                    "rsi14",
                                    "htf_bull",
                                    "htf_bear",
                                    "pattern",
                                )
                            }
                            intent["regime"] = rf_reg
                            intent["trend_strength"] = float(
                                trend.get("strength") or pro.get("confidence") or 0.0
                            )
                            intent["learn_bonus"] = self.learner.selection_bonus(
                                symbol, intent["contract_type"]
                            )
                            intent["deepseek_mult"] = ds_mult
                            intent["deepseek_boost"] = self.deepseek.selection_boost(
                                symbol, rf_type
                            )
                            intent["min_conf_used"] = need
                            gated = self._apply_analytics_gate(
                                intent, ticks, family="rise_fall"
                            )
                            if gated:
                                candidates.append(gated)
                    else:
                        logger.debug(
                            "%s trend %s conf=%.2f adj=%.2f < min=%.2f phase=%s",
                            symbol,
                            rf_type,
                            rf_conf,
                            adj,
                            need,
                            phase,
                        )
              except Exception as e:
                logger.exception("Rise/Fall path error on %s (digits continue): %s", symbol, e)
            elif skip_rf:
                logger.info(
                    "Skip rise/fall %s: %s (chop=%.2f eff=%.2f)",
                    symbol,
                    rf_reason,
                    rf_reg.get("chop_score", 0),
                    rf_reg.get("efficiency", 0),
                )

            # ---- Minute Rise/Fall (isolated) ----
            if self.enable_minute and rf_allowed and not skip_rf:
              try:
                need = max(
                    self.minute_min_conf,
                    self.learner.effective_min_confidence(
                        min_conf_base,
                        family="minute_rise_fall",
                        contract_type="CALL",
                    )
                    * 0.97,
                )
                msig = analyze_minute(
                    ticks,
                    period_sec=60,
                    duration_minutes=self.minute_duration,
                    min_confidence=need,
                )
                if msig and msig["contract_type"] in rf_allowed:
                    mct = msig["contract_type"]
                    if self.deepseek.is_banned(symbol, mct):
                        logger.info("DeepSeek ban skip minute %s %s", symbol, mct)
                        msig = None
                if msig and msig["contract_type"] in rf_allowed:
                    mct = msig["contract_type"]
                    adj = self.learner.adjust_confidence(
                        symbol, mct, float(msig["confidence"])
                    )
                    ds_mult = self.deepseek.confidence_multiplier(symbol, mct)
                    adj = float(max(0.0, min(0.99, adj * ds_mult)))
                    if adj >= need:
                        intent = self.strategy_engine.apply_signal(
                            symbol=symbol,
                            signal_type=mct,
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
                            intent["deepseek_mult"] = ds_mult
                            intent["deepseek_boost"] = self.deepseek.selection_boost(
                                symbol, mct
                            )
                            intent["learn_bonus"] = self.learner.selection_bonus(
                                symbol, intent["contract_type"]
                            )
                            intent["trend_strength"] = float(
                                (msig.get("details") or {}).get("call_pts")
                                or (msig.get("details") or {}).get("put_pts")
                                or 0
                            ) / 5.0
                            gated = self._apply_analytics_gate(
                                intent, ticks, family="minute_rise_fall"
                            )
                            if gated:
                                candidates.append(gated)
                                logger.info(
                                    "Minute candidate %s %s conf=%.2f dur=%sm notes=%s",
                                    symbol,
                                    msig["contract_type"],
                                    adj,
                                    msig["duration"],
                                    (msig.get("details") or {}).get("notes"),
                                )
              except Exception as e:
                logger.exception("Minute RF path error on %s: %s", symbol, e)

            # Custom strategy marketplace rules (no-code)
            try:
                last_won = self.risk_manager.consecutive_losses == 0 and (
                    self.risk_manager.wins_today > 0
                )
                for hit in self.strategy_builder.evaluate_all(
                    ticks, last_won=last_won
                ):
                    ct = normalize_contract_type(hit.get("contract_type")) or str(
                        hit.get("contract_type") or ""
                    ).upper()
                    if ct not in (allowed or digit_allowed or rf_allowed or []):
                        # allow if type is generally valid
                        if not (
                            is_digit_contract(ct)
                            or is_rise_fall(ct)
                            or ct in {"CALL", "PUT"}
                        ):
                            continue
                    conf_sb = min_conf_base
                    adj = self.learner.adjust_confidence(symbol, ct, conf_sb)
                    if adj < min_conf_base:
                        continue
                    intent = self.strategy_engine.apply_signal(
                        symbol=symbol,
                        signal_type=ct,
                        signal_barrier=hit.get("barrier"),
                        confidence=adj,
                    )
                    if not intent:
                        continue
                    intent["family"] = "strategy_builder"
                    intent["raw_confidence"] = conf_sb
                    intent["strategy_rule"] = hit.get("strategy_name")
                    gated = self._apply_analytics_gate(
                        intent, ticks, family="digits"
                    )
                    if gated:
                        candidates.append(gated)
            except Exception as e:
                logger.debug("strategy_builder eval failed: %s", e)

            for intent in candidates:
                # Flat stake mode: never martingale-double (stops loss pits)
                if self.stake_mode == "flat":
                    intent["stake"] = float(intent.get("base_stake") or intent["stake"])
                # Ensure duration fields for offer-gate checks
                if not intent.get("duration"):
                    intent["duration"] = int(
                        getattr(runtime, "duration", None) or TRADE_DURATION_TICKS
                    )
                if not intent.get("duration_unit"):
                    intent["duration_unit"] = "t"
                if intent.get("family") == "minute_rise_fall" or intent.get(
                    "horizon"
                ) == "minute":
                    intent["duration_unit"] = "m"
                    intent["duration"] = int(
                        intent.get("duration") or self.minute_duration
                    )
                blocked, br = self.offer_gate.is_blocked(
                    symbol,
                    contract_type=str(intent.get("contract_type") or ""),
                    duration=int(intent.get("duration") or 5),
                    duration_unit=str(intent.get("duration_unit") or "t"),
                )
                if blocked:
                    logger.info(
                        "Skip candidate %s %s: offer_gate (%s)",
                        symbol,
                        intent.get("contract_type"),
                        br,
                    )
                    continue
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
                    "(raw=%.2f) family=%s strat=%s",
                    symbol,
                    intent["contract_type"],
                    intent["stake"],
                    intent.get("barrier"),
                    intent["confidence"],
                    intent.get("raw_confidence"),
                    intent.get("family"),
                    intent["strategy"],
                )

        self._last_scan_signals = list(signals)
        if not signals:
            logger.info(
                "No signals ≥ %.0f%% confidence across %s markets "
                "(anti_spiral=%s offer_blocks=%s)",
                min_conf_base * 100,
                len(self.active_symbols),
                self.anti_spiral.snapshot(),
                (self.offer_gate.snapshot() or {}).get("count"),
            )
            return None

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

        head = await self.scan_markets()
        if not head:
            logger.info(
                "No qualifying signals | balance=%s %s open=%s strategies=%s",
                balance,
                self.client.get_currency(),
                open_count,
                self.strategy_engine.snapshots(),
            )
            return None

        # Ranked multi-market list — skip dead duration/offers, try next open market
        candidates = self.selector.rank_trades(self._last_scan_signals or [head])
        if not candidates:
            candidates = [head]
        assert balance is not None

        last_hard_error: Optional[str] = None
        for cand_i, best in enumerate(candidates[:8]):
            if self.deepseek.is_banned(
                str(best.get("symbol") or ""),
                str(best.get("contract_type") or ""),
            ):
                logger.info(
                    "DeepSeek ban skip execute %s %s",
                    best.get("symbol"),
                    best.get("contract_type"),
                )
                continue

            quality_risk = best.get("risk_pct")
            if quality_risk is None:
                dq = best.get("decision_quality")
                if dq is not None:
                    from src.analytics.no_trade_engine import risk_pct_from_quality

                    quality_risk = risk_pct_from_quality(float(dq))
            if quality_risk is not None and float(quality_risk) <= 0:
                logger.info(
                    "No-trade risk sizing 0%% (decision_quality=%s) — skip %s %s",
                    best.get("decision_quality"),
                    best.get("symbol"),
                    best.get("contract_type"),
                )
                continue

            base_for_adapt = float(
                quality_risk
                if quality_risk is not None
                else (self.risk_manager.max_stake_pct or 1.0)
            )
            base_for_adapt = min(
                base_for_adapt, float(self.risk_manager.max_stake_pct or 2.0)
            )
            risk_plan = adaptive_risk_pct(
                base_risk_pct=base_for_adapt,
                consecutive_losses=self.risk_manager.consecutive_losses,
                consecutive_wins=0,
                daily_pnl=self.risk_manager.daily_pnl,
                session_start_balance=self.risk_manager.session_start_balance,
                min_risk_pct=0.25 if base_for_adapt <= 0.5 else 0.5,
                max_risk_pct=min(2.0, float(self.risk_manager.max_stake_pct or 2.0)),
            )
            risk_plan["quality_risk_pct"] = quality_risk
            risk_plan["decision_quality"] = best.get("decision_quality")
            raw_stake = float(best.get("stake", best.get("base_stake", MIN_STAKE)))
            if self.risk_manager.base_stake is not None:
                raw_stake = float(self.risk_manager.base_stake)
            adapt_stake = stake_from_risk(
                balance,
                risk_plan["risk_pct"],
                min_stake=MIN_STAKE,
                max_stake=self.risk_manager.max_stake,
            )
            raw_stake = min(raw_stake, adapt_stake) if adapt_stake > 0 else raw_stake
            stake = self.risk_manager.clamp_stake(raw_stake, balance)
            best["risk_plan"] = risk_plan
            best["adaptive_stake"] = adapt_stake
            best["martingale_safety"] = survival_probability(
                balance, stake or MIN_STAKE, win_rate=0.48, max_levels=5
            )
            if (
                self.stake_mode == "martingale"
                and best["martingale_safety"].get("danger_level")
                in {"HIGH", "CRITICAL"}
            ):
                stake = self.risk_manager.clamp_stake(
                    min(stake, float(self.risk_manager.base_stake or MIN_STAKE)),
                    balance,
                )
            if stake <= 0:
                continue

            stake_decision = self.risk_manager.can_trade(
                balance, open_trades=open_count, proposed_stake=stake
            )
            if not stake_decision:
                logger.warning("Risk block on stake: %s", stake_decision.reason)
                return None

            duration = int(best.get("duration") or TRADE_DURATION_TICKS)
            duration_unit = str(best.get("duration_unit") or "t")
            if (
                best.get("family") == "minute_rise_fall"
                or best.get("horizon") == "minute"
            ):
                duration_unit = "m"
                duration = int(best.get("duration") or self.minute_duration)
            horizon = str(
                best.get("horizon")
                or ("minute" if duration_unit == "m" else "tick")
            )

            blocked, br = self.offer_gate.is_blocked(
                str(best["symbol"]),
                contract_type=str(best.get("contract_type") or ""),
                duration=duration,
                duration_unit=duration_unit,
            )
            if blocked:
                logger.info(
                    "Offer gate skip %s %s %s%s (%s)",
                    best["symbol"],
                    best.get("contract_type"),
                    duration,
                    duration_unit,
                    br,
                )
                continue

            logger.info(
                "Trade candidate [%s/%s]: %s %s stake=%s barrier=%s conf=%.2f "
                "dur=%s%s family=%s horizon=%s execute=%s",
                cand_i + 1,
                min(8, len(candidates)),
                best["symbol"],
                best["contract_type"],
                stake,
                best.get("barrier"),
                best.get("confidence", 0),
                duration,
                duration_unit,
                best.get("family"),
                horizon,
                self.execute_trades,
            )

            result, duration, duration_unit, offer_reason = (
                await self._propose_with_offer_fallbacks(
                    best=best,
                    stake=stake,
                    duration=duration,
                    duration_unit=duration_unit,
                )
            )
            horizon = str(
                best.get("horizon")
                or ("minute" if duration_unit == "m" else "tick")
            )

            if not result:
                err = self.executor.last_error or "unknown"
                last_hard_error = err
                self._log_trade_event(
                    {
                        "status": "failed_offer"
                        if offer_reason
                        else "failed",
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
                        "offer_reason": offer_reason,
                    }
                )
                if offer_reason:
                    # Soft skip — keep scanning other open markets this cycle
                    logger.warning(
                        "Offer rejected %s %s (%s): %s — try next market",
                        best["symbol"],
                        best.get("contract_type"),
                        offer_reason,
                        err,
                    )
                    continue
                # Hard failure (network etc.) — notify once, stop cycle
                await self.telegram.send_notification(
                    self.telegram.format_trade_error(
                        title="Trade failed",
                        error=err,
                        balance=self.client.get_balance(),
                        currency=self.client.get_currency(),
                        symbol=best["symbol"],
                        contract_type=best["contract_type"],
                        stake=stake,
                    )
                )
                return None

            if result.get("skipped_low_payout"):
                err = (
                    result.get("error")
                    or self.executor.last_error
                    or "payout too low"
                )
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
                self.anti_spiral.note_selected(
                    str(best["symbol"]), str(best["contract_type"])
                )
                logger.info(
                    "Skipped low-payout quote: %s %s — try next market",
                    best["symbol"],
                    best["contract_type"],
                )
                continue

            if result.get("buy_failed"):
                err = result.get("error") or self.executor.last_error or "buy failed"
                offer_reason = self.offer_gate.note_error(
                    str(best["symbol"]),
                    err,
                    contract_type=str(best.get("contract_type") or ""),
                    duration=duration,
                    duration_unit=duration_unit,
                )
                self._log_trade_event(
                    {
                        "status": "buy_failed",
                        "symbol": best["symbol"],
                        "contract_type": best["contract_type"],
                        "stake": stake,
                        "error": err,
                        "family": best.get("family"),
                        "horizon": horizon,
                        "offer_reason": offer_reason,
                    }
                )
                if offer_reason:
                    logger.warning(
                        "Buy offer block %s: %s — try next market",
                        best["symbol"],
                        err,
                    )
                    continue
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

            # Success path (proposal and/or buy)
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

        logger.info(
            "All candidates skipped/rejected this cycle (offer_blocks=%s last_err=%s)",
            (self.offer_gate.snapshot() or {}).get("count"),
            last_hard_error,
        )
        return None

    async def _propose_with_offer_fallbacks(
        self,
        *,
        best: Dict[str, Any],
        stake: float,
        duration: int,
        duration_unit: str,
    ) -> tuple:
        """
        Try primary duration then fallbacks. On duration/market-closed errors,
        cool down that offer and continue. Returns
        (result|None, duration_used, unit_used, offer_reason|None).
        """
        symbol = str(best["symbol"])
        contract_type = str(best["contract_type"])
        primary = (int(duration), str(duration_unit or "t").lower())
        attempts = [primary] + duration_fallbacks(primary[0], primary[1])
        last_offer_reason: Optional[str] = None
        last_dur, last_unit = primary

        for d, u in attempts:
            blocked, _br = self.offer_gate.is_blocked(
                symbol,
                contract_type=contract_type,
                duration=d,
                duration_unit=u,
            )
            if blocked or self.offer_gate.is_symbol_blocked(symbol):
                continue
            last_dur, last_unit = d, u
            result = await self.executor.propose_and_buy(
                symbol=symbol,
                contract_type=contract_type,
                stake=stake,
                barrier=best.get("barrier"),
                currency=self.client.get_currency(),
                duration=d,
                duration_unit=u,
                execute=self.execute_trades,
                min_net_return=self.min_net_return if self.execute_trades else None,
            )
            if result and not result.get("buy_failed"):
                # proposal ok (maybe dry-run or executed or low payout skip)
                return result, d, u, None

            err = None
            if result and result.get("buy_failed"):
                err = result.get("error") or self.executor.last_error
            else:
                err = self.executor.last_error or "unknown"

            reason = self.offer_gate.note_error(
                symbol,
                err,
                contract_type=contract_type,
                duration=d,
                duration_unit=u,
            )
            last_offer_reason = reason
            if reason == REASON_DURATION:
                logger.info(
                    "Duration not offered %s %s %s%s — try fallback",
                    symbol,
                    contract_type,
                    d,
                    u,
                )
                continue
            if reason in {REASON_MARKET_CLOSED, REASON_UNAVAILABLE}:
                logger.warning(
                    "Market not tradeable %s (%s): %s",
                    symbol,
                    reason,
                    err,
                )
                return None, d, u, reason
            # Non-offer error (balance, network, …)
            if result and result.get("buy_failed"):
                return result, d, u, None
            return None, d, u, None

        return None, last_dur, last_unit, last_offer_reason or REASON_DURATION

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
                # Self-optimizing market scan priority + every-500 PF report
                try:
                    from src.analytics.market_scanner import get_priority_book

                    rep = get_priority_book().record_trade(
                        str(symbol),
                        is_win=is_win,
                        profit=float(profit),
                        hpp_velocity=meta.get("hpp_velocity"),
                        clarity=meta.get("pattern_clarity"),
                    )
                    if rep:
                        logger.info(
                            "Market scanner report (every 500):\n%s",
                            rep.get("display"),
                        )
                except Exception as e:
                    logger.debug("market priority record failed: %s", e)
            # Session recorder + hour/index analytics
            try:
                self.session_analytics.record(
                    symbol=str(symbol or "?"),
                    contract_type=str(contract_type),
                    is_win=is_win,
                    profit=profit,
                    stake=meta.get("stake"),
                    indicators={
                        "live_edge": meta.get("live_edge"),
                        "quality_score": meta.get("quality_score"),
                        "pattern_strength": meta.get("pattern_strength"),
                        "family": meta.get("family"),
                    },
                )
            except Exception as e:
                logger.debug("session_analytics record failed: %s", e)
            # Contract profile + HPP self-learning (outcome attribution)
            try:
                from src.analytics.contract_profiles import get_weight_engine

                metrics = meta.get("profile_metrics") or meta.get("metrics") or {}
                if not metrics and meta.get("filter"):
                    metrics = (meta.get("filter") or {}).get("metrics") or {}
                if metrics and contract_type:
                    get_weight_engine().record_outcome(
                        str(contract_type),
                        metrics,
                        is_win,
                        profit=float(profit),
                        symbol=str(symbol or ""),
                        clarity=meta.get("pattern_clarity")
                        or meta.get("quality_score"),
                    )
                # HPP time series: snapshot every N trades / daily
                from src.analytics.hpp_timeseries import get_hpp_timeseries

                get_hpp_timeseries().note_trade()
            except Exception as e:
                logger.debug("contract profile learning failed: %s", e)
            # Calibration + rich outcome attribution (which metrics matter?)
            try:
                from src.analytics.calibration import get_calibration_tracker

                dq = meta.get("decision_quality")
                if dq is None:
                    dq = meta.get("quality_score")
                mp = meta.get("momentum_persistence") or {}
                if not isinstance(mp, dict):
                    mp = {}
                get_calibration_tracker().record(
                    contract=str(contract_type or ""),
                    is_win=is_win,
                    predicted_p=meta.get("p_win") or meta.get("confidence"),
                    quality=dq,
                    entropy=(meta.get("rolling_entropy") or {}).get("composite_entropy")
                    if isinstance(meta.get("rolling_entropy"), dict)
                    else meta.get("entropy"),
                    clarity=meta.get("pattern_clarity"),
                    hpp=meta.get("hpp"),
                    velocity=meta.get("hpp_velocity"),
                    strength=meta.get("pattern_strength"),
                    momentum=(mp.get("momentum") or {}).get("momentum_score")
                    if isinstance(mp.get("momentum"), dict)
                    else meta.get("momentum"),
                    persistence=(mp.get("persistence") or {}).get("persistence")
                    if isinstance(mp.get("persistence"), dict)
                    else meta.get("persistence"),
                    momentum_persistence=meta.get("mp_score")
                    or (mp.get("mp_score") if isinstance(mp, dict) else None),
                    persistence_velocity=(
                        (mp.get("persistence_velocity") or {}).get("velocity")
                        if isinstance(mp.get("persistence_velocity"), dict)
                        else meta.get("persistence_velocity")
                    ),
                    persistence_acceleration=(
                        (mp.get("persistence_velocity") or {}).get("acceleration")
                        if isinstance(mp.get("persistence_velocity"), dict)
                        else meta.get("persistence_acceleration")
                    ),
                    regime=meta.get("regime"),
                    profit=float(profit),
                    stake=meta.get("stake"),
                    symbol=str(symbol or ""),
                    extra={
                        "ev": meta.get("ev"),
                        "live_edge": meta.get("live_edge"),
                        "family": meta.get("family"),
                    },
                )
            except Exception as e:
                logger.debug("calibration record failed: %s", e)

        self.closed_trades.append(
            {"contract": contract, "meta": meta, "profit": profit}
        )
        # Strategy failure soft-stop: expected WR vs current collapse
        try:
            if symbol and contract_type:
                n = self.learner.samples(str(symbol), str(contract_type))
                wr = self.learner.win_rate(str(symbol), str(contract_type))
                if n >= 12 and wr < 0.38:
                    logger.warning(
                        "Strategy failure %s|%s wr=%.0f%% n=%s — smart pause 10m",
                        symbol,
                        contract_type,
                        wr * 100,
                        n,
                    )
                    self.risk_manager.pause(
                        minutes=10,
                        reason=f"strategy_failure_{symbol}_{contract_type}_wr={wr:.0%}",
                    )
        except Exception:
            pass
        # DeepSeek periodic analysis → updates type multipliers for learning curve
        try:
            if self.deepseek.note_closed_trade():
                self._run_deepseek_analysis(source="auto")
        except Exception as e:
            logger.debug("DeepSeek auto-analyze skipped: %s", e)

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
            "deepseek": self.deepseek.snapshot(),
        }

    def _run_deepseek_analysis(self, source: str = "manual") -> Optional[Dict[str, Any]]:
        """Analyze recent trades with DeepSeek and apply type multipliers."""
        trades = [
            e
            for e in self.trade_log
            if str(e.get("status") or "").lower() in {"win", "loss", "push"}
        ]
        # Fallback: build from closed_trades if log is thin
        if len(trades) < 2:
            for row in self.closed_trades[-30:]:
                meta = row.get("meta") or {}
                trades.append(
                    {
                        "status": "win" if float(row.get("profit") or 0) > 0 else "loss",
                        "symbol": meta.get("symbol"),
                        "contract_type": meta.get("contract_type"),
                        "stake": meta.get("stake"),
                        "profit": row.get("profit"),
                        "confidence": meta.get("confidence"),
                        "family": meta.get("family"),
                    }
                )
        rec = self.deepseek.analyze(
            trades=trades,
            learning=self.learner.snapshot(),
            risk=self.risk_manager.session_limits_snapshot(),
            strategies=self.strategy_engine.snapshots(),
        )
        if rec:
            logger.info(
                "DeepSeek analysis (%s): score=%s summary=%s",
                source,
                rec.get("risk_score"),
                str(rec.get("summary") or "")[:100],
            )
            # Optional: apply stake advice conservatively (lower only)
            stake_adv = rec.get("stake_advice") or {}
            if str(stake_adv.get("action") or "").lower() == "lower":
                try:
                    pct = float(stake_adv.get("pct_of_balance") or 0)
                    if 0.5 <= pct <= 2.0:
                        self.risk_manager.max_stake_pct = min(
                            self.risk_manager.max_stake_pct, pct
                        )
                except (TypeError, ValueError):
                    pass
            sess = rec.get("session_advice") or {}
            if sess.get("stop_loss_pct") is not None:
                try:
                    self.risk_manager.configure_session_risk(
                        stop_loss_pct=float(sess["stop_loss_pct"]),
                        target_rr=(
                            float(sess["target_rr"])
                            if sess.get("target_rr") is not None
                            else None
                        ),
                    )
                except (TypeError, ValueError):
                    pass
        return rec

    def configure_risk_ui(
        self,
        *,
        base_stake: Optional[float] = None,
        stop_loss_pct: Optional[float] = None,
        target_rr: Optional[float] = None,
        max_stake_pct: Optional[float] = None,
        stop_on_target: Optional[bool] = None,
        reset_session: bool = False,
    ) -> Dict[str, Any]:
        """Dashboard / API: update stake + session target/stop-loss live."""
        applied = self.risk_manager.configure_session_risk(
            stop_loss_pct=stop_loss_pct,
            target_rr=target_rr,
            stop_on_target=stop_on_target,
            base_stake=base_stake,
            max_stake_pct=max_stake_pct,
        )
        if reset_session:
            bal = self.client.get_balance()
            self.risk_manager.reset_session_run(bal)
            applied = self.risk_manager.session_limits_snapshot(bal)
        logger.info("Risk UI update: %s", applied)
        return {
            **applied,
            **self.risk_status(),
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
            "offer_gate": self.offer_gate.snapshot(),
            "deepseek": self.deepseek.snapshot(),
            "stake_mode": self.stake_mode,
            "enable_minute": self.enable_minute,
            "minute_duration": self.minute_duration,
            "min_net_return": self.min_net_return,
            "recent_trades": list(reversed(self.trade_log[-20:])),
            "open_trade_details": self._open_trade_details(),
            "analytics": self.analytics_snapshot(),
        }

    def analytics_snapshot(self) -> Dict[str, Any]:
        """Digit heatmap, edge scan, filter, martingale safety, session insights."""
        # All active symbols so dashboard market dropdown can switch freely
        heat = {}
        probs = {}
        for sym in list(self.active_symbols or []):
            ticks = self.fetcher.get_recent_data(sym, 150) if self.fetcher else []
            if ticks:
                heat[sym] = digit_snapshot(ticks)
                probs[sym] = probability_table(ticks, symbol=sym)
        try:
            hist_map: Dict[str, list] = {}
            for e in self.trade_log:
                if str(e.get("status") or "").lower() not in {"win", "loss"}:
                    continue
                k = f"{e.get('symbol')}|{e.get('contract_type')}"
                hist_map.setdefault(k, []).append({"profit": e.get("profit")})
            scan = scan_markets(
                list(self.active_symbols or []),
                lambda s: self.fetcher.get_recent_data(s, 120) if self.fetcher else [],
                history_by_key=hist_map,
                top_n=10,
                global_samples=self.learner.global_samples(),
            )
            self.last_scan = scan
        except Exception as e:
            logger.debug("edge scan failed: %s", e)
            scan = self.last_scan or {}

        bal = self.client.get_balance() or 0.0
        base = float(
            self.risk_manager.base_stake
            or self.risk_manager.min_stake
            or MIN_STAKE
        )
        mg = survival_probability(float(bal), base, max_levels=5)
        risk_suggest = None
        if bal and self.risk_manager.consecutive_losses >= 2:
            risk_suggest = {
                "current_risk": "High",
                "message": (
                    f"Reduce stake — consecutive losses="
                    f"{self.risk_manager.consecutive_losses}"
                ),
                "risk_plan": adaptive_risk_pct(
                    base_risk_pct=float(self.risk_manager.max_stake_pct or 1.5),
                    consecutive_losses=self.risk_manager.consecutive_losses,
                    daily_pnl=self.risk_manager.daily_pnl,
                    session_start_balance=self.risk_manager.session_start_balance,
                ),
            }
        # Rolling entropy engines per active symbol
        entropy_by_sym = {}
        try:
            from src.analytics.rolling_entropy import feed_ticks

            for sym in list(self.active_symbols or []):
                tks = self.fetcher.get_recent_data(sym, 500) if self.fetcher else []
                if tks:
                    entropy_by_sym[sym] = feed_ticks(sym, tks)
        except Exception as e:
            logger.debug("rolling entropy snapshot failed: %s", e)

        hpp_dash = {}
        try:
            from src.analytics.hpp_timeseries import get_hpp_timeseries

            ts = get_hpp_timeseries()
            if not ts.points:
                ts.capture_snapshot(reason="status")
            hpp_dash = ts.dashboard_bundle(
                [
                    "DIGITDIFF",
                    "DIGITMATCH",
                    "DIGITEVEN",
                    "DIGITODD",
                    "DIGITOVER",
                    "DIGITUNDER",
                    "CALL",
                    "PUT",
                ]
            )
        except Exception as e:
            logger.debug("hpp timeseries dashboard failed: %s", e)

        market_book = {}
        try:
            from src.strategy.market_categories import (
                category_summary,
                profiles_for_symbols,
            )

            market_book = {
                "profiles": profiles_for_symbols(list(self.active_symbols or [])),
                "categories": category_summary(),
            }
        except Exception as e:
            logger.debug("market categories failed: %s", e)

        return {
            "gate_enabled": self.analytics_gate,
            "digit_heatmaps": heat,
            "probability": probs,
            "edge_scan": scan,
            "last_filter": self.last_filter,
            "session": self.session_analytics.snapshot(),
            "martingale_safety": mg,
            "risk_suggestion": risk_suggest,
            "strategies": self.strategy_builder.list_strategies(),
            "rolling_entropy": entropy_by_sym,
            "hpp_timeseries": hpp_dash,
            "market_book": market_book,
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
