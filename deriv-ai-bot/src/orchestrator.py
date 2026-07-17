import asyncio
import logging
from typing import Any, Dict, Optional

from config.settings import (
    EXECUTE_TRADES,
    MAX_OPEN_TRADES,
    MAX_STAKE,
    MAX_STAKE_PCT,
    MIN_BALANCE,
    MIN_STAKE,
    SYMBOLS,
    TRADE_DURATION_TICKS,
)
from src.api.deriv_client import DerivClient
from src.api.price_fetcher import PriceFetcher
from src.api.trade_executor import TradeExecutor
from src.api.trade_monitor import TradeMonitor
from src.strategy.xml_parser import XMLStrategyParser
from src.strategy.strategy_engine import StrategyEngine
from src.strategy.risk_manager import RiskManager
from src.strategy.trade_selector import TradeSelector
from src.strategy.signal_generator import SignalGenerator
from src.ai.predictor import Predictor
from src.utils.telegram_bot import TelegramBot

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
        self.signal_gen = SignalGenerator()
        self.predictor = Predictor()
        self.telegram = TelegramBot()
        self.telegram.set_status_provider(self.risk_status)
        self.telegram.set_stats_provider(self.stats_snapshot)
        self.active_symbols = SYMBOLS
        self.max_open_trades = MAX_OPEN_TRADES

        self.execute_trades = bool(EXECUTE_TRADES)
        if mode == "real" and not EXECUTE_TRADES:
            self.execute_trades = False

        self.monitor = TradeMonitor(client, on_close=self._on_contract_closed)
        self.closed_trades: list = []

        bal = client.get_balance()
        if bal is not None:
            self.risk_manager.set_session_balance(bal)

        logger.info(
            "Orchestrator ready mode=%s execute_trades=%s symbols=%s "
            "strategies=%s min_balance=%s max_open=%s telegram=%s",
            mode,
            self.execute_trades,
            self.active_symbols,
            {s: r.stype for s, r in self.strategy_engine.runtimes.items()},
            min_balance,
            MAX_OPEN_TRADES,
            self.telegram.is_configured(),
        )

    def open_trade_count(self) -> int:
        mon = set(self.monitor.open_contracts.keys())
        exe = set(self.executor.open_trades.keys())
        return len(mon | exe)

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
        """Predict per symbol, apply strategy rules, select best trade."""
        signals = []
        min_conf = self.parser.config.get("global", {}).get("min_confidence", 0.75)

        for symbol in self.active_symbols:
            ticks = self.fetcher.get_recent_data(symbol, 100)
            if not ticks:
                logger.debug("No ticks yet for %s", symbol)
                continue

            runtime = self.strategy_engine.get(symbol)
            if not runtime.is_tradeable():
                logger.info("Skip %s: strategy inactive (e.g. martingale max steps)", symbol)
                continue

            pred = self.predictor.predict(ticks)
            # Attach ticks so signal gen can fall back to last-digit extraction
            pred = {**pred, "recent_ticks": ticks}
            confidence = float(pred.get("confidence", 0.5))

            allowed = runtime.allowed_types or None
            signal_type, signal_barrier, conf = self.signal_gen.generate_signal(
                pred, confidence, min_conf, allowed_types=allowed
            )
            if not signal_type and not runtime.zuno:
                continue
            if not signal_type and runtime.zuno:
                if confidence < min_conf:
                    continue
                conf = confidence

            intent = self.strategy_engine.apply_signal(
                symbol=symbol,
                signal_type=signal_type,
                signal_barrier=signal_barrier,
                confidence=conf or confidence,
            )
            if intent:
                signals.append(intent)
                logger.info(
                    "Signal+strategy %s: type=%s stake=%.2f barrier=%s conf=%.2f strat=%s",
                    symbol,
                    intent["contract_type"],
                    intent["stake"],
                    intent.get("barrier"),
                    intent["confidence"],
                    intent["strategy"],
                )

        return self.selector.select_best_trade(signals)

    async def execute_trade_cycle(self) -> Optional[Dict[str, Any]]:
        """Full cycle: risk → scan/strategy → stake clamp → proposal → buy → monitor."""
        # Telegram /pause master switch
        if not self.telegram.trading_enabled:
            logger.info("Trading paused via Telegram (/pause). Skipping cycle.")
            return None

        balance = await self._live_balance(refresh=False)
        open_count = self.open_trade_count()

        decision = self.risk_manager.can_trade(balance, open_trades=open_count)
        if not decision:
            logger.warning(
                "Risk block: %s | balance=%s open=%s/%s",
                decision.reason,
                balance,
                open_count,
                self.max_open_trades,
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

        logger.info(
            "Trade candidate: %s %s stake=%s barrier=%s conf=%.2f "
            "strategy=%s execute=%s balance=%.2f open=%s",
            best["symbol"],
            best["contract_type"],
            stake,
            best.get("barrier"),
            best.get("confidence", 0),
            best.get("strategy"),
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
            execute=self.execute_trades,
        )

        if not result:
            err = self.executor.last_error or "unknown"
            logger.error("Trade path failed: %s", err)
            await self.telegram.send_notification(f"❌ Trade failed: {err}")
            return None

        if result.get("buy_failed"):
            err = result.get("error") or self.executor.last_error or "buy failed"
            logger.error("Buy failed after proposal: %s", err)
            await self._live_balance(refresh=True)
            await self.telegram.send_notification(f"❌ Buy failed: {err}")
            return result

        proposal = result.get("proposal") or {}
        buy = result.get("buy") or {}

        if buy.get("balance_after") is not None:
            try:
                self.client.set_balance(float(buy["balance_after"]))
                self.risk_manager.update_balance(float(buy["balance_after"]))
            except (TypeError, ValueError):
                pass

        await self.telegram.send_notification(
            f"{'🚀' if result.get('executed') else '📄'} "
            f"{best['symbol']} {best['contract_type']} "
            f"strat={best.get('strategy')} stake={stake} "
            f"conf={best['confidence']:.1%} ask={proposal.get('ask_price')} "
            f"bal={self.client.get_balance()} "
            f"{'BOUGHT' if result.get('executed') else 'PROPOSAL ONLY'}"
        )

        contract_id = result.get("contract_id")
        if result.get("executed") and contract_id is not None:
            meta = {
                **best,
                "stake": stake,
                "proposal_id": proposal.get("id"),
            }
            await self.monitor.watch(contract_id, meta=meta)
            logger.info(
                "Trade opened contract_id=%s strategy=%s open_now=%s",
                contract_id,
                best.get("strategy"),
                self.open_trade_count(),
            )
        elif not self.execute_trades:
            logger.info("Proposal-only cycle complete (EXECUTE_TRADES=false)")

        return result

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

        self.executor.mark_closed(contract_id)
        self.risk_manager.record_trade_result(profit)

        # Advance Martingale / Zuno for this symbol
        if symbol:
            self.strategy_engine.on_trade_result(symbol, is_win=is_win, profit=profit)

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
        msg = (
            f"🏁 {status} {symbol} {meta.get('contract_type')} "
            f"P&L={profit} contract={contract_id} "
            f"daily_pnl={snap['daily_pnl']:.2f} "
            f"streak={snap['consecutive_losses']} "
            f"next_stake={strat_snap.get('next_stake')} "
            f"next_type={strat_snap.get('next_type')}"
        )
        logger.info(msg)
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
        }
