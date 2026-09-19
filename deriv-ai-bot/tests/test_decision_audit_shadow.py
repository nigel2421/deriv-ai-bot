import unittest
from datetime import datetime, timezone
from pathlib import Path
from src.strategy.decision_auditor import (
    DecisionAuditor,
    DecisionTrace,
    ShadowTrade,
    ShadowTrader,
    GateAnalytics,
)
from src.strategy.experiment_engine import ExperimentEngine
from src.strategy.risk_manager import RiskManager


class TestDecisionAuditShadow(unittest.TestCase):

    def setUp(self):
        import shutil
        test_dir = Path("data/test_tmp")
        test_dir.mkdir(parents=True, exist_ok=True)

        self.auditor = DecisionAuditor(jsonl_path=test_dir / "test_decision_audit.jsonl")
        self.auditor._memory_traces.clear()

        self.shadow_trader = ShadowTrader(jsonl_path=test_dir / "test_shadow_trades.jsonl")
        self.shadow_trader.pending_shadows.clear()
        self.shadow_trader.completed_shadows.clear()

        self.experiment_engine = ExperimentEngine(storage_path=test_dir / "test_experiments.json")
        for exp in self.experiment_engine.experiments.values():
            exp.control_results.clear()
            exp.challenger_results.clear()

        self.risk_manager = RiskManager(
            max_daily_loss_pct=5.0,
            max_consecutive_losses=3,
            min_balance=10.0,
        )

    def test_decision_trace_creation_and_auditing(self):
        trace = DecisionTrace(
            symbol="R_100",
            market_regime="NORMAL",
            proposed_contract_type="DIGITEVEN",
            proposed_duration=5,
            proposed_duration_unit="t",
            original_agent_confidence=0.85,
            individual_specialist_votes=[
                {"agent": "PatternAgent", "vote": "DIGITEVEN", "confidence": 0.88, "weight": 1.1},
                {"agent": "TrendAgent", "vote": "DIGITEVEN", "confidence": 0.82, "weight": 1.2},
            ],
            agent_reputation_weights={"PatternAgent": 1.1, "TrendAgent": 1.2},
            consensus_score=0.85,
            quorum_result=True,
            htf_alignment_result=True,
            atr_volatility_value=1.0,
            adaptive_expiry_selected=5,
            boom_crash_cooldown_state=False,
            live_payout=0.95,
            estimated_probability=0.85,
            expected_value=0.61,
            portfolio_manager_decision=True,
            final_decision="EXECUTED",
        )

        self.auditor.record_trace(trace)
        self.assertEqual(len(self.auditor.traces), 1)
        self.assertEqual(self.auditor.traces[0].symbol, "R_100")

        # Test rejection funnel
        rejected_trace = DecisionTrace(
            symbol="R_50",
            market_regime="NORMAL",
            proposed_contract_type="CALL",
            proposed_duration=5,
            proposed_duration_unit="t",
            original_agent_confidence=0.80,
            consensus_score=0.80,
            quorum_result=True,
            htf_alignment_result=False,  # HTF failed
            expected_value=0.5,
            portfolio_manager_decision=True,
            final_decision="REJECTED_SHADOW",
            rejection_reason="htf_alignment",
        )
        self.auditor.record_trace(rejected_trace)

        funnel = self.auditor.rejection_funnel_counts()
        self.assertEqual(funnel["total_scanned"], 2)
        self.assertEqual(funnel["executed"], 1)
        self.assertEqual(funnel["by_gate"]["htf_alignment"], 1)

    def test_shadow_trade_spawn_and_tick_settlement(self):
        shadow_id = self.shadow_trader.spawn_shadow_trade(
            symbol="R_100",
            contract_type="DIGITEVEN",
            rejection_gate="htf_alignment",
            entry_price=123.456,  # last digit is 6 (EVEN)
            duration=5,
            duration_unit="t",
            audit_id=123,
        )

        self.assertIn(shadow_id, self.shadow_trader.shadow_trades)
        st = self.shadow_trader.shadow_trades[shadow_id]
        self.assertEqual(st.status, "pending")
        self.assertEqual(st.contract_type, "DIGITEVEN")

        # Tick 1 to 4
        for i in range(4):
            settled = self.shadow_trader.on_tick("R_100", 123.400 + i, epoch=100 + i)
            self.assertEqual(len(settled), 0)

        # Tick 5 (expiry tick) -> price 123.458 -> last digit 8 (EVEN -> WIN)
        settled = self.shadow_trader.on_tick("R_100", 123.458, epoch=105)
        self.assertEqual(len(settled), 1)
        res = settled[0]
        self.assertEqual(res["shadow_id"], shadow_id)
        self.assertEqual(res["status"], "SETTLED")
        self.assertEqual(res["outcome"], "WIN")
        self.assertGreater(res["profit"], 0.0)

    def test_gate_analytics_verdict(self):
        # Create 35 shadow trades for htf_alignment (10 wins, 25 losses -> 28.6% WR -> POSITIVE gate)
        for i in range(35):
            st = ShadowTrade(
                shadow_id=f"st_htf_{i}",
                audit_id=i,
                symbol="R_100",
                contract_type="CALL",
                rejection_gate="htf_alignment",
                entry_price=100.0,
                duration=5,
                status="SETTLED",
                outcome="WIN" if i < 10 else "LOSS",
                profit=0.87 if i < 10 else -1.0,
            )
            self.shadow_trader.completed_shadows.append(st)

        analytics = GateAnalytics(self.shadow_trader.shadow_trades)
        summary = analytics.summary()
        htf_info = summary["gates"]["htf_alignment"]

        self.assertEqual(htf_info["shadow_samples"], 35)
        self.assertLess(htf_info["shadow_win_rate"], 45.0)
        self.assertEqual(htf_info["verdict"], "POSITIVE")

    def test_experiment_engine_control_vs_challenger(self):
        # Evaluate opportunity where control rejects (ev=0.05 < 0.08), challenger passes (min_ev=0.05)
        self.experiment_engine.evaluate_opportunity(
            symbol="R_25",
            contract_type="CALL",
            control_passed=False,
            control_ev=0.05,
            control_quorum=2,
            shadow_trader=self.shadow_trader,
            entry_price=50.0,
            duration=5,
            duration_unit="t",
        )

        summary = self.experiment_engine.summary()
        self.assertGreater(len(summary["active_experiments"]), 0)
        exp0 = summary["active_experiments"][0]
        self.assertEqual(exp0["name"], "EV_Threshold_Exp")
        self.assertEqual(exp0["challenger"]["samples"], 1)

    def test_safety_portfolio_manager_veto_cannot_be_overridden(self):
        # Set risk manager to paused/halted state
        self.risk_manager.consecutive_losses = 10
        self.risk_manager.pause(minutes=60, reason="daily_limit")

        decision = self.risk_manager.can_trade(100.0)
        self.assertFalse(bool(decision))
        self.assertIn("paused", decision.reason.lower())

        # Ensure shadow trader trade creation does NOT execute live trade
        shadow_id = self.shadow_trader.spawn_shadow_trade(
            symbol="R_100",
            contract_type="PUT",
            rejection_gate="portfolio_manager",
            entry_price=100.0,
            duration=5,
            duration_unit="t",
        )
        self.assertIsNotNone(shadow_id)
        self.assertEqual(self.shadow_trader.shadow_trades[shadow_id].rejection_gate, "portfolio_manager")


if __name__ == "__main__":
    unittest.main()
