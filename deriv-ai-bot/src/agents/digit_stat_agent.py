"""
Digit Statistical Specialist Agent (DigitStatAgent)

2nd Dedicated Intelligence Agent for Deriv Last-Digit Contracts.
Complements PatternAgent by applying formal statistical models:
  1. 1-Step Markov Transition Probabilities P(Digit_{t+1} = j | Digit_t = i)
  2. Hiatus & Entropy Tracker (Cold-Digit Identification for high-win DIGITDIFF setups)
  3. Chi-Square Parity & Barrier Distribution Skew
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from src.agents.base_agent import BaseAgent, AgentSignal
from src.strategy.digit_contracts import (
    extract_last_digit,
    last_digits_from_ticks,
)
from src.strategy.contract_types import is_digit_contract

logger = logging.getLogger(__name__)


def compute_markov_row(digits: List[int], current_digit: int) -> Dict[int, float]:
    """
    Compute empirical 1-step transition probability row for the current digit.
    P(next_digit = j | current_digit) across recent digit history.
    """
    if len(digits) < 10:
        # Uniform prior if insufficient history
        return {d: 0.10 for d in range(10)}

    transitions = {d: 0 for d in range(10)}
    total_occurrences = 0

    for i in range(len(digits) - 1):
        if digits[i] == current_digit:
            next_digit = digits[i + 1]
            transitions[next_digit] += 1
            total_occurrences += 1

    if total_occurrences == 0:
        return {d: 0.10 for d in range(10)}

    # Laplace smoothing (add-1) to avoid zero probabilities
    smoothed_total = total_occurrences + 10
    return {d: (transitions[d] + 1) / float(smoothed_total) for d in range(10)}


def compute_digit_hiatus(digits: List[int]) -> Dict[int, int]:
    """
    Compute ticks elapsed since each digit (0-9) was last observed.
    """
    hiatus = {d: len(digits) for d in range(10)}
    for index, d in enumerate(reversed(digits)):
        if hiatus[d] == len(digits):
            hiatus[d] = index
    return hiatus


class DigitStatAgent(BaseAgent):
    """
    Statistical Digit Intelligence Agent:
    - Analyzes Markov transition matrices on last-digit streams.
    - Tracks hiatus count for cold digits (high win-rate DIGITDIFF entries).
    - Measures empirical distribution skew across digit barriers.
    """

    def __init__(self, name: str = "DigitStatAgent", enabled: bool = True):
        super().__init__(name=name, enabled=enabled)

    async def evaluate(self, context: Dict[str, Any]) -> List[AgentSignal]:
        if not self.enabled:
            return []

        symbol = context.get("symbol", "")
        ticks = context.get("ticks")
        allowed_types = context.get("allowed_types", [])

        if not symbol or not ticks or len(ticks) < 20:
            return []

        digit_allowed = [t for t in allowed_types if is_digit_contract(t)]
        if not digit_allowed and "ALL" not in allowed_types:
            digit_allowed = ["DIGITOVER", "DIGITUNDER", "DIGITDIFF", "DIGITEVEN", "DIGITODD"]

        digits = last_digits_from_ticks(ticks, n=100)
        if len(digits) < 15:
            return []

        current_digit = digits[-1]
        signals: List[AgentSignal] = []

        # 1. Hiatus Model for DIGITDIFF (Cold Digit Selection)
        hiatus = compute_digit_hiatus(digits)
        cold_digit: Optional[int] = None
        max_hiatus = 0
        for d, h_count in hiatus.items():
            if h_count > max_hiatus:
                max_hiatus = h_count
                cold_digit = d

        # Standard average hiatus is ~10 ticks. A hiatus >= 22 represents a statistically cold digit.
        if (
            cold_digit is not None
            and max_hiatus >= 22
            and ("DIGITDIFF" in digit_allowed or not digit_allowed)
        ):
            # Compute Markov probability of cold digit recurring next
            markov_probs = compute_markov_row(digits, current_digit)
            p_cold = markov_probs.get(cold_digit, 0.10)
            p_win_diff = round(1.0 - p_cold, 4)

            if p_win_diff >= 0.88:
                signals.append(
                    AgentSignal(
                        agent_name=self.name,
                        symbol=symbol,
                        contract_type="DIGITDIFF",
                        confidence=p_win_diff,
                        raw_confidence=p_win_diff,
                        barrier=cold_digit,
                        weight=1.1,
                        duration=5,
                        duration_unit="t",
                        family="digits",
                        horizon="tick",
                        rationale=f"DigitStat hiatus cold-digit: DIGITDIFF barrier={cold_digit} (hiatus={max_hiatus} ticks, P(win)={p_win_diff:.2f})",
                        metadata={
                            "cold_digit": cold_digit,
                            "hiatus_count": max_hiatus,
                            "markov_p_cold": p_cold,
                        },
                    )
                )

        # 2. Markov Matrix & Skew Distribution Model for OVER / UNDER / EVEN / ODD
        markov_row = compute_markov_row(digits, current_digit)

        # Expected probabilities via Markov 1-step matrix
        p_even_markov = sum(markov_row[d] for d in (0, 2, 4, 6, 8))
        p_odd_markov = sum(markov_row[d] for d in (1, 3, 5, 7, 9))

        # Recent 30-digit empirical parity
        recent_30 = digits[-30:]
        empirical_even = sum(1 for d in recent_30 if d % 2 == 0) / float(len(recent_30))
        empirical_odd = 1.0 - empirical_even

        # DIGITEVEN Signal: Both Markov and empirical window confirm even skew >= 0.62
        if (
            ("DIGITEVEN" in digit_allowed or not digit_allowed)
            and p_even_markov >= 0.60
            and empirical_even >= 0.60
        ):
            combined_conf = round((p_even_markov * 0.5) + (empirical_even * 0.5), 4)
            if combined_conf >= 0.65:
                signals.append(
                    AgentSignal(
                        agent_name=self.name,
                        symbol=symbol,
                        contract_type="DIGITEVEN",
                        confidence=combined_conf,
                        raw_confidence=combined_conf,
                        weight=1.0,
                        duration=5,
                        duration_unit="t",
                        family="digits",
                        horizon="tick",
                        rationale=f"DigitStat Markov parity: DIGITEVEN (Markov={p_even_markov:.2f}, window={empirical_even:.2f})",
                        metadata={
                            "markov_p_even": p_even_markov,
                            "empirical_even": empirical_even,
                        },
                    )
                )
        elif (
            ("DIGITODD" in digit_allowed or not digit_allowed)
            and p_odd_markov >= 0.60
            and empirical_odd >= 0.60
        ):
            combined_conf = round((p_odd_markov * 0.5) + (empirical_odd * 0.5), 4)
            if combined_conf >= 0.65:
                signals.append(
                    AgentSignal(
                        agent_name=self.name,
                        symbol=symbol,
                        contract_type="DIGITODD",
                        confidence=combined_conf,
                        raw_confidence=combined_conf,
                        weight=1.0,
                        duration=5,
                        duration_unit="t",
                        family="digits",
                        horizon="tick",
                        rationale=f"DigitStat Markov parity: DIGITODD (Markov={p_odd_markov:.2f}, window={empirical_odd:.2f})",
                        metadata={
                            "markov_p_odd": p_odd_markov,
                            "empirical_odd": empirical_odd,
                        },
                    )
                )

        # 3. DIGITOVER / DIGITUNDER Markov Skew Analysis
        # Test OVER barrier 4 (win if digit > 4 -> 5,6,7,8,9)
        p_over_4 = sum(markov_row[d] for d in (5, 6, 7, 8, 9))
        empirical_over_4 = sum(1 for d in recent_30 if d > 4) / float(len(recent_30))

        if (
            ("DIGITOVER" in digit_allowed or not digit_allowed)
            and p_over_4 >= 0.62
            and empirical_over_4 >= 0.60
        ):
            conf_over = round((p_over_4 * 0.6) + (empirical_over_4 * 0.4), 4)
            if conf_over >= 0.66:
                signals.append(
                    AgentSignal(
                        agent_name=self.name,
                        symbol=symbol,
                        contract_type="DIGITOVER",
                        confidence=conf_over,
                        raw_confidence=conf_over,
                        barrier=4,
                        weight=1.0,
                        duration=5,
                        duration_unit="t",
                        family="digits",
                        horizon="tick",
                        rationale=f"DigitStat Markov distribution: DIGITOVER barrier=4 (Markov={p_over_4:.2f}, window={empirical_over_4:.2f})",
                        metadata={
                            "p_over_4": p_over_4,
                            "empirical_over_4": empirical_over_4,
                        },
                    )
                )

        # Test UNDER barrier 5 (win if digit < 5 -> 0,1,2,3,4)
        p_under_5 = sum(markov_row[d] for d in (0, 1, 2, 3, 4))
        empirical_under_5 = sum(1 for d in recent_30 if d < 5) / float(len(recent_30))

        if (
            ("DIGITUNDER" in digit_allowed or not digit_allowed)
            and p_under_5 >= 0.62
            and empirical_under_5 >= 0.60
        ):
            conf_under = round((p_under_5 * 0.6) + (empirical_under_5 * 0.4), 4)
            if conf_under >= 0.66:
                signals.append(
                    AgentSignal(
                        agent_name=self.name,
                        symbol=symbol,
                        contract_type="DIGITUNDER",
                        confidence=conf_under,
                        raw_confidence=conf_under,
                        barrier=5,
                        weight=1.0,
                        duration=5,
                        duration_unit="t",
                        family="digits",
                        horizon="tick",
                        rationale=f"DigitStat Markov distribution: DIGITUNDER barrier=5 (Markov={p_under_5:.2f}, window={empirical_under_5:.2f})",
                        metadata={
                            "p_under_5": p_under_5,
                            "empirical_under_5": empirical_under_5,
                        },
                    )
                )

        return signals
