"""
Lightweight digit-strategy backtest on last N ticks (10k / 100k style).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from src.strategy.digit_contracts import (
    extract_last_digit,
    last_digits_from_ticks,
    would_win,
)


def backtest_digit_rule(
    ticks: Sequence[Dict[str, Any]],
    *,
    contract_type: str,
    barrier: Optional[int] = None,
    stake: float = 1.0,
    payout_ratio: float = 0.95,  # net profit on win as fraction of stake (approx)
    lookback_signal: int = 20,
    max_ticks: int = 10000,
    signal_fn: Optional[Callable[[List[int]], bool]] = None,
) -> Dict[str, Any]:
    """
    Walk forward: at each tick, if signal_fn(past digits) True, 'trade' next digit.
    Default signal: always true after warmup (baseline).
    """
    digits = last_digits_from_ticks(ticks, n=max_ticks + lookback_signal + 5)
    if len(digits) < lookback_signal + 5:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "avg_streak_loss": 0.0,
            "total_pnl": 0.0,
        }

    wins = losses = 0
    gp = gl = 0.0
    pnl = 0.0
    peak = 0.0
    max_dd = 0.0
    loss_streak = 0
    loss_streaks: List[int] = []
    equity_curve = []

    def default_signal(past: List[int]) -> bool:
        return len(past) >= lookback_signal

    sig = signal_fn or default_signal
    ct = str(contract_type).upper()

    for i in range(lookback_signal, len(digits) - 1):
        past = digits[:i]
        if not sig(past):
            continue
        outcome = digits[i]  # settle on this digit (simple model)
        try:
            won = would_win(ct, barrier, outcome)
        except Exception:
            # fallback: even/odd
            if ct == "DIGITEVEN":
                won = outcome % 2 == 0
            elif ct == "DIGITODD":
                won = outcome % 2 == 1
            else:
                won = False

        if won:
            profit = stake * payout_ratio
            wins += 1
            gp += profit
            if loss_streak:
                loss_streaks.append(loss_streak)
            loss_streak = 0
        else:
            profit = -stake
            losses += 1
            gl += stake
            loss_streak += 1
        pnl += profit
        peak = max(peak, pnl)
        max_dd = max(max_dd, peak - pnl)
        equity_curve.append(pnl)

    n = wins + losses
    wr = wins / n if n else 0.0
    pf = gp / gl if gl > 1e-9 else (99.0 if gp > 0 else 0.0)
    avg_ls = sum(loss_streaks) / len(loss_streaks) if loss_streaks else float(loss_streak)
    return {
        "trades": n,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wr, 4),
        "profit_factor": round(min(pf, 99.0), 3),
        "max_drawdown": round(max_dd, 2),
        "avg_streak_loss": round(avg_ls, 2),
        "total_pnl": round(pnl, 2),
        "contract_type": ct,
        "barrier": barrier,
        "max_ticks": max_ticks,
    }
