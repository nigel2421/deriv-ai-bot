"""
Monte Carlo robustness check on a fixed sequence of trade outcomes
(or synthetic edge), replaying Martingale path with shuffled outcomes.

Usage:
  # After a backtest export:
  python scripts/monte_carlo_backtest.py --trades data/training/backtest_trades.csv

  # Or generate outcomes from a ticks backtest first (embedded):
  python scripts/monte_carlo_backtest.py --data data/historical/ticks.csv --symbol R_100

  python scripts/monte_carlo_backtest.py --sims 2000 --edge 0.02
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine, load_ticks_csv
from src.strategy.martingale import MartingaleStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("monte_carlo")


def outcomes_from_trade_log(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if "is_win" not in df.columns and "profit" in df.columns:
        return (df["profit"].astype(float) > 0).astype(int).values
    return df["is_win"].astype(int).values


def outcomes_from_backtest(
    data: Path, symbol: str, min_confidence: float, use_model: bool
) -> np.ndarray:
    df = load_ticks_csv(str(data))
    eng = BacktestEngine(
        symbol=symbol,
        min_confidence=min_confidence,
        use_model=use_model,
    )
    result = eng.run(df)
    if not result.trade_log:
        return np.array([], dtype=int)
    return np.array([1 if t.is_win else 0 for t in result.trade_log], dtype=int)


def run_mc(
    outcomes: np.ndarray,
    *,
    num_simulations: int = 1000,
    initial_balance: float = 1000.0,
    base_stake: float = 1.0,
    max_steps: int = 6,
    win_payout: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Shuffle the historical win/loss sequence many times and replay Martingale.
    Preserves empirical win rate while testing path dependency.
    """
    if len(outcomes) == 0:
        raise ValueError("No outcomes to simulate")

    rng = np.random.default_rng(seed)
    rows = []
    n = len(outcomes)
    base_wr = float(outcomes.mean())

    for sim in range(num_simulations):
        order = rng.permutation(outcomes)
        mg = MartingaleStrategy(base_stake=base_stake, max_steps=max_steps)
        balance = initial_balance
        peak = balance
        max_dd = 0.0
        wins = 0
        ruined = False
        for o in order:
            stake = mg.peek_stake()
            if stake <= 0 or stake > balance:
                ruined = balance < base_stake
                break
            is_win = bool(o)
            if is_win:
                balance += stake * win_payout
                wins += 1
            else:
                balance -= stake
            mg.on_result(is_win)
            peak = max(peak, balance)
            max_dd = max(max_dd, peak - balance)
            if balance < base_stake * 0.5:
                ruined = True
                break
        rows.append(
            {
                "final_balance": balance,
                "win_rate": wins / n,
                "max_drawdown": max_dd,
                "ruined": ruined,
                "return_pct": (balance - initial_balance) / initial_balance,
            }
        )

    df = pd.DataFrame(rows)
    logger.info(
        "Monte Carlo %d sims | empirical WR=%.1f%% | "
        "mean final=$%.2f median=$%.2f p05=$%.2f ruin=%.1f%%",
        num_simulations,
        base_wr * 100,
        df["final_balance"].mean(),
        df["final_balance"].median(),
        df["final_balance"].quantile(0.05),
        df["ruined"].mean() * 100,
    )
    print(df.describe())
    return df


def synthetic_outcomes(n: int, edge: float, seed: int = 0) -> np.ndarray:
    """edge: P(win) = 0.5 + edge (e.g. 0.02 → 52%)."""
    rng = np.random.default_rng(seed)
    p = 0.5 + edge
    return rng.random(n) < p


def main() -> int:
    p = argparse.ArgumentParser(description="Monte Carlo Martingale path test")
    p.add_argument("--trades", default=None, help="CSV with is_win column from backtest")
    p.add_argument("--data", default=None, help="Ticks CSV to backtest first")
    p.add_argument("--symbol", default="R_100")
    p.add_argument("--sims", type=int, default=1000)
    p.add_argument("--balance", type=float, default=1000.0)
    p.add_argument("--base-stake", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=6)
    p.add_argument("--payout", type=float, default=0.95)
    p.add_argument("--edge", type=float, default=None, help="Synthetic edge if no data")
    p.add_argument("--n-outcomes", type=int, default=200)
    p.add_argument("--min-confidence", type=float, default=0.55)
    p.add_argument("--no-model", action="store_true")
    p.add_argument("--export", default=None, help="Save MC results CSV")
    args = p.parse_args()

    outcomes: np.ndarray
    if args.trades:
        outcomes = outcomes_from_trade_log(Path(args.trades))
        logger.info("Loaded %d outcomes from %s (WR=%.1f%%)",
                    len(outcomes), args.trades, outcomes.mean() * 100)
    elif args.data:
        outcomes = outcomes_from_backtest(
            Path(args.data),
            args.symbol,
            args.min_confidence,
            use_model=not args.no_model,
        )
        logger.info(
            "Backtest produced %d outcomes (WR=%.1f%%)",
            len(outcomes),
            outcomes.mean() * 100 if len(outcomes) else 0,
        )
    elif args.edge is not None:
        outcomes = synthetic_outcomes(args.n_outcomes, args.edge).astype(int)
        logger.info("Synthetic outcomes n=%d edge=%.3f WR=%.1f%%",
                    len(outcomes), args.edge, outcomes.mean() * 100)
    else:
        # Try default tick file
        default = ROOT / "data" / "historical" / "ticks.csv"
        if default.is_file():
            outcomes = outcomes_from_backtest(
                default, args.symbol, args.min_confidence, use_model=not args.no_model
            )
        else:
            logger.error(
                "Provide --trades, --data, or --edge. "
                "Or run data_collector + backtest first."
            )
            return 1

    if len(outcomes) == 0:
        logger.error("No outcomes available.")
        return 1

    df = run_mc(
        outcomes,
        num_simulations=args.sims,
        initial_balance=args.balance,
        base_stake=args.base_stake,
        max_steps=args.max_steps,
        win_payout=args.payout,
    )
    if args.export:
        out = Path(args.export)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info("Saved MC results → %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
