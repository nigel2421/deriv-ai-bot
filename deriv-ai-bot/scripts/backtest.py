"""
Run a digit strategy backtest on historical ticks.

Usage:
  python scripts/backtest.py
  python scripts/backtest.py --data data/historical/R_100_ticks.csv --symbol R_100
  python scripts/backtest.py --min-confidence 0.5 --no-model --export data/training/backtest_trades.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.backtest.engine import BacktestEngine, load_ticks_csv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backtest")


def resolve_data(path: str | None, symbol: str) -> Path:
    if path:
        return Path(path)
    candidates = [
        ROOT / "data" / "historical" / f"{symbol}_ticks.csv",
        ROOT / "data" / "historical" / "ticks.csv",
        ROOT / "data" / "training" / "features.csv",
    ]
    for c in candidates:
        if c.is_file() and c.stat().st_size > 50:
            return c
    raise FileNotFoundError(
        "No tick CSV found. Run: python scripts/data_collector.py"
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Backtest digit strategy")
    p.add_argument("--data", default=None, help="Ticks CSV path")
    p.add_argument("--symbol", default="R_100")
    p.add_argument("--balance", type=float, default=1000.0)
    p.add_argument("--min-confidence", type=float, default=0.55)
    p.add_argument("--duration", type=int, default=5, help="Contract length in ticks")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--payout", type=float, default=0.95, help="Win profit as fraction of stake")
    p.add_argument("--no-model", action="store_true", help="Heuristic signals only")
    p.add_argument("--max-trades", type=int, default=None)
    p.add_argument("--export", default=None, help="Export trade log CSV")
    p.add_argument(
        "--summary-json",
        default=None,
        help="Write summary metrics JSON",
    )
    args = p.parse_args()

    try:
        data_path = resolve_data(args.data, args.symbol)
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1

    logger.info("Loading %s", data_path)
    df = load_ticks_csv(str(data_path))
    logger.info("Loaded %d rows", len(df))

    engine = BacktestEngine(
        symbol=args.symbol,
        initial_balance=args.balance,
        min_confidence=args.min_confidence,
        duration_ticks=args.duration,
        win_payout=args.payout,
        warmup=args.warmup,
        step=args.step,
        use_model=not args.no_model,
        max_trades=args.max_trades,
    )
    result = engine.run(df)
    print(result.summary())

    if args.export and result.trade_log:
        out = Path(args.export)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([t.__dict__ for t in result.trade_log]).to_csv(out, index=False)
        logger.info("Exported %d trades → %s", len(result.trade_log), out)

    if args.summary_json:
        outj = Path(args.summary_json)
        outj.parent.mkdir(parents=True, exist_ok=True)
        payload = result.to_dict()
        # equity can be long — keep it
        outj.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Wrote summary → %s", outj)

    return 0


if __name__ == "__main__":
    sys.exit(main())
