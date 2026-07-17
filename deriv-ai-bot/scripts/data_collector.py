"""
Collect market data (ticks) and optional account statements from Deriv.

Usage (from project root):
  python scripts/data_collector.py
  python scripts/data_collector.py --symbols R_100,R_75 --count 1000
  python scripts/data_collector.py --statements 50
  python scripts/data_collector.py --no-save-combined
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from config.settings import DERIV_APP_ID, MODE, SYMBOLS
from src.api.auth_manager import AuthManager
from src.api.deriv_client import DerivClient
from src.api.price_fetcher import PriceFetcher
from src.api.statement_fetcher import StatementFetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("data_collector")

HIST_DIR = ROOT / "data" / "historical"
TRAIN_DIR = ROOT / "data" / "training"


async def collect(
    symbols: list[str],
    count: int,
    *,
    statements: int = 0,
    save_combined: bool = True,
) -> int:
    auth = AuthManager()
    app_id, token = auth.get_credentials()
    if not token or str(token).startswith("your_"):
        logger.error("Set DERIV_API_TOKEN in .env")
        return 1

    client = DerivClient(str(app_id or DERIV_APP_ID), str(token), MODE or "demo")
    ok = await client.connect()
    if not ok:
        logger.error("Authorize failed")
        await client.close()
        return 1

    HIST_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    fetcher = PriceFetcher(client)
    sizes = await fetcher.bootstrap_history(
        symbols,
        count=count,
        min_required=min(50, count),
        save_dir=str(HIST_DIR),
    )
    logger.info("Tick buffers: %s", sizes)

    # Also write/update a combined ticks.csv for training convenience
    if save_combined:
        frames = []
        for sym in symbols:
            ticks = client.get_latest_ticks(sym, count)
            if not ticks:
                p = HIST_DIR / f"{sym}_ticks.csv"
                if p.is_file():
                    frames.append(pd.read_csv(p))
                continue
            df = pd.DataFrame(ticks)
            frames.append(df)
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            if "epoch" in combined.columns:
                combined = combined.sort_values("epoch")
            out = HIST_DIR / "ticks.csv"
            # Prefer symbol-pure series for model training: use first symbol if multi
            # but still save full combined as ticks_all.csv
            all_out = HIST_DIR / "ticks_all.csv"
            combined.to_csv(all_out, index=False)
            # Primary training file: densest single-symbol series
            best_sym = max(symbols, key=lambda s: sizes.get(s, 0))
            best_ticks = client.get_latest_ticks(best_sym, count)
            if best_ticks:
                pd.DataFrame(best_ticks).to_csv(out, index=False)
                logger.info(
                    "Wrote %s (%d rows from %s) and %s",
                    out,
                    len(best_ticks),
                    best_sym,
                    all_out,
                )

    if statements > 0:
        sf = StatementFetcher(client)
        rows = await sf.fetch_recent_statements(limit=statements)
        sf.save_to_csv(rows, str(TRAIN_DIR / "statements.csv"))

    await client.close()
    logger.info("Data collection complete.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Collect Deriv ticks / statements")
    p.add_argument(
        "--symbols",
        default=",".join(SYMBOLS),
        help="Comma-separated symbols (default from .env SYMBOLS)",
    )
    p.add_argument("--count", type=int, default=1000, help="Ticks per symbol")
    p.add_argument(
        "--statements",
        type=int,
        default=0,
        help="Also fetch N account statement rows (0=skip)",
    )
    p.add_argument(
        "--no-save-combined",
        action="store_true",
        help="Do not write data/historical/ticks.csv",
    )
    args = p.parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    return asyncio.run(
        collect(
            symbols,
            args.count,
            statements=args.statements,
            save_combined=not args.no_save_combined,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
