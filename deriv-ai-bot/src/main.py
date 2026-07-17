import argparse
import asyncio
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as `python src/main.py`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import MODE
from src.utils.logger import setup_logger
from src.bot_runtime import run_bot_forever

logger = setup_logger()


async def main(mode: str):
    await run_bot_forever(mode, cycle_seconds=60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deriv AI Trading Bot")
    parser.add_argument("--mode", default=MODE, choices=["demo", "real"])
    args = parser.parse_args()
    try:
        asyncio.run(main(args.mode))
    except KeyboardInterrupt:
        logger.info("Interrupted")
