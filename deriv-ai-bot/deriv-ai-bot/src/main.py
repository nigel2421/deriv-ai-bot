import asyncio
import argparse
import logging
from config.settings import MODE, SYMBOLS
from src.utils.logger import setup_logger
from src.api.auth_manager import AuthManager
from src.api.deriv_client import DerivClient
from src.api.price_fetcher import PriceFetcher

logger = setup_logger()

async def main(mode: str):
    logger.info(f"Starting Deriv AI Bot in {mode} mode")
    
    # Phase 1: API Connection & Tick Subscription
    auth = AuthManager()
    app_id, token = auth.get_credentials()
    
    if not token:
        logger.error("No API token provided. Set DERIV_API_TOKEN in .env")
        return
    
    client = DerivClient(app_id, token, mode)
    await client.connect()
    
    fetcher = PriceFetcher(client)
    fetcher.subscribe_symbols(SYMBOLS)
    
    logger.info("Phase 1 complete: Connected and subscribed to ticks.")
    logger.info("Monitoring ticks... (Ctrl+C to stop)")
    
    try:
        while True:
            await asyncio.sleep(30)
            # Log sample data for verification
            for symbol in SYMBOLS:
                ticks = fetcher.get_recent_data(symbol, 5)
                if ticks:
                    logger.info(f"Latest {symbol} price: {ticks[-1].get('quote') if ticks else 'N/A'}")
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deriv AI Trading Bot")
    parser.add_argument('--mode', default=MODE, choices=['demo', 'real'])
    args = parser.parse_args()
    asyncio.run(main(args.mode))
