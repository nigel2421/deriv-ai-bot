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
    
    # Phase 5: Orchestrator for Multi-Market + Full Logic
    from src.orchestrator import TradingOrchestrator
    orchestrator = TradingOrchestrator(client, mode)
    
    logger.info("Phase 5: Multi-market scanning + trade orchestration active.")
    logger.info(f"Scanning symbols: {SYMBOLS}. Running trading cycles...")
    
    try:
        while True:
            await orchestrator.execute_trade_cycle()
            await asyncio.sleep(60)  # Trade cycle interval
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deriv AI Trading Bot")
    parser.add_argument('--mode', default=MODE, choices=['demo', 'real'])
    args = parser.parse_args()
    asyncio.run(main(args.mode))
