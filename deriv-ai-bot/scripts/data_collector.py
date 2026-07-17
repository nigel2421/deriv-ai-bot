import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.statement_fetcher import StatementFetcher
# Assume client passed or global

async def collect_data(client):
    fetcher = StatementFetcher(client)
    data = fetcher.fetch_recent_statements()
    fetcher.save_to_csv(data)
    print("Data collection complete. Ready for retraining.")
