import asyncio
import json
import logging
import os
import time
from typing import Dict, Any, Optional, List
import websocket
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class DerivClient:
    """
    WebSocket client for Deriv.com API with reconnection and rate limiting.
    """
    def __init__(self, app_id: str, token: str, mode: str = "demo"):
        self.app_id = app_id
        self.token = token
        self.mode = mode
        self.ws = None
        self.connected = False
        self.subscriptions = {}
        self.tick_buffers: Dict[str, List[Dict]] = {}
        self.max_buffer_size = 1000  # Last N ticks per symbol
        self.last_request_time = 0
        self.rate_limit = 0.1  # 10 requests per second

    async def connect(self):
        """Establish WebSocket connection with auto-reconnect."""
        url = f"wss://ws.binaryws.com/websockets/v3?app_id={self.app_id}"
        try:
            self.ws = websocket.WebSocketApp(
                url,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open
            )
            # Run in thread or use asyncio integration
            asyncio.create_task(self.run_websocket())
            logger.info("Connecting to Deriv WebSocket...")
            await asyncio.sleep(2)  # Wait for connection
        except Exception as e:
            logger.error(f"Connection failed: {e}")

    def run_websocket(self):
        """Run the WebSocket in a separate thread."""
        self.ws.run_forever()

    def on_open(self, ws):
        logger.info("WebSocket connection opened.")
        self.connected = True
        # Authorize
        auth_msg = {"authorize": self.token}
        ws.send(json.dumps(auth_msg))

    def on_message(self, ws, message):
        data = json.loads(message)
        msg_type = data.get('msg_type')
        
        if msg_type == 'authorize':
            logger.info("Successfully authorized with Deriv API.")
            if data.get('error'):
                logger.error(f"Auth error: {data['error']}")
        elif msg_type == 'tick':
            self.handle_tick(data)
        elif msg_type == 'proposal':
            logger.info(f"Proposal received: {data}")
        elif msg_type == 'buy':
            logger.info(f"Buy response: {data}")
        # Handle other message types...

    def handle_tick(self, data: Dict):
        """Process incoming tick data and buffer it."""
        symbol = data.get('tick', {}).get('symbol')
        if symbol:
            tick = data['tick']
            if symbol not in self.tick_buffers:
                self.tick_buffers[symbol] = []
            self.tick_buffers[symbol].append(tick)
            # Keep buffer size limited
            if len(self.tick_buffers[symbol]) > self.max_buffer_size:
                self.tick_buffers[symbol].pop(0)
            logger.debug(f"Tick received for {symbol}: {tick.get('quote')}")

    def on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        logger.warning(f"WebSocket closed: {close_status_code} - {close_msg}")
        self.connected = False
        # Exponential backoff reconnect
        asyncio.create_task(self.reconnect())

    async def reconnect(self, retries: int = 5):
        """Reconnect with exponential backoff."""
        for attempt in range(retries):
            wait = (2 ** attempt) * 5
            logger.info(f"Reconnecting in {wait} seconds (attempt {attempt+1}/{retries})...")
            await asyncio.sleep(wait)
            await self.connect()
            if self.connected:
                return
        logger.error("Max reconnection attempts reached.")

    def subscribe_ticks(self, symbols: List[str]):
        """Subscribe to real-time ticks for multiple symbols."""
        for symbol in symbols:
            if symbol not in self.subscriptions:
                msg = {"ticks": symbol}
                self.send_message(msg)
                self.subscriptions[symbol] = True
                self.tick_buffers[symbol] = []
                logger.info(f"Subscribed to ticks for {symbol}")

    def send_message(self, msg: Dict):
        """Send message with rate limiting."""
        if not self.ws or not self.connected:
            logger.warning("Cannot send message: not connected")
            return
        
        current_time = time.time()
        if current_time - self.last_request_time < self.rate_limit:
            time.sleep(self.rate_limit)
        
        try:
            self.ws.send(json.dumps(msg))
            self.last_request_time = time.time()
        except Exception as e:
            logger.error(f"Failed to send message: {e}")

    def get_latest_ticks(self, symbol: str, count: int = 50) -> List[Dict]:
        """Get recent ticks for a symbol."""
        buffer = self.tick_buffers.get(symbol, [])
        return buffer[-count:] if buffer else []

    async def close(self):
        if self.ws:
            self.ws.close()
            logger.info("WebSocket connection closed.")
