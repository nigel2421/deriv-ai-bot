import asyncio
import json
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import websocket

from src.api.deriv_v2_auth import is_legacy_app_id, resolve_authenticated_ws_url

logger = logging.getLogger(__name__)

MessageHandler = Callable[[Dict[str, Any]], None]


class DerivAPIError(Exception):
    """Raised when Deriv returns an error payload for a request."""

    def __init__(self, error: Dict[str, Any]):
        self.error = error
        code = error.get("code", "Unknown")
        message = error.get("message", str(error))
        super().__init__(f"Deriv API error [{code}]: {message}")


class DerivClient:
    """
    WebSocket client for Deriv.com API with:
    - legacy numeric app_id + authorize token, OR
    - new OAuth/PAT app: REST Bearer + OTP WebSocket URL
    - background WS thread
    - req_id correlated request/response (asyncio Futures)
    - stream handlers (ticks, open contracts, etc.)
    """

    def __init__(
        self,
        app_id: str,
        token: str,
        mode: str = "demo",
        *,
        api_mode: Optional[str] = None,
        account_id: Optional[str] = None,
        api_base: str = "https://api.derivws.com",
    ):
        self.app_id = str(app_id).strip()
        self.token = (token or "").strip()
        self.mode = mode
        # auto | legacy | v2
        if api_mode in {"legacy", "v2"}:
            self.api_mode = api_mode
        else:
            self.api_mode = "legacy" if is_legacy_app_id(self.app_id) else "v2"
        self.account_id = account_id
        self.api_base = api_base.rstrip("/")
        self.ws: Optional[websocket.WebSocketApp] = None
        self.connected = False
        self.authorized = False
        self.account: Dict[str, Any] = {}
        self.subscriptions: Dict[str, bool] = {}
        self.tick_buffers: Dict[str, List[Dict]] = {}
        self.max_buffer_size = 1000
        self.last_request_time = 0.0
        self.rate_limit = 0.1  # min seconds between sends
        self._ws_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = threading.Lock()
        self._closing = False
        self._reconnect_lock = asyncio.Lock()
        self._req_id = 0
        self._pending: Dict[int, asyncio.Future] = {}
        self._handlers: Dict[str, List[MessageHandler]] = {}
        self._send_lock = threading.Lock()
        self._balance_subscribed = False
        self._ws_url: Optional[str] = None
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------------ handlers
    def register_handler(self, msg_type: str, handler: MessageHandler) -> None:
        """Register a callback for a msg_type (streams + unsolicited messages)."""
        self._handlers.setdefault(msg_type, []).append(handler)

    def unregister_handler(self, msg_type: str, handler: MessageHandler) -> None:
        handlers = self._handlers.get(msg_type, [])
        if handler in handlers:
            handlers.remove(handler)

    def _next_req_id(self) -> int:
        with self._lock:
            self._req_id += 1
            return self._req_id

    # ------------------------------------------------------------------ lifecycle
    async def connect(self) -> bool:
        """Establish WebSocket connection (legacy authorize or v2 OTP URL)."""
        self._loop = asyncio.get_running_loop()
        self._closing = False
        self.last_error = None
        self.authorized = False
        self.connected = False

        try:
            if self.api_mode == "v2":
                url = await self._resolve_v2_url()
            else:
                # Prefer modern host; binaryws still works for numeric app ids
                url = f"wss://ws.derivws.com/websockets/v3?app_id={self.app_id}"
            self._ws_url = url

            if self.ws:
                try:
                    self.ws.close()
                except Exception:
                    pass

            self.ws = websocket.WebSocketApp(
                url,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open,
            )
            self._ws_thread = threading.Thread(
                target=self._run_websocket, daemon=True, name="deriv-ws"
            )
            self._ws_thread.start()
            logger.info(
                "Connecting to Deriv WebSocket (mode=%s)...", self.api_mode
            )

            for _ in range(60):  # ~12s
                if self.authorized:
                    return True
                await asyncio.sleep(0.2)

            # V2 OTP sockets may be pre-authenticated — probe with balance
            if self.api_mode == "v2" and self.connected:
                try:
                    bal = await self.request({"balance": 1}, timeout=8.0)
                    if not bal.get("error") and bal.get("balance") is not None:
                        self._apply_balance_payload(bal)
                        self.authorized = True
                        logger.info("V2 WebSocket authorized via balance probe.")
                        return True
                    if bal.get("error"):
                        self.last_error = str(bal.get("error"))
                        logger.error("V2 balance probe error: %s", bal.get("error"))
                except Exception as e:
                    self.last_error = str(e)
                    logger.error("V2 balance probe failed: %s", e)

            logger.warning(
                "WebSocket connected but authorize not confirmed yet (%s).",
                self.last_error or "timeout",
            )
            return False
        except Exception as e:
            self.last_error = str(e)
            logger.error("Connection failed: %s", e)
            return False

    async def _resolve_v2_url(self) -> str:
        if not self.token:
            raise RuntimeError(
                "DERIV_API_TOKEN missing. For OAuth/PAT apps this must be a "
                "Bearer access token (or PAT) from developers.deriv.com."
            )
        logger.info(
            "Resolving v2 OTP WebSocket URL (app_id=%s… account=%s)",
            self.app_id[:8],
            self.account_id or "auto",
        )
        url, account = await resolve_authenticated_ws_url(
            self.app_id,
            self.token,
            mode=self.mode,
            account_id=self.account_id,
            api_base=self.api_base,
        )
        # Seed account snapshot from REST
        self.account = {
            "loginid": account.get("account_id")
            or account.get("loginid")
            or account.get("id"),
            "balance": account.get("balance"),
            "currency": account.get("currency") or "USD",
            "account_type": account.get("account_type") or account.get("group"),
            **{k: v for k, v in account.items() if k not in {"balance"}},
        }
        self.account_id = str(self.account.get("loginid") or self.account_id or "")
        return url

    def _run_websocket(self) -> None:
        if self.ws:
            self.ws.run_forever(ping_interval=30, ping_timeout=10)

    def on_open(self, ws) -> None:
        logger.info("WebSocket connection opened (mode=%s).", self.api_mode)
        self.connected = True
        if self.api_mode == "legacy":
            # Classic flow: authorize with API token
            ws.send(json.dumps({"authorize": self.token}))
        else:
            # V2 OTP URL is pre-authenticated; confirm with balance when loop is ready
            self.authorized = True
            logger.info(
                "V2 OTP socket open — treating as authorized (account=%s).",
                self.account.get("loginid"),
            )
            # Resubscribe streams after reconnect
            symbols = list(self.subscriptions.keys())
            self.subscriptions.clear()
            if symbols:
                self.subscribe_ticks(symbols)
            if self._balance_subscribed:
                self._balance_subscribed = False
                self.subscribe_balance()

    def on_message(self, ws, message: str) -> None:
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            logger.error("Invalid JSON from Deriv: %s", message[:200])
            return

        req_id = data.get("req_id")
        if req_id is not None:
            fut = None
            with self._lock:
                fut = self._pending.pop(req_id, None)
            if fut is not None and not fut.done() and self._loop:
                self._loop.call_soon_threadsafe(self._resolve_future, fut, data)

        msg_type = data.get("msg_type")

        if msg_type == "authorize":
            self._handle_authorize(data)
        elif msg_type == "tick":
            self.handle_tick(data)
        elif msg_type == "balance":
            self._handle_balance(data)
        elif data.get("error") and req_id is None:
            logger.error("API error (no req_id): %s", data["error"])

        # Stream / type handlers (proposal_open_contract updates, etc.)
        if msg_type:
            for handler in list(self._handlers.get(msg_type, [])):
                try:
                    handler(data)
                except Exception as e:
                    logger.exception("Handler for %s failed: %s", msg_type, e)

    def _resolve_future(self, fut: asyncio.Future, data: Dict[str, Any]) -> None:
        if not fut.done():
            fut.set_result(data)

    def _fail_future(self, fut: asyncio.Future, exc: Exception) -> None:
        if not fut.done():
            fut.set_exception(exc)

    def _handle_authorize(self, data: Dict[str, Any]) -> None:
        if data.get("error"):
            logger.error("Auth error: %s", data["error"])
            self.authorized = False
            self.account = {}
            return

        auth = data.get("authorize") or {}
        self.account = auth
        self.authorized = True
        self._balance_subscribed = False
        balance = auth.get("balance")
        currency = auth.get("currency")
        loginid = auth.get("loginid")
        logger.info(
            "Authorized loginid=%s balance=%s %s",
            loginid,
            balance,
            currency,
        )
        # Resubscribe after (re)auth
        symbols = list(self.subscriptions.keys())
        self.subscriptions.clear()
        if symbols:
            self.subscribe_ticks(symbols)
        # Live balance stream (fire-and-forget; also available via refresh_balance)
        self.subscribe_balance()

    def handle_tick(self, data: Dict) -> None:
        tick = data.get("tick") or {}
        symbol = tick.get("symbol")
        if not symbol:
            return
        with self._lock:
            if symbol not in self.tick_buffers:
                self.tick_buffers[symbol] = []
            self.tick_buffers[symbol].append(tick)
            if len(self.tick_buffers[symbol]) > self.max_buffer_size:
                self.tick_buffers[symbol].pop(0)
        logger.debug("Tick %s: %s", symbol, tick.get("quote"))

    def on_error(self, ws, error) -> None:
        logger.error("WebSocket error: %s", error)

    def on_close(self, ws, close_status_code, close_msg) -> None:
        logger.warning("WebSocket closed: %s - %s", close_status_code, close_msg)
        self.connected = False
        self.authorized = False
        self._reject_all_pending(ConnectionError("WebSocket closed"))
        if self._closing:
            return
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self.reconnect(), self._loop)

    def _reject_all_pending(self, exc: Exception) -> None:
        with self._lock:
            pending = list(self._pending.items())
            self._pending.clear()
        if not self._loop:
            return
        for _, fut in pending:
            if not fut.done():
                self._loop.call_soon_threadsafe(self._fail_future, fut, exc)

    async def reconnect(self, retries: int = 5) -> None:
        if self._closing:
            return
        async with self._reconnect_lock:
            if self._closing or self.authorized:
                return
            for attempt in range(retries):
                if self._closing:
                    return
                wait = min((2**attempt) * 5, 60)
                logger.info(
                    "Reconnecting in %ss (attempt %s/%s)...",
                    wait,
                    attempt + 1,
                    retries,
                )
                await asyncio.sleep(wait)
                ok = await self.connect()
                if ok:
                    return
            logger.error("Max reconnection attempts reached.")

    # ------------------------------------------------------------------ send / request
    def send_message(self, msg: Dict[str, Any]) -> None:
        """Fire-and-forget send (subscriptions). Prefer request() for RPC."""
        self._send_raw(msg)

    def _send_raw(self, msg: Dict[str, Any]) -> None:
        if not self.ws or not self.connected:
            logger.warning("Cannot send message: not connected (%s)", list(msg.keys())[:3])
            return

        with self._send_lock:
            now = time.time()
            delta = now - self.last_request_time
            if delta < self.rate_limit:
                time.sleep(self.rate_limit - delta)
            try:
                self.ws.send(json.dumps(msg))
                self.last_request_time = time.time()
            except Exception as e:
                logger.error("Failed to send message: %s", e)

    async def request(
        self,
        msg: Dict[str, Any],
        timeout: float = 20.0,
        *,
        raise_on_error: bool = False,
    ) -> Dict[str, Any]:
        """
        Send a message with req_id and wait for the matching response.

        If raise_on_error is True and the payload contains error, raises DerivAPIError.
        """
        if not self._loop:
            raise RuntimeError("Client not connected (no event loop). Call connect() first.")

        req_id = self._next_req_id()
        payload = {**msg, "req_id": req_id}
        fut: asyncio.Future = self._loop.create_future()
        with self._lock:
            self._pending[req_id] = fut

        self._send_raw(payload)

        try:
            data = await asyncio.wait_for(asyncio.shield(fut), timeout=timeout)
        except asyncio.TimeoutError:
            with self._lock:
                self._pending.pop(req_id, None)
            raise TimeoutError(
                f"Deriv request timed out after {timeout}s: {list(msg.keys())}"
            ) from None
        except Exception:
            with self._lock:
                self._pending.pop(req_id, None)
            raise

        if raise_on_error and data.get("error"):
            raise DerivAPIError(data["error"])
        return data

    def subscribe_ticks(self, symbols: List[str]) -> None:
        """Subscribe to real-time ticks for multiple symbols."""
        for symbol in symbols:
            if symbol in self.subscriptions:
                continue
            msg = {"ticks": symbol, "subscribe": 1}
            self.send_message(msg)
            self.subscriptions[symbol] = True
            with self._lock:
                if symbol not in self.tick_buffers:
                    self.tick_buffers[symbol] = []
            logger.info("Subscribed to ticks for %s", symbol)

    def get_latest_ticks(self, symbol: str, count: int = 50) -> List[Dict]:
        with self._lock:
            buffer = list(self.tick_buffers.get(symbol, []))
        return buffer[-count:] if buffer else []

    def buffer_size(self, symbol: str) -> int:
        with self._lock:
            return len(self.tick_buffers.get(symbol, []))

    @staticmethod
    def parse_history_response(
        data: Dict[str, Any], symbol: str
    ) -> List[Dict[str, Any]]:
        """
        Normalize a ticks_history API response into tick dicts:
        [{symbol, quote, epoch}, ...] oldest → newest.
        """
        if not data or data.get("error"):
            return []

        ticks: List[Dict[str, Any]] = []

        history = data.get("history") or {}
        prices = history.get("prices") or []
        times = history.get("times") or []
        if prices and times and len(prices) == len(times):
            for price, epoch in zip(prices, times):
                ticks.append(
                    {
                        "symbol": symbol,
                        "quote": float(price),
                        "epoch": int(epoch),
                    }
                )
            return ticks

        # Alternate shapes
        raw_ticks = data.get("ticks") or data.get("history", {}).get("ticks") or []
        if isinstance(raw_ticks, list):
            for t in raw_ticks:
                if not isinstance(t, dict):
                    continue
                quote = t.get("quote", t.get("price"))
                epoch = t.get("epoch", t.get("time"))
                if quote is None:
                    continue
                ticks.append(
                    {
                        "symbol": t.get("symbol") or symbol,
                        "quote": float(quote),
                        "epoch": int(epoch) if epoch is not None else 0,
                    }
                )
        return ticks

    def seed_tick_buffer(
        self,
        symbol: str,
        ticks: List[Dict[str, Any]],
        *,
        prepend: bool = True,
    ) -> int:
        """
        Seed / merge historical ticks into the live buffer.

        When prepend=True (default), history is placed before any live ticks,
        deduped by epoch. Buffer is trimmed to max_buffer_size (keeps newest).
        Returns resulting buffer length.
        """
        if not ticks:
            return self.buffer_size(symbol)

        normalized: List[Dict[str, Any]] = []
        for t in ticks:
            try:
                normalized.append(
                    {
                        "symbol": t.get("symbol") or symbol,
                        "quote": float(t["quote"]),
                        "epoch": int(t.get("epoch") or 0),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue

        with self._lock:
            existing = list(self.tick_buffers.get(symbol, []))
            if prepend:
                combined = normalized + existing
            else:
                combined = existing + normalized

            # Dedupe by epoch (keep last occurrence = prefer live over history)
            by_epoch: Dict[int, Dict[str, Any]] = {}
            no_epoch: List[Dict[str, Any]] = []
            for t in combined:
                ep = int(t.get("epoch") or 0)
                if ep:
                    by_epoch[ep] = t
                else:
                    no_epoch.append(t)
            merged = sorted(by_epoch.values(), key=lambda x: int(x.get("epoch") or 0))
            merged.extend(no_epoch)

            if len(merged) > self.max_buffer_size:
                merged = merged[-self.max_buffer_size :]
            self.tick_buffers[symbol] = merged
            return len(merged)

    async def fetch_ticks_history(
        self,
        symbol: str,
        count: int = 500,
        *,
        end: str = "latest",
        style: str = "ticks",
        timeout: float = 25.0,
        seed_buffer: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Request historical ticks via ticks_history and optionally seed the buffer.
        """
        count = max(1, min(int(count), 5000))
        msg = {
            "ticks_history": symbol,
            "count": count,
            "end": end,
            "style": style,
            "adjust_start_time": 1,
        }
        logger.info("Fetching ticks_history %s count=%s end=%s", symbol, count, end)
        try:
            data = await self.request(msg, timeout=timeout)
        except Exception as e:
            logger.error("ticks_history request failed for %s: %s", symbol, e)
            return []

        if data.get("error"):
            logger.error(
                "ticks_history error for %s: %s", symbol, data.get("error")
            )
            return []

        ticks = self.parse_history_response(data, symbol)
        if not ticks:
            logger.warning("ticks_history returned no ticks for %s", symbol)
            return []

        if seed_buffer:
            size = self.seed_tick_buffer(symbol, ticks, prepend=True)
            logger.info(
                "Bootstrapped %s with %d history ticks (buffer=%d)",
                symbol,
                len(ticks),
                size,
            )
        else:
            logger.info("Fetched %d history ticks for %s (not seeded)", len(ticks), symbol)

        return ticks

    def get_balance(self) -> Optional[float]:
        bal = self.account.get("balance")
        try:
            return float(bal) if bal is not None else None
        except (TypeError, ValueError):
            return None

    def get_currency(self) -> str:
        return str(self.account.get("currency") or "USD")

    def set_balance(self, balance: float) -> None:
        """Update cached balance (e.g. from buy.balance_after)."""
        try:
            self.account["balance"] = float(balance)
        except (TypeError, ValueError):
            pass

    def _handle_balance(self, data: Dict[str, Any]) -> None:
        if data.get("error"):
            logger.error("Balance error: %s", data["error"])
            return
        self._apply_balance_payload(data)

    def _apply_balance_payload(self, data: Dict[str, Any]) -> None:
        bal_obj = data.get("balance") or {}
        # Payload may be {"balance": {"balance": 123, "currency": "USD"}} or flat
        if isinstance(bal_obj, dict):
            value = bal_obj.get("balance", bal_obj.get("amount"))
            currency = bal_obj.get("currency")
            loginid = bal_obj.get("loginid")
            if currency:
                self.account["currency"] = currency
            if loginid:
                self.account["loginid"] = loginid
        else:
            value = bal_obj
        if value is not None:
            try:
                self.account["balance"] = float(value)
                logger.debug(
                    "Balance update: %s %s",
                    self.account["balance"],
                    self.get_currency(),
                )
            except (TypeError, ValueError):
                logger.warning("Could not parse balance payload: %s", data)

    def subscribe_balance(self) -> None:
        """Subscribe to live balance updates."""
        if self._balance_subscribed or not self.connected:
            return
        self.send_message({"balance": 1, "subscribe": 1})
        self._balance_subscribed = True
        logger.info("Subscribed to balance stream")

    async def refresh_balance(self, timeout: float = 10.0) -> Optional[float]:
        """One-shot balance request; updates cache and returns balance."""
        try:
            data = await self.request({"balance": 1}, timeout=timeout)
        except Exception as e:
            logger.error("refresh_balance failed: %s", e)
            return self.get_balance()

        if data.get("error"):
            logger.error("refresh_balance error: %s", data["error"])
            return self.get_balance()

        self._handle_balance(data)
        return self.get_balance()

    async def close(self) -> None:
        self._closing = True
        self.connected = False
        self.authorized = False
        self._reject_all_pending(ConnectionError("Client closed"))
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
            logger.info("WebSocket connection closed.")
