import asyncio
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

try:
    import redis.asyncio as aioredis
    HAS_REDIS = True
except ImportError:
    aioredis = None
    HAS_REDIS = False

logger = logging.getLogger(__name__)



# Standard System Event Topics
TOPIC_TICKS = "deriv:ticks"
TOPIC_SIGNALS = "deriv:signals"
TOPIC_CONSENSUS = "deriv:consensus"
TOPIC_EXECUTIONS = "deriv:executions"
TOPIC_LEARNING = "deriv:learning"
TOPIC_ALERTS = "deriv:alerts"
TOPIC_CONTROL = "deriv:control"


class AgentEventBus:
    """
    Decoupled Async Event Bus for Multi-Agent Communication.
    Supports high-performance Redis Pub/Sub for containerized multi-agent execution,
    with automatic fallback to in-memory asyncio queues for standalone/test operation.
    """

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL")
        self._subscribers: Dict[str, List[Callable[[Dict[str, Any]], Any]]] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._loop_task: Optional[asyncio.Task] = None
        self._redis_client: Optional[aioredis.Redis] = None
        self._pubsub_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        """Establish async Redis connection if URL configured."""
        if self.redis_url and not self._redis_client:
            try:
                self._redis_client = aioredis.from_url(
                    self.redis_url, decode_responses=True
                )
                await self._redis_client.ping()
                logger.info("Connected to Redis Pub/Sub at %s", self.redis_url)
            except Exception as e:
                logger.warning("Redis connection failed (%s). Falling back to in-memory event bus.", e)
                self._redis_client = None

    async def disconnect(self) -> None:
        """Cleanly shut down Redis connection."""
        if self._redis_client:
            try:
                await self._redis_client.close()
                logger.info("Closed Redis connection")
            except Exception as e:
                logger.error("Error closing Redis connection: %s", e)
            self._redis_client = None

    def subscribe(self, topic: str, callback: Callable[[Dict[str, Any]], Any]) -> None:
        """Subscribe a handler callback to a specific topic."""
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        if callback not in self._subscribers[topic]:
            self._subscribers[topic].append(callback)
            logger.debug("Subscribed %s to topic '%s'", getattr(callback, "__name__", str(callback)), topic)

    def unsubscribe(self, topic: str, callback: Callable[[Dict[str, Any]], Any]) -> None:
        """Remove a subscriber callback from a topic."""
        if topic in self._subscribers and callback in self._subscribers[topic]:
            self._subscribers[topic].remove(callback)

    async def publish(self, topic: str, payload: Dict[str, Any]) -> None:
        """Publish an event message payload to a topic."""
        event = {"topic": topic, "payload": payload}

        # 1. Publish to Redis channel if connected
        if self._redis_client:
            try:
                msg_str = json.dumps(payload, default=str)
                await self._redis_client.publish(topic, msg_str)
            except Exception as e:
                logger.error("Failed to publish to Redis topic '%s': %s", topic, e)

        # 2. Enqueue in local queue & notify local subscribers
        await self._queue.put(event)
        await self._dispatch_local(topic, payload)

    async def _dispatch_local(self, topic: str, payload: Dict[str, Any]) -> None:
        """Dispatch payload to all in-memory subscribers for topic."""
        subscribers = self._subscribers.get(topic, [])
        for sub in subscribers:
            try:
                res = sub(payload)
                if asyncio.iscoroutine(res):
                    asyncio.create_task(res)
            except Exception as e:
                logger.error("Error in subscriber callback for topic '%s': %s", topic, e, exc_info=True)

    async def start(self) -> None:
        """Start background processing loop and Redis listener."""
        if self._running:
            return
        self._running = True
        await self.connect()

        self._loop_task = asyncio.create_task(self._process_queue())
        if self._redis_client and self._subscribers:
            self._pubsub_task = asyncio.create_task(self._redis_listener())

        logger.info("AgentEventBus started (Redis active: %s)", bool(self._redis_client))

    async def stop(self) -> None:
        """Stop background event loops and disconnect."""
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass

        if self._pubsub_task:
            self._pubsub_task.cancel()
            try:
                await self._pubsub_task
            except asyncio.CancelledError:
                pass

        await self.disconnect()
        logger.info("AgentEventBus stopped")

    async def _redis_listener(self) -> None:
        """Background task listening for Redis Pub/Sub messages across topics."""
        if not self._redis_client:
            return
        pubsub = self._redis_client.pubsub()
        try:
            topics = list(self._subscribers.keys())
            if not topics:
                return
            await pubsub.subscribe(*topics)
            logger.info("Redis PubSub listening on topics: %s", topics)

            async for message in pubsub.listen():
                if not self._running:
                    break
                if message["type"] == "message":
                    topic = message["channel"]
                    try:
                        data = json.loads(message["data"])
                        await self._dispatch_local(topic, data)
                    except json.JSONDecodeError:
                        logger.warning("Received invalid JSON payload on Redis topic '%s'", topic)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Redis listener encountered error: %s", e)
        finally:
            try:
                await pubsub.unsubscribe()
                await pubsub.close()
            except Exception:
                pass

    async def _process_queue(self) -> None:
        """Internal consumer loop for local queue processing."""
        while self._running:
            try:
                event = await self._queue.get()
                topic = event.get("topic")
                logger.debug("EventBus queue processed event on topic '%s'", topic)
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error processing queue event: %s", e)

