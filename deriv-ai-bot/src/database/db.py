"""
PostgreSQL & Redis database helper for Deriv AI Trading Bot.
Provides high-performance persistent connection pooling and robust JSON/SQLite fallbacks.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Optional database driver imports
try:
    import psycopg2
    import psycopg2.extras
    from psycopg2.pool import SimpleConnectionPool
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False


class DatabaseManager:
    """Singleton manager for PostgreSQL connection pool and Redis cache."""

    _instance: Optional[DatabaseManager] = None

    def __init__(self):
        self.pg_pool: Optional[Any] = None
        self.redis_client: Optional[Any] = None
        self.enabled = False
        self._init_connections()

    @classmethod
    def get_instance(cls) -> DatabaseManager:
        if cls._instance is None:
            cls._instance = DatabaseManager()
        return cls._instance

    def _init_connections(self):
        pg_host = os.getenv("POSTGRES_HOST", "postgres")
        pg_port = int(os.getenv("POSTGRES_PORT", "5432"))
        pg_db = os.getenv("POSTGRES_DB", "deriv_trading_db")
        pg_user = os.getenv("POSTGRES_USER", "deriv_user")
        pg_pass = os.getenv("POSTGRES_PASSWORD", "deriv_secure_pass")

        if HAS_POSTGRES and os.getenv("ENABLE_POSTGRES", "true").lower() in {"1", "true", "yes"}:
            try:
                self.pg_pool = SimpleConnectionPool(
                    minconn=1,
                    maxconn=10,
                    host=pg_host,
                    port=pg_port,
                    dbname=pg_db,
                    user=pg_user,
                    password=pg_pass,
                    connect_timeout=5,
                )
                self.enabled = True
                logger.info("Connected to PostgreSQL pool host=%s db=%s", pg_host, pg_db)
            except Exception as e:
                logger.warning("PostgreSQL connection failed (%s). Falling back to JSON file storage.", e)
                self.pg_pool = None

        redis_host = os.getenv("REDIS_HOST", "redis")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))

        if HAS_REDIS and os.getenv("ENABLE_REDIS", "true").lower() in {"1", "true", "yes"}:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    socket_connect_timeout=3,
                    decode_responses=True,
                )
                self.redis_client.ping()
                logger.info("Connected to Redis host=%s port=%s", redis_host, redis_port)
            except Exception as e:
                logger.warning("Redis connection failed (%s). Continuing without cache.", e)
                self.redis_client = None

    def get_connection(self):
        if self.pg_pool:
            return self.pg_pool.getconn()
        return None

    def release_connection(self, conn):
        if self.pg_pool and conn:
            self.pg_pool.putconn(conn)

    def record_trade(self, trade: Dict[str, Any]) -> bool:
        """Insert or update trade in PostgreSQL database."""
        conn = self.get_connection()
        if not conn:
            return False

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO trades (
                        contract_id, account_id, account_mode, symbol, contract_type,
                        barrier, stake, pnl, profit, buy_price, sell_price,
                        confidence, confidence_level, mor_score, family, horizon,
                        opened_at, closed_at, status, strategy_name, raw_payload
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (contract_id) DO UPDATE SET
                        pnl = EXCLUDED.pnl,
                        profit = EXCLUDED.profit,
                        sell_price = EXCLUDED.sell_price,
                        closed_at = EXCLUDED.closed_at,
                        status = EXCLUDED.status,
                        raw_payload = EXCLUDED.raw_payload;
                    """,
                    (
                        trade.get("contract_id"),
                        trade.get("account_id", "unknown"),
                        os.getenv("MODE", "demo"),
                        trade.get("symbol"),
                        trade.get("contract_type"),
                        str(trade.get("barrier")) if trade.get("barrier") is not None else None,
                        float(trade.get("stake") or 0.0),
                        float(trade.get("pnl") or 0.0) if trade.get("pnl") is not None else None,
                        float(trade.get("profit") or 0.0) if trade.get("profit") is not None else None,
                        float(trade.get("buy_price") or 0.0) if trade.get("buy_price") is not None else None,
                        float(trade.get("sell_price") or 0.0) if trade.get("sell_price") is not None else None,
                        float(trade.get("confidence") or 0.0),
                        trade.get("confidence_level"),
                        float(trade.get("mor_score") or 0.0) if trade.get("mor_score") is not None else None,
                        trade.get("family"),
                        trade.get("horizon"),
                        trade.get("opened_at") or datetime.now(timezone.utc),
                        trade.get("closed_at"),
                        trade.get("status", "open"),
                        trade.get("strategy", "Default"),
                        json.dumps(trade),
                    ),
                )
                conn.commit()
            return True
        except Exception as e:
            logger.exception("Failed to record trade contract_id=%s in DB: %s", trade.get("contract_id"), e)
            conn.rollback()
            return False
        finally:
            self.release_connection(conn)

    def fetch_recent_trades(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch recent trades from PostgreSQL."""
        conn = self.get_connection()
        if not conn:
            return []

        try:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(
                    """
                    SELECT contract_id, symbol, contract_type, barrier, stake, pnl, profit,
                           confidence, confidence_level, mor_score, family, horizon,
                           opened_at, closed_at, status, strategy_name, raw_payload
                    FROM trades
                    ORDER BY id DESC
                    LIMIT %s;
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error("Failed to fetch trades from DB: %s", e)
            return []
        finally:
            self.release_connection(conn)

    def record_learning_cycle(self, summary: Dict[str, Any]) -> bool:
        """Store learning engine cycle results in PostgreSQL."""
        conn = self.get_connection()
        if not conn:
            return False

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO learning_cycles (
                        trades_analyzed, overall_win_rate, overall_calibration_error,
                        top_patterns, banned_setups, boosted_setups, confidence_adjustments,
                        audit_report, deepseek_summary
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
                    """,
                    (
                        summary.get("trades_analyzed", 0),
                        summary.get("overall_win_rate", 0.0),
                        summary.get("overall_calibration_error", 0.0),
                        json.dumps(summary.get("top_patterns", [])),
                        json.dumps(summary.get("banned_setups", [])),
                        json.dumps(summary.get("boosted_setups", [])),
                        json.dumps(summary.get("confidence_adjustments", {})),
                        json.dumps(summary.get("audit_report", {})),
                        summary.get("deepseek_summary", ""),
                    ),
                )
                conn.commit()
            return True
        except Exception as e:
            logger.exception("Failed to record learning cycle in DB: %s", e)
            conn.rollback()
            return False
        finally:
            self.release_connection(conn)

    def cache_set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set key in Redis cache."""
        if not self.redis_client:
            return False
        try:
            data = json.dumps(value) if not isinstance(value, str) else value
            if ttl_seconds:
                self.redis_client.setex(key, ttl_seconds, data)
            else:
                self.redis_client.set(key, data)
            return True
        except Exception as e:
            logger.error("Redis set error key=%s: %s", key, e)
            return False

    def cache_get(self, key: str) -> Optional[Any]:
        """Get key from Redis cache."""
        if not self.redis_client:
            return None
        try:
            val = self.redis_client.get(key)
            if val is None:
                return None
            try:
                return json.loads(val)
            except json.JSONDecodeError:
                return val
    def record_trade_memory(self, memory_payload: Dict[str, Any]) -> bool:
        """Store comprehensive trade telemetry (ticks, indicators, votes) into PostgreSQL trade_memory."""
        conn = self.get_connection()
        if not conn or not memory_payload:
            return False

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO trade_memory (
                        contract_id, symbol, contract_type, stake, pnl, profit,
                        status, confidence, ensemble_score, tick_history,
                        indicators_snapshot, agent_votes, feature_vector, opened_at, closed_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (contract_id) DO UPDATE SET
                        pnl = EXCLUDED.pnl,
                        profit = EXCLUDED.profit,
                        status = EXCLUDED.status,
                        closed_at = EXCLUDED.closed_at;
                    """,
                    (
                        memory_payload.get("contract_id"),
                        memory_payload.get("symbol"),
                        memory_payload.get("contract_type"),
                        float(memory_payload.get("stake", 1.0)),
                        float(memory_payload.get("pnl", 0.0)) if memory_payload.get("pnl") is not None else None,
                        float(memory_payload.get("profit", 0.0)) if memory_payload.get("profit") is not None else None,
                        memory_payload.get("status", "open"),
                        float(memory_payload.get("confidence", 0.80)),
                        float(memory_payload.get("ensemble_score", 0.80)),
                        json.dumps(memory_payload.get("ticks", [])),
                        json.dumps(memory_payload.get("indicators", {})),
                        json.dumps(memory_payload.get("agent_votes", [])),
                        json.dumps(memory_payload.get("feature_vector", {})),
                        memory_payload.get("opened_at") or datetime.now(timezone.utc),
                        memory_payload.get("closed_at"),
                    ),
                )
                conn.commit()
            return True
        except Exception as e:
            logger.error("Failed to record trade_memory in DB: %s", e)
            conn.rollback()
            return False
        finally:
            self.release_connection(conn)

    def fetch_trade_memories(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Fetch rich trade memories for offline AI model training."""
        conn = self.get_connection()
        if not conn:
            return []
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM trade_memory
                    ORDER BY id DESC
                    LIMIT %s;
                    """,
                    (limit,),
                )
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error("Failed to fetch trade_memory from DB: %s", e)
            return []
        finally:
            self.release_connection(conn)




    def record_votes(self, votes: List[Dict[str, Any]]) -> bool:
        """Record consensus agent votes in PostgreSQL."""
        conn = self.get_connection()
        if not conn or not votes:
            return False

        try:
            with conn.cursor() as cur:
                for v in votes:
                    cur.execute(
                        """
                        INSERT INTO agent_votes (
                            symbol, contract_type, agent_name, confidence, weight, weighted_score, rationale
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s);
                        """,
                        (
                            v.get("symbol"),
                            v.get("contract_type"),
                            v.get("agent_name"),
                            float(v.get("confidence", 0.0)),
                            float(v.get("weight", 1.0)),
                            float(v.get("score", 0.0)),
                            v.get("rationale", ""),
                        ),
                    )
                conn.commit()
            return True
        except Exception as e:
            logger.error("Failed to record agent votes in DB: %s", e)
            conn.rollback()
            return False
        finally:
            self.release_connection(conn)

    def fetch_agent_states(self) -> List[Dict[str, Any]]:
        """Fetch all agent states from PostgreSQL."""
        conn = self.get_connection()
        if not conn:
            return []
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute("SELECT * FROM agent_states ORDER BY agent_name;")
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error("Failed to fetch agent states: %s", e)
            return []
        finally:
            self.release_connection(conn)

    def update_agent_state(self, agent_name: str, enabled: Optional[bool] = None, status: Optional[str] = None, weight: Optional[float] = None) -> bool:
        """Update agent status/weight in agent_states table."""
        conn = self.get_connection()
        if not conn:
            return False
        try:
            with conn.cursor() as cur:
                updates = []
                params = []
                if enabled is not None:
                    updates.append("enabled = %s")
                    params.append(enabled)
                if status is not None:
                    updates.append("status = %s")
                    params.append(status)
                if weight is not None:
                    updates.append("weight = %s")
                    params.append(weight)

                updates.append("last_heartbeat = CURRENT_TIMESTAMP")
                if updates:
                    query = f"UPDATE agent_states SET {', '.join(updates)} WHERE agent_name = %s;"
                    params.append(agent_name)
                    cur.execute(query, tuple(params))
                    conn.commit()
            return True
        except Exception as e:
            logger.error("Failed to update agent state for %s: %s", agent_name, e)
            conn.rollback()
            return False
        finally:
            self.release_connection(conn)


db_manager = DatabaseManager.get_instance()

