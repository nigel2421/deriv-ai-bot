import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from src.agents.base_agent import BaseAgent, AgentSignal, ConsensusDecision
from src.agents.bus import (
    AgentEventBus,
    TOPIC_TICKS,
    TOPIC_SIGNALS,
    TOPIC_CONSENSUS,
    TOPIC_CONTROL,
    TOPIC_ALERTS,
)

from src.agents.market_subagent import MarketWatcherSubAgent, MONITORED_MARKETS

logger = logging.getLogger(__name__)


class AgentManager:
    """
    Agent Manager ('The CEO')
    
    Coordinates decision-making across all specialist strategy agents and 20 market sub-agents,
    manages agent lifecycles (start, stop, scale), and communicates over Redis.
    """

    def __init__(self, bus: Optional[AgentEventBus] = None):
        self.bus = bus or AgentEventBus()
        self.agents: Dict[str, BaseAgent] = {}
        self._running = False
        self._loop_task: Optional[asyncio.Task] = None
        self._agent_weights: Dict[str, float] = {}
        self._register_market_subagents()

    def _register_market_subagents(self) -> None:

        """Register sub-agents actively watching all 20 markets."""
        for item in MONITORED_MARKETS:
            sub = MarketWatcherSubAgent(symbol=item["symbol"], category=item["category"])
            self.register_agent(sub, weight=1.0)


    def register_agent(self, agent: BaseAgent, weight: float = 1.0) -> None:
        """Register a specialist intelligence agent with the manager."""
        self.agents[agent.name] = agent
        self._agent_weights[agent.name] = weight
        logger.info("Registered agent '%s' (class=%s, weight=%.2f)", agent.name, agent.__class__.__name__, weight)

    def unregister_agent(self, name: str) -> None:
        """Unregister an agent by name."""
        if name in self.agents:
            del self.agents[name]
            self._agent_weights.pop(name, None)
            logger.info("Unregistered agent '%s'", name)

    def set_agent_status(self, name: str, enabled: bool) -> bool:
        """Toggle an agent on/off dynamically."""
        if name in self.agents:
            self.agents[name].enabled = enabled
            logger.info("Agent '%s' enabled set to %s", name, enabled)
            return True
        return False

    def get_agent_status(self) -> List[Dict[str, Any]]:
        """Retrieve runtime status and confidence weights for all registered agents."""
        status_list = []
        for name, agent in self.agents.items():
            st = agent.get_status()
            st["weight"] = self._agent_weights.get(name, 1.0)
            status_list.append(st)
        return status_list

    async def evaluate_market_context(self, context: Dict[str, Any]) -> List[AgentSignal]:
        """
        Evaluate market context across all enabled strategy agents asynchronously.
        Returns a list of all generated signals and publishes them to TOPIC_SIGNALS.
        """
        symbol = context.get("symbol", "R_100")
        signals: List[AgentSignal] = []

        tasks = []
        enabled_agents = [ag for ag in self.agents.values() if ag.enabled]

        for agent in enabled_agents:
            tasks.append(self._safe_evaluate(agent, context))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for agent, res in zip(enabled_agents, results):
            if isinstance(res, Exception):
                logger.error("Agent '%s' failed evaluation on %s: %s", agent.name, symbol, res)
            elif isinstance(res, list):
                for sig in res:
                    # Enrich with configured weight
                    sig.weight = self._agent_weights.get(agent.name, 1.0)
                    signals.append(sig)

                    # Publish signal over Redis bus
                    await self.bus.publish(TOPIC_SIGNALS, sig.to_dict())

        logger.debug("Evaluated market for %s: generated %d signals across %d agents", symbol, len(signals), len(enabled_agents))
        return signals

    async def _safe_evaluate(self, agent: BaseAgent, context: Dict[str, Any]) -> List[AgentSignal]:
        """Safely execute agent evaluation with error catching."""
        try:
            return await agent.evaluate(context)
        except Exception as e:
            logger.exception("Error during agent %s evaluation: %s", agent.name, e)
            return []

    async def handle_control_command(self, payload: Dict[str, Any]) -> None:
        """Handle control messages from API / Android app over TOPIC_CONTROL."""
        action = payload.get("action")
        target = payload.get("target_agent")

        logger.info("AgentManager received control payload: action=%s, target=%s", action, target)

        if action == "enable_agent" and target:
            self.set_agent_status(target, True)
        elif action == "disable_agent" and target:
            self.set_agent_status(target, False)
        elif action == "set_weight" and target:
            weight = payload.get("weight", 1.0)
            if target in self._agent_weights:
                self._agent_weights[target] = float(weight)
        elif action == "stop_all":
            for ag in self.agents.values():
                ag.enabled = False
        elif action == "start_all":
            for ag in self.agents.values():
                ag.enabled = True

    async def start(self) -> None:
        """Start AgentManager event loops and subscribers."""
        if self._running:
            return
        self._running = True
        await self.bus.start()
        self.bus.subscribe(TOPIC_CONTROL, self.handle_control_command)
        logger.info("AgentManager (CEO) started with %d registered agents", len(self.agents))

    async def stop(self) -> None:
        """Stop AgentManager and underlying bus."""
        self._running = False
        await self.bus.stop()
        logger.info("AgentManager stopped")
