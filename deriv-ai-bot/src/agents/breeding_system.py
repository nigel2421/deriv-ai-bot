import random
import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StrategyDNA:
    """Strategy DNA representation for Genetic Algorithm breeding."""
    agent_id: str
    ema_fast: int = 10
    ema_slow: int = 20
    rsi_period: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    confidence_threshold: float = 0.75
    weight: float = 1.0
    generation: int = 1
    fitness_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AgentBreedingSystem:
    """
    Stage 7: Agent Breeding System (Genetic Algorithm)
    
    Responsibilities:
    1. Encapsulates strategy parameters as StrategyDNA.
    2. Ranks agent population by fitness score: (Win_Rate * Profit_Factor) - Drawdown.
    3. Performs Crossover between top-performing parent agents.
    4. Applies Mutation to generate innovative child sub-agents.
    5. Evolves strategy population over generations.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/agent_dna_population.json")
        self.population: List[StrategyDNA] = []
        self._load_population()

    def _load_population(self) -> None:
        """Load stored DNA population from disk if exists."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.population = [StrategyDNA(**item) for item in data]
                logger.info("Loaded DNA population with %d agents", len(self.population))
            except Exception as e:
                logger.warning("Failed to load DNA population (%s), generating defaults", e)
                self._seed_default_population()
        else:
            self._seed_default_population()

    def _seed_default_population(self) -> None:
        """Initialize baseline generation 1 DNA population."""
        self.population = [
            StrategyDNA(agent_id="DNA_Alpha_01", ema_fast=9, ema_slow=21, rsi_period=14, confidence_threshold=0.80),
            StrategyDNA(agent_id="DNA_Beta_02", ema_fast=12, ema_slow=26, rsi_period=14, confidence_threshold=0.75),
            StrategyDNA(agent_id="DNA_Gamma_03", ema_fast=5, ema_slow=15, rsi_period=10, confidence_threshold=0.85),
        ]
        self.save_population()

    def save_population(self) -> None:
        """Persist current DNA population to disk."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump([dna.to_dict() for dna in self.population], f, indent=2)
        except Exception as e:
            logger.error("Failed to save DNA population: %s", e)

    def calculate_fitness(self, win_rate: float, profit: float, total_trades: int) -> float:
        """
        Fitness function = (Win_Rate * 100) + (Profit * 2) + min(total_trades, 50)
        """
        if total_trades < 5:
            return 50.0  # Default baseline before minimum samples
        return round((win_rate * 100.0) + (profit * 2.0) + min(total_trades, 50), 2)

    def crossover(self, parent1: StrategyDNA, parent2: StrategyDNA, child_id: str) -> StrategyDNA:
        """
        Genetic Crossover operator: Combines gene traits from 2 fit parents.
        """
        child_generation = max(parent1.generation, parent2.generation) + 1
        return StrategyDNA(
            agent_id=child_id,
            ema_fast=parent1.ema_fast if random.random() > 0.5 else parent2.ema_fast,
            ema_slow=parent1.ema_slow if random.random() > 0.5 else parent2.ema_slow,
            rsi_period=parent1.rsi_period if random.random() > 0.5 else parent2.rsi_period,
            rsi_overbought=parent1.rsi_overbought if random.random() > 0.5 else parent2.rsi_overbought,
            rsi_oversold=parent1.rsi_oversold if random.random() > 0.5 else parent2.rsi_oversold,
            confidence_threshold=round((parent1.confidence_threshold + parent2.confidence_threshold) / 2.0, 4),
            generation=child_generation,
        )

    def mutate(self, dna: StrategyDNA, mutation_rate: float = 0.15) -> StrategyDNA:
        """
        Genetic Mutation operator: Perturbs parameters randomly.
        """
        if random.random() < mutation_rate:
            dna.ema_fast = max(5, min(20, dna.ema_fast + random.choice([-2, -1, 1, 2])))
        if random.random() < mutation_rate:
            dna.ema_slow = max(18, min(40, dna.ema_slow + random.choice([-3, -1, 1, 3])))
        if random.random() < mutation_rate:
            dna.confidence_threshold = max(0.65, min(0.92, round(dna.confidence_threshold + random.choice([-0.05, 0.05]), 4)))
        return dna

    def breed_next_generation(self) -> List[StrategyDNA]:
        """
        Select top fittest parents and breed next generation offspring.
        """
        if len(self.population) < 2:
            return self.population

        # Sort population by fitness DESC
        self.population.sort(key=lambda x: x.fitness_score, reverse=True)
        top_parents = self.population[:2]

        child_id = f"Child_Gen{top_parents[0].generation + 1}_{random.randint(100, 999)}"
        offspring = self.crossover(top_parents[0], top_parents[1], child_id=child_id)
        offspring = self.mutate(offspring, mutation_rate=0.20)

        self.population.append(offspring)
        self.save_population()
        logger.info("AgentBreedingSystem bred new offspring: %s (gen=%d)", offspring.agent_id, offspring.generation)
        return self.population
