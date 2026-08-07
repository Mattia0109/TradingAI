"""Aggiornamento controllato dei pesi, senza modifica automatica del codice."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import numpy as np

from adaptive.models import Direction, StrategyForecast


@dataclass(frozen=True)
class StrategyPerformanceState:
    strategy_id: str
    observations: int = 0
    ewma_skill: float = 0.50
    weight: float = 1.0


@dataclass(frozen=True)
class LearningConfig:
    skill_alpha: float = 0.05
    learning_rate: float = 0.20
    minimum_weight: float = 0.50
    maximum_weight: float = 1.50
    maximum_weight_step: float = 0.05
    return_scale: float = 0.01

    def __post_init__(self) -> None:
        if not 0.0 < self.skill_alpha <= 1.0:
            raise ValueError("skill_alpha deve essere compreso tra 0 e 1.")
        if self.learning_rate < 0.0:
            raise ValueError("learning_rate non può essere negativo.")
        if self.minimum_weight <= 0.0:
            raise ValueError("minimum_weight deve essere positivo.")
        if self.maximum_weight < self.minimum_weight:
            raise ValueError("Range dei pesi non valido.")
        if self.maximum_weight_step < 0.0:
            raise ValueError("maximum_weight_step non può essere negativo.")
        if self.return_scale <= 0.0:
            raise ValueError("return_scale deve essere positivo.")


class ControlledPerformanceTracker:
    """Mantiene pesi limitati e aggiornati solo dopo un outcome osservato."""

    def __init__(self, config: LearningConfig | None = None) -> None:
        self.config = config or LearningConfig()
        self._states: dict[str, StrategyPerformanceState] = {}
        self._lock = threading.RLock()

    def state(self, strategy_id: str) -> StrategyPerformanceState:
        with self._lock:
            return self._states.get(
                strategy_id,
                StrategyPerformanceState(strategy_id=strategy_id),
            )

    def weights(self) -> dict[str, float]:
        with self._lock:
            return {
                strategy_id: state.weight
                for strategy_id, state in self._states.items()
            }

    def update(
        self,
        forecast: StrategyForecast,
        realized_asset_return: float,
        transaction_cost_return: float = 0.0,
    ) -> StrategyPerformanceState:
        if forecast.direction is Direction.FLAT:
            return self.state(forecast.strategy_id)

        realized = float(realized_asset_return)
        costs = max(0.0, float(transaction_cost_return))
        if not math.isfinite(realized) or not math.isfinite(costs):
            raise ValueError("Outcome e costi devono essere finiti.")

        config = self.config
        strategy_return = forecast.direction.sign * realized - costs
        utility = float(np.tanh(strategy_return / config.return_scale))
        skill_observation = 0.5 + 0.5 * utility

        with self._lock:
            previous = self.state(forecast.strategy_id)
            skill = (
                (1.0 - config.skill_alpha) * previous.ewma_skill
                + config.skill_alpha * skill_observation
            )
            unconstrained_weight = previous.weight * math.exp(
                config.learning_rate * utility
            )
            desired_weight = float(
                np.clip(
                    unconstrained_weight,
                    config.minimum_weight,
                    config.maximum_weight,
                )
            )
            lower_step = previous.weight - config.maximum_weight_step
            upper_step = previous.weight + config.maximum_weight_step
            bounded_weight = float(
                np.clip(desired_weight, lower_step, upper_step)
            )
            updated = StrategyPerformanceState(
                strategy_id=forecast.strategy_id,
                observations=previous.observations + 1,
                ewma_skill=skill,
                weight=bounded_weight,
            )
            self._states[forecast.strategy_id] = updated
            return updated
