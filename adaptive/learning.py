"""Aggiornamento controllato dei pesi, senza modifica automatica del codice."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import numpy as np

from adaptive.models import Direction, MarketRegime, StrategyForecast


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

    def weights(self, regime: MarketRegime | None = None) -> dict[str, float]:
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
        regime: MarketRegime | None = None,
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


@dataclass(frozen=True)
class RegimeLearningConfig:
    skill_alpha: float = 0.10
    return_scale: float = 0.02
    minimum_observations: int = 30
    minimum_weight: float = 0.25
    maximum_weight: float = 1.50
    maximum_weight_step: float = 0.05
    weight_sensitivity: float = 1.50
    quarantine_threshold: float = -0.10
    release_threshold: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 < self.skill_alpha <= 1.0:
            raise ValueError("skill_alpha deve essere compreso tra 0 e 1.")
        if self.return_scale <= 0.0:
            raise ValueError("return_scale deve essere positivo.")
        if int(self.minimum_observations) <= 0:
            raise ValueError("minimum_observations deve essere positivo.")
        if not 0.0 < self.minimum_weight <= self.maximum_weight:
            raise ValueError("Range dei pesi regime-aware non valido.")
        if self.maximum_weight_step < 0.0:
            raise ValueError("maximum_weight_step non può essere negativo.")
        if self.weight_sensitivity < 0.0:
            raise ValueError("weight_sensitivity non può essere negativo.")
        if self.release_threshold <= self.quarantine_threshold:
            raise ValueError("release_threshold deve superare quarantine_threshold.")


@dataclass(frozen=True)
class RegimePerformanceState:
    strategy_id: str
    regime: MarketRegime
    observations: int = 0
    ewma_utility: float = 0.0
    weight: float = 1.0
    quarantined: bool = False


class CausalRegimePerformanceTracker:
    """Pesi walk-forward per modello e regime, aggiornati solo da outcome maturi."""

    def __init__(self, config: RegimeLearningConfig | None = None) -> None:
        self.config = config or RegimeLearningConfig()
        self._states: dict[tuple[str, MarketRegime], RegimePerformanceState] = {}
        self._lock = threading.RLock()

    def state(self, strategy_id: str, regime: MarketRegime) -> RegimePerformanceState:
        key = (str(strategy_id), MarketRegime(regime))
        with self._lock:
            return self._states.get(
                key, RegimePerformanceState(strategy_id=key[0], regime=key[1])
            )

    def weights(self, regime: MarketRegime | None = None) -> dict[str, float]:
        if regime is None:
            return {}
        normalized = MarketRegime(regime)
        with self._lock:
            return {
                strategy_id: state.weight
                for (strategy_id, state_regime), state in self._states.items()
                if state_regime is normalized
            }

    def states(self) -> tuple[RegimePerformanceState, ...]:
        with self._lock:
            return tuple(
                self._states[key]
                for key in sorted(self._states, key=lambda item: (item[0], item[1].value))
            )

    def update(
        self,
        forecast: StrategyForecast,
        realized_asset_return: float,
        transaction_cost_return: float = 0.0,
        regime: MarketRegime | None = None,
    ) -> RegimePerformanceState:
        normalized_regime = MarketRegime(regime or MarketRegime.UNKNOWN)
        if forecast.direction is Direction.FLAT:
            return self.state(forecast.strategy_id, normalized_regime)
        realized = float(realized_asset_return)
        costs = max(0.0, float(transaction_cost_return))
        if not math.isfinite(realized) or not math.isfinite(costs):
            raise ValueError("Outcome e costi devono essere finiti.")
        config = self.config
        utility = float(
            np.tanh(
                (forecast.direction.sign * realized - costs) / config.return_scale
            )
        )
        with self._lock:
            previous = self.state(forecast.strategy_id, normalized_regime)
            observations = previous.observations + 1
            ewma_utility = (
                (1.0 - config.skill_alpha) * previous.ewma_utility
                + config.skill_alpha * utility
            )
            quarantined = previous.quarantined
            if observations >= config.minimum_observations:
                if quarantined and ewma_utility >= config.release_threshold:
                    quarantined = False
                elif not quarantined and ewma_utility <= config.quarantine_threshold:
                    quarantined = True
            desired = (
                config.minimum_weight
                if quarantined
                else float(
                    np.clip(
                        math.exp(config.weight_sensitivity * ewma_utility),
                        config.minimum_weight,
                        config.maximum_weight,
                    )
                )
            )
            if observations < config.minimum_observations:
                desired = 1.0
            weight = float(
                np.clip(
                    desired,
                    previous.weight - config.maximum_weight_step,
                    previous.weight + config.maximum_weight_step,
                )
            )
            updated = RegimePerformanceState(
                strategy_id=forecast.strategy_id,
                regime=normalized_regime,
                observations=observations,
                ewma_utility=ewma_utility,
                weight=weight,
                quarantined=quarantined,
            )
            self._states[(forecast.strategy_id, normalized_regime)] = updated
            return updated
