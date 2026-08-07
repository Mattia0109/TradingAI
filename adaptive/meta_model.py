"""Meta-modello conservativo per aggregare specialisti indipendenti."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from adaptive.models import (
    Direction,
    MetaDecision,
    RegimeAssessment,
    StrategyForecast,
)


@dataclass(frozen=True)
class MetaModelConfig:
    minimum_active_forecasts: int = 1
    minimum_agreement: float = 0.55
    minimum_confidence: float = 0.35
    minimum_expected_edge_bps: float = 5.0
    minimum_performance_weight: float = 0.25
    maximum_performance_weight: float = 2.0

    def __post_init__(self) -> None:
        if int(self.minimum_active_forecasts) <= 0:
            raise ValueError("minimum_active_forecasts deve essere positivo.")
        for name in ("minimum_agreement", "minimum_confidence"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} deve essere compreso tra 0 e 1.")
        if self.minimum_expected_edge_bps < 0.0:
            raise ValueError("minimum_expected_edge_bps non può essere negativo.")
        if self.minimum_performance_weight <= 0.0:
            raise ValueError("minimum_performance_weight deve essere positivo.")
        if self.maximum_performance_weight < self.minimum_performance_weight:
            raise ValueError("Range dei performance weight non valido.")


class AdaptiveMetaModel:
    """Combina forecast, ma mantiene NO_TRADE come esito predefinito."""

    def __init__(self, config: MetaModelConfig | None = None) -> None:
        self.config = config or MetaModelConfig()

    @staticmethod
    def _flat(ticker: str, reason: str) -> MetaDecision:
        return MetaDecision(
            ticker=ticker,
            direction=Direction.FLAT,
            confidence=0.0,
            expected_return=0.0,
            expected_volatility=0.0,
            agreement=0.0,
            horizon_bars=1,
            contributors={},
            reasons=(reason,),
        )

    def combine(
        self,
        forecasts: Sequence[StrategyForecast],
        regime: RegimeAssessment,
        performance_weights: Mapping[str, float] | None = None,
        *,
        ticker: str | None = None,
    ) -> MetaDecision:
        config = self.config
        performance_weights = performance_weights or {}
        inferred_ticker = ticker or (forecasts[0].ticker if forecasts else "UNKNOWN")

        if regime.risk_multiplier <= 0.0:
            return self._flat(inferred_ticker, "Il regime vieta nuova esposizione.")

        active = [
            forecast
            for forecast in forecasts
            if forecast.direction is not Direction.FLAT
            and forecast.confidence > 0.0
            and forecast.regime_fit > 0.0
        ]
        if len(active) < config.minimum_active_forecasts:
            return self._flat(
                inferred_ticker,
                "Nessuno specialista possiede edge sufficiente.",
            )

        if any(
            forecast.ticker != inferred_ticker.upper().strip()
            for forecast in active
        ):
            raise ValueError("Il meta-modello non può combinare ticker differenti.")

        raw_weights: list[float] = []
        for forecast in active:
            performance_weight = float(
                np.clip(
                    performance_weights.get(forecast.strategy_id, 1.0),
                    config.minimum_performance_weight,
                    config.maximum_performance_weight,
                )
            )
            raw_weights.append(
                max(
                    1e-12,
                    forecast.confidence
                    * forecast.regime_fit
                    * performance_weight,
                )
            )

        total_weight = sum(raw_weights)
        signed_vote = sum(
            weight * forecast.direction.sign
            for weight, forecast in zip(raw_weights, active)
        )
        agreement = abs(signed_vote) / total_weight
        if agreement < config.minimum_agreement:
            return self._flat(
                inferred_ticker,
                "Gli specialisti sono in conflitto: NO_TRADE.",
            )

        direction = Direction.LONG if signed_vote > 0.0 else Direction.SHORT
        horizon_bars = max(
            1,
            round(
                sum(
                    weight * forecast.horizon_bars
                    for weight, forecast in zip(raw_weights, active)
                )
                / total_weight
            ),
        )
        expected_return_per_bar = sum(
            weight * forecast.expected_return / forecast.horizon_bars
            for weight, forecast in zip(raw_weights, active)
        ) / total_weight
        expected_return = expected_return_per_bar * horizon_bars
        if expected_return * direction.sign <= 0.0:
            return self._flat(
                inferred_ticker,
                "Il rendimento aggregato non conferma la direzione del voto.",
            )

        expected_volatility = sum(
            weight * forecast.expected_volatility
            for weight, forecast in zip(raw_weights, active)
        ) / total_weight
        average_confidence = sum(
            weight * forecast.confidence
            for weight, forecast in zip(raw_weights, active)
        ) / total_weight
        confidence = float(
            np.clip(
                average_confidence
                * agreement
                * (0.75 + 0.25 * regime.confidence),
                0.0,
                1.0,
            )
        )

        if confidence < config.minimum_confidence:
            return self._flat(inferred_ticker, "Confidence aggregata insufficiente.")
        if abs(expected_return) * 10_000.0 < config.minimum_expected_edge_bps:
            return self._flat(inferred_ticker, "Edge lordo previsto troppo piccolo.")

        contributors = {
            forecast.strategy_id: forecast.direction.sign * weight / total_weight
            for weight, forecast in zip(raw_weights, active)
        }
        return MetaDecision(
            ticker=inferred_ticker,
            direction=direction,
            confidence=confidence,
            expected_return=expected_return,
            expected_volatility=expected_volatility,
            agreement=agreement,
            horizon_bars=horizon_bars,
            contributors=contributors,
            reasons=(
                f"{len(active)} specialisti attivi.",
                "Direzione confermata dal rendimento atteso aggregato.",
            ),
        )
