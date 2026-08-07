"""Rilevazione trasparente e multi-etichetta del regime di mercato."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adaptive.models import FeatureSnapshot, MarketRegime, RegimeAssessment


@dataclass(frozen=True)
class RegimeDetectorConfig:
    trend_threshold: float = 0.35
    high_volatility_ratio: float = 1.40
    low_volatility_ratio: float = 0.70
    shock_threshold: float = 3.0
    severe_shock_threshold: float = 5.0
    correlation_spike_threshold: float = 0.80
    liquidity_crisis_threshold: float = 0.20
    maximum_spread_bps: float = 100.0
    minimum_data_quality: float = 0.60

    def __post_init__(self) -> None:
        if not 0.0 <= self.trend_threshold <= 1.0:
            raise ValueError("trend_threshold deve essere compreso tra 0 e 1.")
        if self.high_volatility_ratio <= 0.0 or self.low_volatility_ratio <= 0.0:
            raise ValueError("I rapporti di volatilità devono essere positivi.")
        if self.low_volatility_ratio >= self.high_volatility_ratio:
            raise ValueError("Le soglie di volatilità non sono ordinate.")
        if self.shock_threshold <= 0.0:
            raise ValueError("shock_threshold deve essere positivo.")
        if self.severe_shock_threshold < self.shock_threshold:
            raise ValueError("severe_shock_threshold deve seguire shock_threshold.")
        for name in (
            "correlation_spike_threshold",
            "liquidity_crisis_threshold",
            "minimum_data_quality",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} deve essere compreso tra 0 e 1.")
        if self.maximum_spread_bps <= 0.0:
            raise ValueError("maximum_spread_bps deve essere positivo.")


class MarketRegimeDetector:
    """Assegna probabilità relative e un moltiplicatore prudenziale di rischio."""

    def __init__(self, config: RegimeDetectorConfig | None = None) -> None:
        self.config = config or RegimeDetectorConfig()

    @staticmethod
    def _normalize(scores: dict[MarketRegime, float]) -> dict[MarketRegime, float]:
        sanitized = {regime: max(0.01, float(score)) for regime, score in scores.items()}
        total = sum(sanitized.values())
        return {regime: score / total for regime, score in sanitized.items()}

    def detect(self, snapshot: FeatureSnapshot) -> RegimeAssessment:
        config = self.config
        trend = snapshot.trend_score
        volatility_ratio = snapshot.realized_volatility_short / max(
            snapshot.realized_volatility_long, 1e-12
        )
        correlation = abs(snapshot.market_correlation or 0.0)
        spread_pressure = (
            snapshot.spread_bps / config.maximum_spread_bps
            if snapshot.spread_bps is not None
            else 0.0
        )

        scores = {
            MarketRegime.TREND_UP: 0.05
            + max(0.0, trend) * 1.8
            + max(0.0, snapshot.return_21) * 8.0,
            MarketRegime.TREND_DOWN: 0.05
            + max(0.0, -trend) * 1.8
            + max(0.0, -snapshot.return_21) * 8.0,
            MarketRegime.RANGE: 0.05 + max(0.0, 1.0 - abs(trend)),
            MarketRegime.HIGH_VOLATILITY: 0.05
            + max(0.0, volatility_ratio - 1.0) * 1.8
            + min(snapshot.realized_volatility_short / 0.50, 1.0) * 0.25,
            MarketRegime.LOW_VOLATILITY: 0.05
            + max(0.0, 1.0 - volatility_ratio) * 1.5
            + max(0.0, 0.15 - snapshot.realized_volatility_short) * 2.0,
            MarketRegime.SHOCK: 0.05
            + max(0.0, snapshot.shock_score - 1.5) * 0.8,
            MarketRegime.CORRELATION_SPIKE: 0.05
            + max(0.0, correlation - 0.50) * 3.0,
            MarketRegime.LIQUIDITY_CRISIS: 0.05
            + max(0.0, 0.50 - snapshot.liquidity_score) * 3.0
            + max(0.0, spread_pressure - 0.50) * 1.5,
        }
        probabilities = self._normalize(scores)
        ranked = sorted(
            probabilities.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        primary = ranked[0][0]
        top_probability = ranked[0][1]
        second_probability = ranked[1][1]
        confidence = float(
            np.clip(
                0.5 * top_probability
                + 0.5
                * (top_probability - second_probability)
                / max(top_probability, 1e-12),
                0.0,
                1.0,
            )
        )

        reasons: list[str] = [f"Regime principale: {primary.value}."]
        risk_multiplier = 1.0

        if snapshot.data_quality < config.minimum_data_quality:
            risk_multiplier = 0.0
            reasons.append("Qualità dati insufficiente: rischio azzerato.")

        if (
            snapshot.liquidity_score < config.liquidity_crisis_threshold
            or (
                snapshot.spread_bps is not None
                and snapshot.spread_bps > config.maximum_spread_bps
            )
        ):
            risk_multiplier = 0.0
            reasons.append("Crisi di liquidità o spread eccessivo.")

        if snapshot.shock_score >= config.severe_shock_threshold:
            risk_multiplier = min(risk_multiplier, 0.15)
            reasons.append("Shock severo: esposizione quasi azzerata.")
        elif snapshot.shock_score >= config.shock_threshold:
            risk_multiplier = min(risk_multiplier, 0.40)
            reasons.append("Shock rilevato: esposizione ridotta.")

        if volatility_ratio >= config.high_volatility_ratio:
            risk_multiplier = min(risk_multiplier, 0.60)
            reasons.append("Volatilità di breve sopra il regime di lungo periodo.")

        if correlation >= config.correlation_spike_threshold:
            risk_multiplier = min(risk_multiplier, 0.65)
            reasons.append("Correlazione elevata: diversificazione meno efficace.")

        if abs(trend) >= config.trend_threshold:
            reasons.append("Trend statisticamente rilevante nelle feature.")

        return RegimeAssessment(
            primary=primary,
            probabilities=probabilities,
            confidence=confidence,
            risk_multiplier=risk_multiplier,
            reasons=tuple(reasons),
        )
