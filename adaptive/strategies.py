"""Prime strategie specialistiche compatibili con il meta-modello."""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from adaptive.models import (
    Direction,
    FeatureSnapshot,
    MarketRegime,
    RegimeAssessment,
    StrategyForecast,
)


class AdaptiveStrategy(Protocol):
    strategy_id: str

    def forecast(
        self, snapshot: FeatureSnapshot, regime: RegimeAssessment
    ) -> StrategyForecast: ...


def _probability(regime: RegimeAssessment, key: MarketRegime) -> float:
    return float(regime.probabilities.get(key, 0.0))


def _flat(
    strategy_id: str,
    snapshot: FeatureSnapshot,
    horizon_bars: int,
    reason: str,
) -> StrategyForecast:
    return StrategyForecast(
        strategy_id=strategy_id,
        ticker=snapshot.ticker,
        timestamp=snapshot.timestamp,
        direction=Direction.FLAT,
        confidence=0.0,
        expected_return=0.0,
        expected_volatility=snapshot.realized_volatility_short,
        horizon_bars=horizon_bars,
        regime_fit=0.0,
        reasons=(reason,),
    )


class TrendSpecialist:
    strategy_id = "adaptive_trend"

    def __init__(self, minimum_trend_score: float = 0.35, horizon_bars: int = 21):
        self.minimum_trend_score = float(minimum_trend_score)
        self.horizon_bars = int(horizon_bars)
        if not 0.0 <= self.minimum_trend_score <= 1.0:
            raise ValueError("minimum_trend_score deve essere compreso tra 0 e 1.")
        if self.horizon_bars <= 0:
            raise ValueError("horizon_bars deve essere positivo.")

    def forecast(
        self, snapshot: FeatureSnapshot, regime: RegimeAssessment
    ) -> StrategyForecast:
        strength = abs(snapshot.trend_score)
        if strength < self.minimum_trend_score:
            return _flat(
                self.strategy_id,
                snapshot,
                self.horizon_bars,
                "Trend troppo debole.",
            )

        direction = (
            Direction.LONG if snapshot.trend_score > 0.0 else Direction.SHORT
        )
        relevant_regime = (
            MarketRegime.TREND_UP
            if direction is Direction.LONG
            else MarketRegime.TREND_DOWN
        )
        regime_fit = float(
            np.clip(
                0.30 + 1.50 * _probability(regime, relevant_regime),
                0.0,
                1.0,
            )
        )
        confidence = float(
            np.clip(
                (0.25 + 0.75 * strength)
                * (0.60 + 0.40 * regime_fit)
                * snapshot.data_quality,
                0.0,
                1.0,
            )
        )
        daily_volatility = snapshot.realized_volatility_short / math.sqrt(252.0)
        magnitude = daily_volatility * math.sqrt(self.horizon_bars) * confidence * 0.70
        expected_return = direction.sign * magnitude

        return StrategyForecast(
            strategy_id=self.strategy_id,
            ticker=snapshot.ticker,
            timestamp=snapshot.timestamp,
            direction=direction,
            confidence=confidence,
            expected_return=expected_return,
            expected_volatility=snapshot.realized_volatility_short,
            horizon_bars=self.horizon_bars,
            regime_fit=regime_fit,
            reasons=("Direzione coerente con il trend multi-periodo.",),
            metadata={"trend_score": snapshot.trend_score},
        )


class MeanReversionSpecialist:
    strategy_id = "adaptive_mean_reversion"

    def __init__(self, entry_zscore: float = 1.50, horizon_bars: int = 5):
        self.entry_zscore = float(entry_zscore)
        self.horizon_bars = int(horizon_bars)
        if self.entry_zscore <= 0.0:
            raise ValueError("entry_zscore deve essere positivo.")
        if self.horizon_bars <= 0:
            raise ValueError("horizon_bars deve essere positivo.")

    def forecast(
        self, snapshot: FeatureSnapshot, regime: RegimeAssessment
    ) -> StrategyForecast:
        distance = abs(snapshot.return_zscore)
        regime_fit = float(
            np.clip(
                _probability(regime, MarketRegime.RANGE)
                + _probability(regime, MarketRegime.LOW_VOLATILITY),
                0.0,
                1.0,
            )
        )
        if distance < self.entry_zscore or regime_fit < 0.15:
            return _flat(
                self.strategy_id,
                snapshot,
                self.horizon_bars,
                "Deviazione o compatibilità col regime insufficienti.",
            )

        direction = (
            Direction.SHORT if snapshot.return_zscore > 0.0 else Direction.LONG
        )
        excess = min(1.0, (distance - self.entry_zscore) / 2.0)
        confidence = float(
            np.clip(
                (0.35 + 0.65 * excess)
                * (0.55 + 0.45 * regime_fit)
                * snapshot.data_quality
                * snapshot.liquidity_score,
                0.0,
                1.0,
            )
        )
        magnitude = min(
            abs(snapshot.return_5) * 0.60,
            snapshot.realized_volatility_short
            / math.sqrt(252.0)
            * math.sqrt(self.horizon_bars),
        )

        return StrategyForecast(
            strategy_id=self.strategy_id,
            ticker=snapshot.ticker,
            timestamp=snapshot.timestamp,
            direction=direction,
            confidence=confidence,
            expected_return=direction.sign * max(0.0, magnitude),
            expected_volatility=snapshot.realized_volatility_short,
            horizon_bars=self.horizon_bars,
            regime_fit=regime_fit,
            reasons=("Ritorno estremo in regime laterale o di bassa volatilità.",),
            metadata={"return_zscore": snapshot.return_zscore},
        )


class VolatilityBreakoutSpecialist:
    strategy_id = "adaptive_volatility_breakout"

    def __init__(self, minimum_shock_score: float = 2.0, horizon_bars: int = 5):
        self.minimum_shock_score = float(minimum_shock_score)
        self.horizon_bars = int(horizon_bars)
        if self.minimum_shock_score <= 0.0:
            raise ValueError("minimum_shock_score deve essere positivo.")
        if self.horizon_bars <= 0:
            raise ValueError("horizon_bars deve essere positivo.")

    def forecast(
        self, snapshot: FeatureSnapshot, regime: RegimeAssessment
    ) -> StrategyForecast:
        regime_fit = float(
            np.clip(
                _probability(regime, MarketRegime.HIGH_VOLATILITY)
                + _probability(regime, MarketRegime.SHOCK),
                0.0,
                1.0,
            )
        )
        if (
            snapshot.shock_score < self.minimum_shock_score
            or math.isclose(snapshot.return_1, 0.0, abs_tol=1e-15)
            or snapshot.liquidity_score < 0.35
        ):
            return _flat(
                self.strategy_id,
                snapshot,
                self.horizon_bars,
                "Breakout non confermato o liquidità insufficiente.",
            )

        direction = Direction.LONG if snapshot.return_1 > 0.0 else Direction.SHORT
        confirmation = float(
            np.clip(
                0.5
                + 0.15 * snapshot.volume_zscore
                + 0.10 * max(0.0, snapshot.shock_score - 2.0),
                0.0,
                1.0,
            )
        )
        confidence = float(
            np.clip(
                confirmation
                * (0.60 + 0.40 * regime_fit)
                * snapshot.data_quality
                * snapshot.liquidity_score,
                0.0,
                1.0,
            )
        )
        magnitude = abs(snapshot.return_1) * confidence * 0.75

        return StrategyForecast(
            strategy_id=self.strategy_id,
            ticker=snapshot.ticker,
            timestamp=snapshot.timestamp,
            direction=direction,
            confidence=confidence,
            expected_return=direction.sign * magnitude,
            expected_volatility=snapshot.realized_volatility_short,
            horizon_bars=self.horizon_bars,
            regime_fit=regime_fit,
            reasons=("Espansione di volatilità con conferma di direzione.",),
            metadata={
                "shock_score": snapshot.shock_score,
                "volume_zscore": snapshot.volume_zscore,
            },
        )


DEFAULT_ADAPTIVE_STRATEGIES: tuple[AdaptiveStrategy, ...] = (
    TrendSpecialist(),
    MeanReversionSpecialist(),
    VolatilityBreakoutSpecialist(),
)
