"""Autorità di rischio superiore a strategie e meta-modello."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adaptive.models import (
    FeatureSnapshot,
    MetaDecision,
    PortfolioState,
    RegimeAssessment,
    RiskDecision,
)


@dataclass(frozen=True)
class AdaptiveRiskPolicy:
    target_portfolio_volatility: float = 0.10
    max_asset_weight: float = 0.10
    max_asset_class_exposure: float = 0.30
    max_gross_exposure: float = 1.0
    minimum_target_weight: float = 0.005
    minimum_data_quality: float = 0.70
    minimum_liquidity_score: float = 0.25
    correlation_penalty_start: float = 0.70
    soft_drawdown: float = 0.10
    kill_switch_drawdown: float = 0.20
    allow_leverage: bool = False
    maximum_leverage: float = 1.0

    def __post_init__(self) -> None:
        unit_fields = (
            "target_portfolio_volatility",
            "max_asset_weight",
            "max_asset_class_exposure",
            "max_gross_exposure",
            "minimum_target_weight",
            "minimum_data_quality",
            "minimum_liquidity_score",
            "correlation_penalty_start",
            "soft_drawdown",
            "kill_switch_drawdown",
        )
        for name in unit_fields:
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} deve essere compreso tra 0 e 1.")
        if self.soft_drawdown >= self.kill_switch_drawdown:
            raise ValueError("soft_drawdown deve precedere il kill switch.")
        if self.maximum_leverage < 1.0:
            raise ValueError("maximum_leverage non può essere inferiore a 1.")
        if not self.allow_leverage and self.maximum_leverage != 1.0:
            raise ValueError(
                "maximum_leverage deve essere 1 quando la leva è disabilitata."
            )


class AdaptiveRiskEngine:
    def __init__(self, policy: AdaptiveRiskPolicy | None = None) -> None:
        self.policy = policy or AdaptiveRiskPolicy()

    @staticmethod
    def _reject(reason: str, leverage: float = 1.0) -> RiskDecision:
        return RiskDecision(
            approved=False,
            target_weight=0.0,
            allowed_leverage=leverage,
            risk_multiplier=0.0,
            reasons=(reason,),
        )

    def _drawdown_multiplier(self, drawdown: float) -> float:
        policy = self.policy
        if drawdown >= policy.kill_switch_drawdown:
            return 0.0
        if drawdown <= policy.soft_drawdown:
            return 1.0
        progress = (drawdown - policy.soft_drawdown) / (
            policy.kill_switch_drawdown - policy.soft_drawdown
        )
        return max(0.20, 1.0 - 0.80 * progress)

    def _allowed_leverage(
        self,
        meta: MetaDecision,
        snapshot: FeatureSnapshot,
        regime: RegimeAssessment,
    ) -> float:
        policy = self.policy
        if not policy.allow_leverage:
            return 1.0
        correlation = abs(snapshot.market_correlation or 0.0)
        quality = (
            meta.confidence
            * snapshot.liquidity_score
            * snapshot.data_quality
            * regime.risk_multiplier
            * max(0.0, 1.0 - correlation)
        )
        return 1.0 + (policy.maximum_leverage - 1.0) * quality

    def evaluate(
        self,
        meta: MetaDecision,
        snapshot: FeatureSnapshot,
        regime: RegimeAssessment,
        portfolio: PortfolioState,
    ) -> RiskDecision:
        policy = self.policy
        if meta.is_no_trade:
            return self._reject("Il meta-modello ha scelto NO_TRADE.")
        if meta.ticker != snapshot.ticker:
            raise ValueError("Meta-decision e snapshot appartengono a ticker diversi.")
        if portfolio.drawdown >= policy.kill_switch_drawdown:
            return self._reject("Kill switch di drawdown attivo.")
        if snapshot.data_quality < policy.minimum_data_quality:
            return self._reject("Qualità dati sotto la soglia di rischio.")
        if snapshot.liquidity_score < policy.minimum_liquidity_score:
            return self._reject("Liquidità insufficiente.")
        if regime.risk_multiplier <= 0.0:
            return self._reject("Il regime vieta nuova esposizione.")
        if (
            snapshot.news_relevance >= 0.85
            and snapshot.news_sentiment * meta.direction.sign <= -0.40
        ):
            return self._reject("Evento rilevante contrario alla direzione proposta.")

        allowed_leverage = self._allowed_leverage(meta, snapshot, regime)
        drawdown_multiplier = self._drawdown_multiplier(portfolio.drawdown)
        correlation = abs(snapshot.market_correlation or 0.0)
        correlation_multiplier = 1.0
        if correlation > policy.correlation_penalty_start:
            correlation_multiplier = max(
                0.25,
                1.0
                - (correlation - policy.correlation_penalty_start)
                / max(1.0 - policy.correlation_penalty_start, 1e-12)
                * 0.75,
            )
        event_multiplier = 0.70 if snapshot.news_relevance >= 0.85 else 1.0
        liquidity_multiplier = 0.50 + 0.50 * snapshot.liquidity_score
        risk_multiplier = float(
            np.clip(
                regime.risk_multiplier
                * drawdown_multiplier
                * correlation_multiplier
                * event_multiplier
                * liquidity_multiplier,
                0.0,
                1.0,
            )
        )

        effective_volatility = max(
            meta.expected_volatility,
            snapshot.realized_volatility_short,
            0.01,
        )
        volatility_scale = min(
            1.0,
            policy.target_portfolio_volatility / effective_volatility,
        )
        asset_limit = policy.max_asset_weight * allowed_leverage
        class_limit = policy.max_asset_class_exposure * allowed_leverage
        gross_limit = policy.max_gross_exposure * allowed_leverage

        current_weight = abs(float(portfolio.current_weights.get(snapshot.ticker, 0.0)))
        current_class = max(
            0.0,
            float(portfolio.asset_class_exposure.get(snapshot.asset_class, 0.0))
            - current_weight,
        )
        other_gross = max(0.0, portfolio.gross_exposure - current_weight)
        class_headroom = max(0.0, class_limit - current_class)
        gross_headroom = max(0.0, gross_limit - other_gross)

        target_magnitude = (
            asset_limit
            * meta.confidence
            * volatility_scale
            * risk_multiplier
        )
        target_magnitude = min(
            target_magnitude,
            asset_limit,
            class_headroom,
            gross_headroom,
        )
        if target_magnitude < policy.minimum_target_weight:
            return self._reject(
                "Target troppo piccolo dopo i vincoli di rischio.",
                leverage=allowed_leverage,
            )

        reasons = [
            "Target limitato da volatilità, regime, liquidità e confidence.",
            f"Leva massima dinamica: {allowed_leverage:.3f}x.",
        ]
        if drawdown_multiplier < 1.0:
            reasons.append("Esposizione ridotta dal drawdown governor.")
        if correlation_multiplier < 1.0:
            reasons.append("Esposizione ridotta per correlazione elevata.")

        return RiskDecision(
            approved=True,
            target_weight=meta.direction.sign * target_magnitude,
            allowed_leverage=allowed_leverage,
            risk_multiplier=risk_multiplier,
            reasons=tuple(reasons),
        )
