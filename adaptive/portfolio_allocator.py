"""Allocazione cross-asset successiva ai gate individuali."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from adaptive.models import AdaptiveCycleResult, CycleStatus, PortfolioState


@dataclass(frozen=True)
class PortfolioAllocatorConfig:
    max_gross_exposure: float = 1.0
    max_net_exposure: float = 0.60
    max_asset_class_exposure: float = 0.30
    max_asset_weight: float = 0.10
    minimum_target_weight: float = 0.005

    def __post_init__(self) -> None:
        for name in (
            "max_gross_exposure",
            "max_net_exposure",
            "max_asset_class_exposure",
            "max_asset_weight",
            "minimum_target_weight",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} deve essere compreso tra 0 e 1.")
        if self.minimum_target_weight > self.max_asset_weight:
            raise ValueError("Il target minimo supera il limite per asset.")


@dataclass(frozen=True)
class PortfolioAllocation:
    target_weights: Mapping[str, float] = field(default_factory=dict)
    unchanged_tickers: tuple[str, ...] = ()
    rejected: Mapping[str, str] = field(default_factory=dict)
    projected_gross_exposure: float = 0.0
    projected_net_exposure: float = 0.0
    allowed_leverage: float = 1.0
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for ticker, weight in self.target_weights.items():
            if not str(ticker).strip() or not math.isfinite(float(weight)):
                raise ValueError("Target di allocazione non valido.")
        for name in ("projected_gross_exposure", "allowed_leverage"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} non valido.")
        if not math.isfinite(float(self.projected_net_exposure)):
            raise ValueError("projected_net_exposure non valido.")
        if self.allowed_leverage < 1.0:
            raise ValueError("allowed_leverage non può essere inferiore a 1.")


class AdaptivePortfolioAllocator:
    """Seleziona i candidati migliori sotto vincoli cumulativi di portafoglio."""

    def __init__(self, config: PortfolioAllocatorConfig | None = None) -> None:
        self.config = config or PortfolioAllocatorConfig()

    @staticmethod
    def _priority(cycle: AdaptiveCycleResult) -> tuple[float, str]:
        score = (
            max(0.0, cycle.execution_decision.net_edge_bps)
            * cycle.meta_decision.confidence
            * cycle.meta_decision.agreement
            * cycle.risk_decision.risk_multiplier
        )
        return (-score, cycle.snapshot.ticker)

    def allocate(
        self,
        cycles: Sequence[AdaptiveCycleResult],
        portfolio: PortfolioState,
    ) -> PortfolioAllocation:
        tickers = [cycle.snapshot.ticker for cycle in cycles]
        if len(tickers) != len(set(tickers)):
            raise ValueError("L'allocatore ha ricevuto ticker duplicati.")
        eligible = [
            cycle
            for cycle in cycles
            if cycle.status is CycleStatus.PAPER_APPROVED
        ]
        rejected = {
            cycle.snapshot.ticker: f"Candidato non allocabile: {cycle.status.value}."
            for cycle in cycles
            if cycle.status is not CycleStatus.PAPER_APPROVED
        }
        unchanged_from_gates = [
            cycle.snapshot.ticker
            for cycle in cycles
            if cycle.status is not CycleStatus.PAPER_APPROVED
            and cycle.snapshot.ticker in portfolio.current_weights
        ]
        if not eligible:
            return PortfolioAllocation(
                unchanged_tickers=tuple(unchanged_from_gates),
                rejected=rejected,
                projected_gross_exposure=portfolio.gross_exposure,
                projected_net_exposure=portfolio.net_exposure,
                reasons=("Nessun candidato ha superato tutti i gate.",),
            )

        allowed_leverage = min(
            cycle.risk_decision.allowed_leverage for cycle in eligible
        )
        config = self.config
        gross_limit = config.max_gross_exposure * allowed_leverage
        net_limit = config.max_net_exposure * allowed_leverage
        class_limit = config.max_asset_class_exposure * allowed_leverage
        asset_limit = config.max_asset_weight * allowed_leverage

        running_gross = portfolio.gross_exposure
        running_net = portfolio.net_exposure
        running_classes = {
            str(asset_class).upper().strip(): max(0.0, float(exposure))
            for asset_class, exposure in portfolio.asset_class_exposure.items()
        }
        target_weights: dict[str, float] = {}
        unchanged: list[str] = list(unchanged_from_gates)

        for cycle in sorted(eligible, key=self._priority):
            ticker = cycle.snapshot.ticker
            asset_class = cycle.snapshot.asset_class
            proposed = cycle.risk_decision.target_weight
            sign = 1.0 if proposed > 0.0 else -1.0
            current = float(portfolio.current_weights.get(ticker, 0.0))

            other_gross = max(0.0, running_gross - abs(current))
            other_class = max(
                0.0,
                running_classes.get(asset_class, 0.0) - abs(current),
            )
            other_net = running_net - current
            net_headroom = (
                net_limit - other_net
                if sign > 0.0
                else net_limit + other_net
            )
            maximum_magnitude = min(
                abs(proposed),
                asset_limit,
                max(0.0, gross_limit - other_gross),
                max(0.0, class_limit - other_class),
                max(0.0, net_headroom),
            )

            if maximum_magnitude < config.minimum_target_weight:
                rejected[ticker] = (
                    "Capitale insufficiente dopo i limiti cumulativi di "
                    "portafoglio; posizione corrente invariata."
                )
                unchanged.append(ticker)
                continue

            target = sign * maximum_magnitude
            target_weights[ticker] = target
            running_gross = other_gross + abs(target)
            running_classes[asset_class] = other_class + abs(target)
            running_net = other_net + target

        reasons = [
            "Candidati ordinati per edge netto, confidence, agreement e rischio.",
            "Limiti gross, netti, per asset e per classe applicati cumulativamente.",
        ]
        if rejected:
            reasons.append("I candidati non allocati non modificano le posizioni correnti.")

        return PortfolioAllocation(
            target_weights=target_weights,
            unchanged_tickers=tuple(unchanged),
            rejected=rejected,
            projected_gross_exposure=running_gross,
            projected_net_exposure=running_net,
            allowed_leverage=allowed_leverage,
            reasons=tuple(reasons),
        )
