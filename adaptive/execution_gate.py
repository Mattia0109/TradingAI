"""Filtro economico pre-trade; non invia ordini a un broker."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from adaptive.models import (
    ExecutionDecision,
    ExecutionEstimate,
    FeatureSnapshot,
    MetaDecision,
    RiskDecision,
)


@dataclass(frozen=True)
class ExecutionGateConfig:
    minimum_net_edge_bps: float = 5.0
    minimum_edge_cost_ratio: float = 1.50
    maximum_spread_bps: float = 50.0
    maximum_participation_rate: float = 0.02
    maximum_latency_ms: float = 2_000.0
    minimum_liquidity_score: float = 0.25
    maximum_snapshot_age_seconds: float = 300.0

    def __post_init__(self) -> None:
        non_negative = (
            "minimum_net_edge_bps",
            "maximum_spread_bps",
            "maximum_latency_ms",
            "maximum_snapshot_age_seconds",
        )
        for name in non_negative:
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} non può essere negativo.")
        if self.minimum_edge_cost_ratio < 0.0:
            raise ValueError("minimum_edge_cost_ratio non può essere negativo.")
        for name in ("maximum_participation_rate", "minimum_liquidity_score"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} deve essere compreso tra 0 e 1.")


class EconomicExecutionGate:
    def __init__(self, config: ExecutionGateConfig | None = None) -> None:
        self.config = config or ExecutionGateConfig()

    @staticmethod
    def _decision(
        approved: bool,
        edge: float,
        cost: float,
        order_style: str,
        reasons: list[str],
    ) -> ExecutionDecision:
        return ExecutionDecision(
            approved=approved,
            expected_edge_bps=max(0.0, edge),
            estimated_cost_bps=max(0.0, cost),
            net_edge_bps=edge - cost,
            order_style=order_style,
            reasons=tuple(reasons),
        )

    def evaluate(
        self,
        meta: MetaDecision,
        risk: RiskDecision,
        snapshot: FeatureSnapshot,
        estimate: ExecutionEstimate,
        *,
        as_of: datetime | None = None,
    ) -> ExecutionDecision:
        config = self.config
        edge_bps = abs(meta.expected_return) * 10_000.0
        effective_spread = max(estimate.spread_bps, snapshot.spread_bps or 0.0)
        estimated_cost_bps = (
            estimate.round_trip_cost_bps - estimate.spread_bps + effective_spread
        )
        order_style = "MARKETABLE_LIMIT" if effective_spread <= 5.0 else "PASSIVE_LIMIT"
        reasons: list[str] = []

        if not risk.approved:
            reasons.append("Risk Engine non ha autorizzato il target.")
        if snapshot.liquidity_score < config.minimum_liquidity_score:
            reasons.append("Liquidità sotto la soglia di esecuzione.")
        if effective_spread > config.maximum_spread_bps:
            reasons.append("Spread superiore al massimo ammesso.")
        if estimate.participation_rate > config.maximum_participation_rate:
            reasons.append("Partecipazione al volume troppo elevata.")
        if estimate.latency_ms > config.maximum_latency_ms:
            reasons.append("Latenza incompatibile con il segnale.")

        if as_of is not None:
            def utc(value: datetime) -> datetime:
                if value.tzinfo is None:
                    return value.replace(tzinfo=timezone.utc)
                return value.astimezone(timezone.utc)

            age = max(0.0, (utc(as_of) - utc(snapshot.timestamp)).total_seconds())
            if age > config.maximum_snapshot_age_seconds:
                reasons.append("Snapshot di mercato scaduto.")

        net_edge_bps = edge_bps - estimated_cost_bps
        edge_cost_ratio = (
            edge_bps / estimated_cost_bps
            if estimated_cost_bps > 0.0
            else float("inf")
        )
        if net_edge_bps < config.minimum_net_edge_bps:
            reasons.append("Edge netto inferiore alla soglia minima.")
        if edge_cost_ratio < config.minimum_edge_cost_ratio:
            reasons.append("Rapporto edge/costi insufficiente.")

        if reasons:
            return self._decision(
                False,
                edge_bps,
                estimated_cost_bps,
                "NO_ORDER",
                reasons,
            )

        return self._decision(
            True,
            edge_bps,
            estimated_cost_bps,
            order_style,
            ["Edge netto sufficiente dopo tutti i costi stimati."],
        )
