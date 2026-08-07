"""Contratti immutabili condivisi dal sistema adattivo.

Le percentuali sono espresse come frazioni decimali: 0.01 equivale all'1%.
Confidence, qualità e compatibilità sono sempre comprese tra 0 e 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"

    @property
    def sign(self) -> int:
        if self is Direction.LONG:
            return 1
        if self is Direction.SHORT:
            return -1
        return 0


class MarketRegime(str, Enum):
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    SHOCK = "SHOCK"
    CORRELATION_SPIKE = "CORRELATION_SPIKE"
    LIQUIDITY_CRISIS = "LIQUIDITY_CRISIS"
    UNKNOWN = "UNKNOWN"


class CycleStatus(str, Enum):
    NO_TRADE = "NO_TRADE"
    RISK_REJECTED = "RISK_REJECTED"
    EXECUTION_REJECTED = "EXECUTION_REJECTED"
    PAPER_APPROVED = "PAPER_APPROVED"


def _finite(name: str, value: float) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} deve essere finito.")
    return normalized


def _unit_interval(name: str, value: float) -> float:
    normalized = _finite(name, value)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} deve essere compreso tra 0 e 1.")
    return normalized


def _non_negative(name: str, value: float) -> float:
    normalized = _finite(name, value)
    if normalized < 0.0:
        raise ValueError(f"{name} non può essere negativo.")
    return normalized


def _ticker(value: str) -> str:
    normalized = str(value).upper().strip()
    if not normalized:
        raise ValueError("ticker non può essere vuoto.")
    return normalized


@dataclass(frozen=True)
class FeatureSnapshot:
    ticker: str
    asset_class: str
    timestamp: datetime
    price: float
    return_1: float
    return_5: float
    return_21: float
    realized_volatility_short: float
    realized_volatility_long: float
    trend_score: float
    return_zscore: float
    shock_score: float
    volume_zscore: float = 0.0
    spread_bps: float | None = None
    liquidity_score: float = 0.5
    market_correlation: float | None = None
    news_sentiment: float = 0.0
    news_relevance: float = 0.0
    data_quality: float = 1.0
    extras: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "ticker", _ticker(self.ticker))
        normalized_class = str(self.asset_class).upper().strip() or "UNKNOWN"
        object.__setattr__(self, "asset_class", normalized_class)

        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp deve essere datetime.")

        price = _finite("price", self.price)
        if price <= 0.0:
            raise ValueError("price deve essere positivo.")
        object.__setattr__(self, "price", price)

        finite_fields = (
            "return_1",
            "return_5",
            "return_21",
            "realized_volatility_short",
            "realized_volatility_long",
            "trend_score",
            "return_zscore",
            "shock_score",
            "volume_zscore",
            "news_sentiment",
        )
        for name in finite_fields:
            object.__setattr__(self, name, _finite(name, getattr(self, name)))

        if self.realized_volatility_short < 0.0:
            raise ValueError("realized_volatility_short non può essere negativa.")
        if self.realized_volatility_long < 0.0:
            raise ValueError("realized_volatility_long non può essere negativa.")
        if self.shock_score < 0.0:
            raise ValueError("shock_score non può essere negativo.")
        if not -1.0 <= self.trend_score <= 1.0:
            raise ValueError("trend_score deve essere compreso tra -1 e 1.")
        if not -1.0 <= self.news_sentiment <= 1.0:
            raise ValueError("news_sentiment deve essere compreso tra -1 e 1.")

        object.__setattr__(
            self,
            "liquidity_score",
            _unit_interval("liquidity_score", self.liquidity_score),
        )
        object.__setattr__(
            self,
            "news_relevance",
            _unit_interval("news_relevance", self.news_relevance),
        )
        object.__setattr__(
            self,
            "data_quality",
            _unit_interval("data_quality", self.data_quality),
        )

        if self.spread_bps is not None:
            object.__setattr__(
                self,
                "spread_bps",
                _non_negative("spread_bps", self.spread_bps),
            )
        if self.market_correlation is not None:
            correlation = _finite("market_correlation", self.market_correlation)
            if not -1.0 <= correlation <= 1.0:
                raise ValueError(
                    "market_correlation deve essere compresa tra -1 e 1."
                )
            object.__setattr__(self, "market_correlation", correlation)


@dataclass(frozen=True)
class RegimeAssessment:
    primary: MarketRegime
    probabilities: Mapping[MarketRegime, float]
    confidence: float
    risk_multiplier: float
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "confidence", _unit_interval("confidence", self.confidence)
        )
        object.__setattr__(
            self,
            "risk_multiplier",
            _unit_interval("risk_multiplier", self.risk_multiplier),
        )
        normalized = {
            MarketRegime(regime): _unit_interval(
                f"probability[{regime}]", probability
            )
            for regime, probability in self.probabilities.items()
        }
        if not normalized:
            raise ValueError("Serve almeno una probabilità di regime.")
        total = sum(normalized.values())
        if normalized and not math.isclose(total, 1.0, abs_tol=1e-8):
            raise ValueError("Le probabilità di regime devono sommare a 1.")
        object.__setattr__(self, "probabilities", normalized)


@dataclass(frozen=True)
class StrategyForecast:
    strategy_id: str
    ticker: str
    timestamp: datetime
    direction: Direction
    confidence: float
    expected_return: float
    expected_volatility: float
    horizon_bars: int
    regime_fit: float
    reasons: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        strategy_id = str(self.strategy_id).strip()
        if not strategy_id:
            raise ValueError("strategy_id non può essere vuoto.")
        object.__setattr__(self, "strategy_id", strategy_id)
        object.__setattr__(self, "ticker", _ticker(self.ticker))
        object.__setattr__(self, "direction", Direction(self.direction))
        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp deve essere datetime.")
        object.__setattr__(
            self, "confidence", _unit_interval("confidence", self.confidence)
        )
        object.__setattr__(
            self, "regime_fit", _unit_interval("regime_fit", self.regime_fit)
        )
        expected_return = _finite("expected_return", self.expected_return)
        expected_volatility = _non_negative(
            "expected_volatility", self.expected_volatility
        )
        object.__setattr__(self, "expected_return", expected_return)
        object.__setattr__(self, "expected_volatility", expected_volatility)
        if int(self.horizon_bars) <= 0:
            raise ValueError("horizon_bars deve essere positivo.")
        object.__setattr__(self, "horizon_bars", int(self.horizon_bars))

        if self.direction is Direction.LONG and expected_return < 0.0:
            raise ValueError("Un forecast LONG non può avere rendimento atteso negativo.")
        if self.direction is Direction.SHORT and expected_return > 0.0:
            raise ValueError("Un forecast SHORT non può avere rendimento atteso positivo.")
        if self.direction is Direction.FLAT and not math.isclose(
            expected_return, 0.0, abs_tol=1e-15
        ):
            raise ValueError("Un forecast FLAT deve avere rendimento atteso nullo.")


@dataclass(frozen=True)
class MetaDecision:
    ticker: str
    direction: Direction
    confidence: float
    expected_return: float
    expected_volatility: float
    agreement: float
    horizon_bars: int = 1
    contributors: Mapping[str, float] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "ticker", _ticker(self.ticker))
        object.__setattr__(self, "direction", Direction(self.direction))
        object.__setattr__(
            self, "confidence", _unit_interval("confidence", self.confidence)
        )
        object.__setattr__(
            self, "agreement", _unit_interval("agreement", self.agreement)
        )
        object.__setattr__(
            self, "expected_return", _finite("expected_return", self.expected_return)
        )
        object.__setattr__(
            self,
            "expected_volatility",
            _non_negative("expected_volatility", self.expected_volatility),
        )
        if int(self.horizon_bars) <= 0:
            raise ValueError("horizon_bars deve essere positivo.")
        object.__setattr__(self, "horizon_bars", int(self.horizon_bars))

        if self.direction is Direction.LONG and self.expected_return < 0.0:
            raise ValueError("Una meta-decisione LONG richiede edge non negativo.")
        if self.direction is Direction.SHORT and self.expected_return > 0.0:
            raise ValueError("Una meta-decisione SHORT richiede edge non positivo.")
        if self.direction is Direction.FLAT and not math.isclose(
            self.expected_return, 0.0, abs_tol=1e-15
        ):
            raise ValueError("Una meta-decisione FLAT deve avere edge nullo.")

    @property
    def is_no_trade(self) -> bool:
        return self.direction is Direction.FLAT


@dataclass(frozen=True)
class PortfolioState:
    equity: float
    peak_equity: float
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    leverage: float = 1.0
    asset_class_exposure: Mapping[str, float] = field(default_factory=dict)
    current_weights: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        equity = _finite("equity", self.equity)
        peak = _finite("peak_equity", self.peak_equity)
        if equity <= 0.0 or peak <= 0.0:
            raise ValueError("equity e peak_equity devono essere positive.")
        object.__setattr__(self, "equity", equity)
        object.__setattr__(self, "peak_equity", peak)
        object.__setattr__(
            self,
            "gross_exposure",
            _non_negative("gross_exposure", self.gross_exposure),
        )
        object.__setattr__(self, "net_exposure", _finite("net_exposure", self.net_exposure))
        object.__setattr__(self, "leverage", _non_negative("leverage", self.leverage))
        normalized_classes = {
            str(asset_class).upper().strip(): _non_negative(
                f"asset_class_exposure[{asset_class}]",
                exposure,
            )
            for asset_class, exposure in self.asset_class_exposure.items()
            if str(asset_class).strip()
        }
        normalized_weights = {
            _ticker(ticker): _finite(f"current_weights[{ticker}]", weight)
            for ticker, weight in self.current_weights.items()
        }
        object.__setattr__(self, "asset_class_exposure", normalized_classes)
        object.__setattr__(self, "current_weights", normalized_weights)

    @property
    def drawdown(self) -> float:
        effective_peak = max(self.peak_equity, self.equity)
        return max(0.0, 1.0 - self.equity / effective_peak)


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    target_weight: float
    allowed_leverage: float
    risk_multiplier: float
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "target_weight", _finite("target_weight", self.target_weight)
        )
        object.__setattr__(
            self,
            "allowed_leverage",
            _non_negative("allowed_leverage", self.allowed_leverage),
        )
        object.__setattr__(
            self,
            "risk_multiplier",
            _unit_interval("risk_multiplier", self.risk_multiplier),
        )
        if not self.approved and not math.isclose(
            self.target_weight, 0.0, abs_tol=1e-15
        ):
            raise ValueError("Una decisione rischio rifiutata deve avere target nullo.")
        if self.approved and math.isclose(self.target_weight, 0.0, abs_tol=1e-15):
            raise ValueError("Una decisione rischio approvata richiede target non nullo.")
        if self.approved and self.risk_multiplier <= 0.0:
            raise ValueError("Una decisione approvata richiede rischio positivo.")
        if self.allowed_leverage < 1.0:
            raise ValueError("allowed_leverage non può essere inferiore a 1.")


@dataclass(frozen=True)
class ExecutionEstimate:
    spread_bps: float = 0.0
    fees_bps_per_side: float = 0.0
    slippage_bps_per_side: float = 0.0
    market_impact_bps_per_side: float = 0.0
    participation_rate: float = 0.0
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "spread_bps",
            "fees_bps_per_side",
            "slippage_bps_per_side",
            "market_impact_bps_per_side",
            "participation_rate",
            "latency_ms",
        ):
            object.__setattr__(self, name, _non_negative(name, getattr(self, name)))
        if self.participation_rate > 1.0:
            raise ValueError("participation_rate non può superare 1.")

    @property
    def round_trip_cost_bps(self) -> float:
        return (
            self.spread_bps
            + 2.0 * self.fees_bps_per_side
            + 2.0 * self.slippage_bps_per_side
            + 2.0 * self.market_impact_bps_per_side
        )


@dataclass(frozen=True)
class ExecutionDecision:
    approved: bool
    expected_edge_bps: float
    estimated_cost_bps: float
    net_edge_bps: float
    order_style: str
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "expected_edge_bps",
            _non_negative("expected_edge_bps", self.expected_edge_bps),
        )
        object.__setattr__(
            self,
            "estimated_cost_bps",
            _non_negative("estimated_cost_bps", self.estimated_cost_bps),
        )
        object.__setattr__(
            self,
            "net_edge_bps",
            _finite("net_edge_bps", self.net_edge_bps),
        )
        if not math.isclose(
            self.net_edge_bps,
            self.expected_edge_bps - self.estimated_cost_bps,
            abs_tol=1e-9,
        ):
            raise ValueError("net_edge_bps non coincide con edge meno costi.")
        order_style = str(self.order_style).upper().strip()
        if not order_style:
            raise ValueError("order_style non può essere vuoto.")
        object.__setattr__(self, "order_style", order_style)
        if self.approved and order_style == "NO_ORDER":
            raise ValueError("Un'esecuzione approvata richiede uno stile d'ordine.")
        if not self.approved and order_style != "NO_ORDER":
            raise ValueError("Un'esecuzione rifiutata deve usare NO_ORDER.")


@dataclass(frozen=True)
class AdaptiveCycleResult:
    cycle_id: str
    snapshot: FeatureSnapshot
    regime: RegimeAssessment
    forecasts: tuple[StrategyForecast, ...]
    meta_decision: MetaDecision
    risk_decision: RiskDecision
    execution_decision: ExecutionDecision
    status: CycleStatus
    paper_only: bool = True
    strategy_errors: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.cycle_id).strip():
            raise ValueError("cycle_id non può essere vuoto.")
        object.__setattr__(self, "status", CycleStatus(self.status))
        if not self.paper_only:
            raise ValueError("La fondazione adaptive deve restare paper-only.")
        if self.meta_decision.ticker != self.snapshot.ticker:
            raise ValueError("Snapshot e meta-decisione hanno ticker differenti.")
        if any(item.ticker != self.snapshot.ticker for item in self.forecasts):
            raise ValueError("Tutti i forecast devono appartenere allo snapshot.")
        if self.meta_decision.is_no_trade and self.risk_decision.approved:
            raise ValueError("NO_TRADE non può avere un target rischio approvato.")
        if not self.risk_decision.approved and self.execution_decision.approved:
            raise ValueError("Execution non può superare un rifiuto del Risk Engine.")
        if (
            self.risk_decision.approved
            and self.risk_decision.target_weight * self.meta_decision.direction.sign
            <= 0.0
        ):
            raise ValueError("Target rischio contrario alla meta-decisione.")

        if self.meta_decision.is_no_trade:
            expected_status = CycleStatus.NO_TRADE
        elif not self.risk_decision.approved:
            expected_status = CycleStatus.RISK_REJECTED
        elif not self.execution_decision.approved:
            expected_status = CycleStatus.EXECUTION_REJECTED
        else:
            expected_status = CycleStatus.PAPER_APPROVED
        if self.status is not expected_status:
            raise ValueError(
                "status incoerente con meta, rischio ed execution gate."
            )
