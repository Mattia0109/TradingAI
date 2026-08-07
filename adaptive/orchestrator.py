"""Orchestratore paper-only del ciclo decisionale adattivo.

Il modulo coordina componenti indipendenti, ma non contiene alcuna funzione che
possa inviare ordini a un broker. Una decisione ``PAPER_APPROVED`` indica solo
che il candidato ha superato tutti i filtri pre-trade.
"""

from __future__ import annotations

import math
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Mapping, Sequence

import pandas as pd

from adaptive.execution_gate import EconomicExecutionGate
from adaptive.feature_engine import AdaptiveFeatureEngine
from adaptive.journal import SQLiteDecisionJournal
from adaptive.learning import (
    ControlledPerformanceTracker,
    StrategyPerformanceState,
)
from adaptive.meta_model import AdaptiveMetaModel
from adaptive.news_intelligence import (
    ClassifiedNewsEvent,
    StructuredNewsIntelligence,
)
from adaptive.models import (
    AdaptiveCycleResult,
    CycleStatus,
    ExecutionEstimate,
    FeatureSnapshot,
    PortfolioState,
    StrategyForecast,
)
from adaptive.portfolio_allocator import (
    AdaptivePortfolioAllocator,
    PortfolioAllocation,
)
from adaptive.regime_detector import MarketRegimeDetector
from adaptive.risk_engine import AdaptiveRiskEngine
from adaptive.strategies import AdaptiveStrategy, DEFAULT_ADAPTIVE_STRATEGIES


@dataclass(frozen=True)
class AdaptiveSystemConfig:
    """Limiti operativi della fondazione, deliberatamente paper-only."""

    maximum_pending_cycles: int = 10_000

    def __post_init__(self) -> None:
        normalized = int(self.maximum_pending_cycles)
        if normalized <= 0 or normalized != self.maximum_pending_cycles:
            raise ValueError("maximum_pending_cycles deve essere positivo.")
        object.__setattr__(self, "maximum_pending_cycles", normalized)


@dataclass(frozen=True)
class UniverseAnalysisResult:
    """Risultati isolati per ticker: un errore non ferma l'intero universo."""

    cycles: Mapping[str, AdaptiveCycleResult]
    errors: Mapping[str, str]
    allocation: PortfolioAllocation


class AdaptiveTradingSystem:
    """Coordina feature, regimi, strategie, rischio, costi e apprendimento.

    Tutte le dipendenze sono iniettabili per rendere il sistema verificabile e
    per permettere di promuovere nuovi modelli solo dopo test indipendenti.
    """

    paper_only = True

    def __init__(
        self,
        *,
        feature_engine: AdaptiveFeatureEngine | None = None,
        regime_detector: MarketRegimeDetector | None = None,
        strategies: Sequence[AdaptiveStrategy] | None = None,
        meta_model: AdaptiveMetaModel | None = None,
        risk_engine: AdaptiveRiskEngine | None = None,
        execution_gate: EconomicExecutionGate | None = None,
        news_intelligence: StructuredNewsIntelligence | None = None,
        performance_tracker: ControlledPerformanceTracker | None = None,
        portfolio_allocator: AdaptivePortfolioAllocator | None = None,
        journal: SQLiteDecisionJournal | None = None,
        config: AdaptiveSystemConfig | None = None,
    ) -> None:
        self.feature_engine = feature_engine or AdaptiveFeatureEngine()
        self.regime_detector = regime_detector or MarketRegimeDetector()
        self.strategies = tuple(
            DEFAULT_ADAPTIVE_STRATEGIES if strategies is None else strategies
        )
        self.meta_model = meta_model or AdaptiveMetaModel()
        self.risk_engine = risk_engine or AdaptiveRiskEngine()
        self.execution_gate = execution_gate or EconomicExecutionGate()
        self.news_intelligence = news_intelligence or StructuredNewsIntelligence()
        self.performance_tracker = (
            performance_tracker or ControlledPerformanceTracker()
        )
        self.portfolio_allocator = portfolio_allocator or AdaptivePortfolioAllocator()
        self.journal = journal or SQLiteDecisionJournal()
        self.config = config or AdaptiveSystemConfig()
        self._pending: OrderedDict[str, tuple[StrategyForecast, ...]] = OrderedDict()
        self._lock = threading.RLock()

        strategy_ids = [self._strategy_id(strategy) for strategy in self.strategies]
        if len(strategy_ids) != len(set(strategy_ids)):
            raise ValueError("Ogni strategia deve avere uno strategy_id univoco.")

    @staticmethod
    def _strategy_id(strategy: AdaptiveStrategy) -> str:
        strategy_id = str(getattr(strategy, "strategy_id", "")).strip()
        if not strategy_id:
            raise ValueError("Ogni strategia deve dichiarare strategy_id.")
        return strategy_id

    @staticmethod
    def default_execution_estimate(
        snapshot: FeatureSnapshot,
    ) -> ExecutionEstimate:
        """Stima conservativa quando non è disponibile un modello microstrutturale."""

        spread_bps = (
            snapshot.spread_bps
            if snapshot.spread_bps is not None
            else 2.0 + 8.0 * (1.0 - snapshot.liquidity_score)
        )
        return ExecutionEstimate(
            spread_bps=spread_bps,
            fees_bps_per_side=0.50,
            slippage_bps_per_side=max(0.50, spread_bps * 0.15),
            market_impact_bps_per_side=max(
                0.25, (1.0 - snapshot.liquidity_score) * 2.0
            ),
            participation_rate=0.001,
            latency_ms=100.0,
        )

    def _remember_forecasts(
        self, cycle_id: str, forecasts: tuple[StrategyForecast, ...]
    ) -> None:
        with self._lock:
            self._pending[cycle_id] = forecasts
            self._pending.move_to_end(cycle_id)
            while len(self._pending) > self.config.maximum_pending_cycles:
                self._pending.popitem(last=False)

    @staticmethod
    def _cycle_status(result_meta, result_risk, result_execution) -> CycleStatus:
        if result_meta.is_no_trade:
            return CycleStatus.NO_TRADE
        if not result_risk.approved:
            return CycleStatus.RISK_REJECTED
        if not result_execution.approved:
            return CycleStatus.EXECUTION_REJECTED
        return CycleStatus.PAPER_APPROVED

    def analyze_snapshot(
        self,
        snapshot: FeatureSnapshot,
        portfolio: PortfolioState,
        *,
        execution_estimate: ExecutionEstimate | None = None,
        as_of: datetime | None = None,
    ) -> AdaptiveCycleResult:
        """Esegue un ciclo completo senza inviare ordini."""

        regime = self.regime_detector.detect(snapshot)
        forecasts: list[StrategyForecast] = []
        strategy_errors: dict[str, str] = {}

        for strategy in self.strategies:
            strategy_id = self._strategy_id(strategy)
            try:
                forecast = strategy.forecast(snapshot, regime)
                if forecast.strategy_id != strategy_id:
                    raise ValueError("strategy_id del forecast non coerente.")
                if forecast.ticker != snapshot.ticker:
                    raise ValueError("ticker del forecast non coerente.")
                forecasts.append(forecast)
            except Exception as exc:
                strategy_errors[strategy_id] = f"{type(exc).__name__}: {exc}"

        immutable_forecasts = tuple(forecasts)
        meta = self.meta_model.combine(
            immutable_forecasts,
            regime,
            self.performance_tracker.weights(),
            ticker=snapshot.ticker,
        )
        risk = self.risk_engine.evaluate(meta, snapshot, regime, portfolio)
        estimate = execution_estimate or self.default_execution_estimate(snapshot)
        execution = self.execution_gate.evaluate(
            meta,
            risk,
            snapshot,
            estimate,
            as_of=as_of or snapshot.timestamp,
        )
        status = self._cycle_status(meta, risk, execution)
        cycle_id = uuid.uuid4().hex
        result = AdaptiveCycleResult(
            cycle_id=cycle_id,
            snapshot=snapshot,
            regime=regime,
            forecasts=immutable_forecasts,
            meta_decision=meta,
            risk_decision=risk,
            execution_decision=execution,
            status=status,
            paper_only=True,
            strategy_errors=strategy_errors,
        )
        self.journal.record_cycle(result)
        self._remember_forecasts(cycle_id, immutable_forecasts)
        return result

    def analyze_market(
        self,
        data: pd.DataFrame,
        ticker: str,
        portfolio: PortfolioState,
        *,
        asset_class: str = "UNKNOWN",
        market_returns: pd.Series | None = None,
        news_sentiment: float = 0.0,
        news_relevance: float = 0.0,
        news_events: Sequence[ClassifiedNewsEvent] | None = None,
        timestamp: datetime | None = None,
        execution_estimate: ExecutionEstimate | None = None,
        as_of: datetime | None = None,
    ) -> AdaptiveCycleResult:
        snapshot = self.feature_engine.build(
            data,
            ticker,
            asset_class,
            market_returns=market_returns,
            news_sentiment=news_sentiment,
            news_relevance=news_relevance,
            timestamp=timestamp,
        )
        if news_events is not None:
            if news_sentiment != 0.0 or news_relevance != 0.0:
                raise ValueError(
                    "Usare news_events oppure i valori news manuali, non entrambi."
                )
            news = self.news_intelligence.assess(
                snapshot.ticker,
                news_events,
                as_of=snapshot.timestamp,
            )
            snapshot = replace(
                snapshot,
                news_sentiment=news.adjusted_sentiment,
                news_relevance=news.relevance,
                extras={
                    **snapshot.extras,
                    "news_raw_sentiment": news.raw_sentiment,
                    "news_contradiction": news.contradiction_score,
                    "news_event_risk": news.event_risk,
                    "news_event_count": float(news.event_count),
                },
            )
        return self.analyze_snapshot(
            snapshot,
            portfolio,
            execution_estimate=execution_estimate,
            as_of=as_of,
        )

    def analyze_universe(
        self,
        markets: Mapping[str, pd.DataFrame],
        portfolio: PortfolioState,
        *,
        asset_classes: Mapping[str, str] | None = None,
        execution_estimates: Mapping[str, ExecutionEstimate] | None = None,
        news_events: Mapping[str, Sequence[ClassifiedNewsEvent]] | None = None,
        timestamp: datetime | None = None,
    ) -> UniverseAnalysisResult:
        """Analizza un universo con isolamento degli errori per singolo asset."""

        asset_classes = asset_classes or {}
        execution_estimates = execution_estimates or {}
        news_events = news_events or {}
        cycles: dict[str, AdaptiveCycleResult] = {}
        errors: dict[str, str] = {}
        for raw_ticker, data in markets.items():
            ticker = str(raw_ticker).upper().strip()
            if not ticker:
                errors[str(raw_ticker)] = "Ticker vuoto."
                continue
            try:
                cycles[ticker] = self.analyze_market(
                    data,
                    ticker,
                    portfolio,
                    asset_class=asset_classes.get(ticker, "UNKNOWN"),
                    timestamp=timestamp,
                    execution_estimate=execution_estimates.get(ticker),
                    news_events=news_events.get(ticker),
                )
            except Exception as exc:
                errors[ticker] = f"{type(exc).__name__}: {exc}"
        allocation = self.portfolio_allocator.allocate(tuple(cycles.values()), portfolio)
        return UniverseAnalysisResult(
            cycles=cycles,
            errors=errors,
            allocation=allocation,
        )

    def record_outcome(
        self,
        cycle_id: str,
        *,
        realized_asset_return: float,
        transaction_cost_return: float = 0.0,
        slippage_return: float = 0.0,
        model_error: float | None = None,
        closed_at: datetime | None = None,
    ) -> Mapping[str, StrategyPerformanceState]:
        """Registra l'outcome e aggiorna pesi limitati; il codice non cambia."""

        realized = float(realized_asset_return)
        transaction_cost = float(transaction_cost_return)
        slippage = float(slippage_return)
        if not all(
            math.isfinite(value)
            for value in (realized, transaction_cost, slippage)
        ):
            raise ValueError("Outcome, costi e slippage devono essere finiti.")
        if transaction_cost < 0.0 or slippage < 0.0:
            raise ValueError("Costi e slippage non possono essere negativi.")
        if model_error is not None and not math.isfinite(float(model_error)):
            raise ValueError("model_error deve essere finito.")

        with self._lock:
            forecasts = self._pending.get(cycle_id)
            if forecasts is None:
                raise ValueError(f"cycle_id sconosciuto o già chiuso: {cycle_id}")

            self.journal.record_outcome(
                cycle_id,
                realized_asset_return=realized,
                transaction_cost_return=transaction_cost,
                slippage_return=slippage,
                model_error=model_error,
                closed_at=closed_at,
            )
            total_cost = transaction_cost + slippage
            states = {
                forecast.strategy_id: self.performance_tracker.update(
                    forecast,
                    realized,
                    total_cost,
                )
                for forecast in forecasts
            }
            del self._pending[cycle_id]
            return states

    @property
    def pending_cycle_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._pending)
