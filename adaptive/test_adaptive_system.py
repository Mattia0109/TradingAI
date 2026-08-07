from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from adaptive.execution_gate import EconomicExecutionGate
from adaptive.feature_engine import AdaptiveFeatureEngine
from adaptive.journal import SQLiteDecisionJournal
from adaptive.learning import ControlledPerformanceTracker
from adaptive.meta_model import AdaptiveMetaModel
from adaptive.models import (
    CycleStatus,
    Direction,
    ExecutionEstimate,
    FeatureSnapshot,
    PortfolioState,
    StrategyForecast,
)
from adaptive.news_intelligence import (
    ClassifiedNewsEvent,
    StructuredNewsIntelligence,
)
from adaptive.orchestrator import AdaptiveTradingSystem
from adaptive.regime_detector import MarketRegimeDetector


NOW = datetime(2026, 1, 15, 16, 0, 0)


def make_snapshot(**changes) -> FeatureSnapshot:
    values = {
        "ticker": "SPY",
        "asset_class": "ETF",
        "timestamp": NOW,
        "price": 500.0,
        "return_1": 0.005,
        "return_5": 0.02,
        "return_21": 0.08,
        "realized_volatility_short": 0.16,
        "realized_volatility_long": 0.15,
        "trend_score": 0.90,
        "return_zscore": 0.50,
        "shock_score": 0.50,
        "volume_zscore": 1.0,
        "spread_bps": 2.0,
        "liquidity_score": 0.95,
        "market_correlation": 0.30,
        "data_quality": 1.0,
    }
    values.update(changes)
    return FeatureSnapshot(**values)


def cheap_execution() -> ExecutionEstimate:
    return ExecutionEstimate(
        spread_bps=2.0,
        fees_bps_per_side=0.10,
        slippage_bps_per_side=0.10,
        market_impact_bps_per_side=0.10,
        participation_rate=0.001,
        latency_ms=25.0,
    )


class LongStrategy:
    strategy_id = "test_long"

    def forecast(self, snapshot, regime):
        return StrategyForecast(
            strategy_id=self.strategy_id,
            ticker=snapshot.ticker,
            timestamp=snapshot.timestamp,
            direction=Direction.LONG,
            confidence=0.90,
            expected_return=0.02,
            expected_volatility=0.12,
            horizon_bars=10,
            regime_fit=1.0,
            reasons=("Forecast deterministico per il test.",),
        )


class FlatStrategy:
    strategy_id = "test_flat"

    def forecast(self, snapshot, regime):
        return StrategyForecast(
            strategy_id=self.strategy_id,
            ticker=snapshot.ticker,
            timestamp=snapshot.timestamp,
            direction=Direction.FLAT,
            confidence=0.0,
            expected_return=0.0,
            expected_volatility=snapshot.realized_volatility_short,
            horizon_bars=5,
            regime_fit=0.0,
        )


class BrokenStrategy:
    strategy_id = "test_broken"

    def forecast(self, snapshot, regime):
        raise RuntimeError("modello temporaneamente indisponibile")


def empty_portfolio(**changes) -> PortfolioState:
    values = {"equity": 100_000.0, "peak_equity": 100_000.0}
    values.update(changes)
    return PortfolioState(**values)


def trending_market(rows: int = 100) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    close = 100.0 * np.exp(0.001 * index + 0.002 * np.sin(index / 5.0))
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=rows, freq="B"),
            "close": close,
            "volume": 1_000_000.0 + index * 1_000.0,
            "spread_bps": 2.0,
        }
    )


def test_feature_engine_builds_causal_snapshot() -> None:
    engine = AdaptiveFeatureEngine()
    data = trending_market()

    first = engine.build(data.iloc[:90], "spy", "etf")
    repeated = engine.build(data.iloc[:90], "spy", "etf")

    assert first == repeated
    assert first.ticker == "SPY"
    assert first.asset_class == "ETF"
    assert first.trend_score > 0.50
    assert first.return_21 > 0.0
    assert 0.0 <= first.data_quality <= 1.0


def test_feature_engine_rejects_insufficient_history() -> None:
    with pytest.raises(ValueError, match="Storico insufficiente"):
        AdaptiveFeatureEngine().build(trending_market(20), "SPY")


def test_regime_detector_stops_liquidity_crisis() -> None:
    snapshot = make_snapshot(liquidity_score=0.10, spread_bps=150.0)

    regime = MarketRegimeDetector().detect(snapshot)

    assert regime.risk_multiplier == 0.0
    assert any("liquidità" in reason.lower() for reason in regime.reasons)
    assert sum(regime.probabilities.values()) == pytest.approx(1.0)


def test_full_cycle_can_approve_only_a_paper_candidate() -> None:
    system = AdaptiveTradingSystem(strategies=(LongStrategy(),))

    result = system.analyze_snapshot(
        make_snapshot(),
        empty_portfolio(),
        execution_estimate=cheap_execution(),
    )

    assert result.status is CycleStatus.PAPER_APPROVED
    assert result.paper_only is True
    assert result.risk_decision.approved is True
    assert result.execution_decision.approved is True
    assert result.execution_decision.order_style != "NO_ORDER"


def test_no_trade_is_not_coerced_into_a_position() -> None:
    system = AdaptiveTradingSystem(strategies=(FlatStrategy(),))

    result = system.analyze_snapshot(make_snapshot(), empty_portfolio())

    assert result.status is CycleStatus.NO_TRADE
    assert result.meta_decision.direction is Direction.FLAT
    assert result.risk_decision.target_weight == 0.0
    assert result.execution_decision.order_style == "NO_ORDER"


def test_risk_engine_has_authority_over_a_positive_forecast() -> None:
    system = AdaptiveTradingSystem(strategies=(LongStrategy(),))
    portfolio = empty_portfolio(equity=75_000.0, peak_equity=100_000.0)

    result = system.analyze_snapshot(
        make_snapshot(),
        portfolio,
        execution_estimate=cheap_execution(),
    )

    assert result.status is CycleStatus.RISK_REJECTED
    assert result.risk_decision.approved is False
    assert result.risk_decision.target_weight == 0.0
    assert any("kill switch" in reason.lower() for reason in result.risk_decision.reasons)


def test_execution_gate_rejects_an_uneconomic_order() -> None:
    system = AdaptiveTradingSystem(strategies=(LongStrategy(),))
    expensive = replace(
        cheap_execution(),
        spread_bps=250.0,
        slippage_bps_per_side=20.0,
        market_impact_bps_per_side=20.0,
    )

    result = system.analyze_snapshot(
        make_snapshot(),
        empty_portfolio(),
        execution_estimate=expensive,
    )

    assert result.status is CycleStatus.EXECUTION_REJECTED
    assert result.execution_decision.approved is False
    assert result.execution_decision.order_style == "NO_ORDER"


def test_strategy_failure_is_isolated_from_the_other_specialists() -> None:
    system = AdaptiveTradingSystem(strategies=(BrokenStrategy(), LongStrategy()))

    result = system.analyze_snapshot(
        make_snapshot(),
        empty_portfolio(),
        execution_estimate=cheap_execution(),
    )

    assert result.status is CycleStatus.PAPER_APPROVED
    assert "test_broken" in result.strategy_errors
    assert tuple(item.strategy_id for item in result.forecasts) == ("test_long",)


def test_learning_uses_only_recorded_outcomes_and_is_bounded() -> None:
    journal = SQLiteDecisionJournal()
    tracker = ControlledPerformanceTracker()
    system = AdaptiveTradingSystem(
        strategies=(LongStrategy(),),
        journal=journal,
        performance_tracker=tracker,
    )
    result = system.analyze_snapshot(
        make_snapshot(),
        empty_portfolio(),
        execution_estimate=cheap_execution(),
    )

    assert journal.fetch_training_rows() == []
    assert tracker.weights() == {}

    states = system.record_outcome(
        result.cycle_id,
        realized_asset_return=0.03,
        transaction_cost_return=0.001,
        slippage_return=0.0005,
    )

    assert len(journal.fetch_training_rows()) == 1
    assert 1.0 < states["test_long"].weight <= 1.05
    assert result.cycle_id not in system.pending_cycle_ids
    with pytest.raises(ValueError, match="già chiuso"):
        system.record_outcome(result.cycle_id, realized_asset_return=0.01)


def test_universe_analysis_isolates_bad_assets() -> None:
    system = AdaptiveTradingSystem(strategies=(LongStrategy(),))

    result = system.analyze_universe(
        {"SPY": trending_market(), "BAD": trending_market(10)},
        empty_portfolio(),
        asset_classes={"SPY": "ETF", "BAD": "ETF"},
        timestamp=NOW,
    )

    assert "SPY" in result.cycles
    assert "BAD" in result.errors
    assert result.cycles["SPY"].paper_only is True
    assert result.allocation.projected_gross_exposure <= 1.0


def test_universe_allocator_enforces_cumulative_asset_class_limit() -> None:
    system = AdaptiveTradingSystem(strategies=(LongStrategy(),))
    tickers = [f"EQ{index}" for index in range(20)]

    result = system.analyze_universe(
        {ticker: trending_market() for ticker in tickers},
        empty_portfolio(),
        asset_classes={ticker: "EQUITY" for ticker in tickers},
        timestamp=NOW,
    )

    allocated_gross = sum(
        abs(weight) for weight in result.allocation.target_weights.values()
    )
    assert allocated_gross <= 0.30 + 1e-12
    assert result.allocation.projected_gross_exposure == pytest.approx(
        allocated_gross
    )
    assert len(result.allocation.target_weights) < len(tickers)
    assert result.allocation.rejected


def test_execution_age_accepts_mixed_timezone_awareness() -> None:
    system = AdaptiveTradingSystem(strategies=(LongStrategy(),))
    aware_as_of = NOW.replace(tzinfo=timezone.utc) + timedelta(seconds=30)

    result = system.analyze_snapshot(
        make_snapshot(),
        empty_portfolio(),
        execution_estimate=cheap_execution(),
        as_of=aware_as_of,
    )

    assert result.status is CycleStatus.PAPER_APPROVED


def test_meta_model_keeps_no_trade_when_specialists_conflict() -> None:
    snapshot = make_snapshot()
    regime = MarketRegimeDetector().detect(snapshot)
    long_forecast = LongStrategy().forecast(snapshot, regime)
    short_forecast = replace(
        long_forecast,
        strategy_id="test_short",
        direction=Direction.SHORT,
        expected_return=-long_forecast.expected_return,
    )

    decision = AdaptiveMetaModel().combine((long_forecast, short_forecast), regime)

    assert decision.is_no_trade
    assert any("conflitto" in reason.lower() for reason in decision.reasons)


def test_execution_gate_rejects_stale_snapshot() -> None:
    snapshot = make_snapshot()
    regime = MarketRegimeDetector().detect(snapshot)
    forecast = LongStrategy().forecast(snapshot, regime)
    meta = AdaptiveMetaModel().combine((forecast,), regime)
    system = AdaptiveTradingSystem(strategies=(LongStrategy(),))
    risk = system.risk_engine.evaluate(meta, snapshot, regime, empty_portfolio())

    decision = EconomicExecutionGate().evaluate(
        meta,
        risk,
        snapshot,
        cheap_execution(),
        as_of=NOW + timedelta(minutes=10),
    )

    assert decision.approved is False
    assert any("scaduto" in reason.lower() for reason in decision.reasons)


def test_news_intelligence_uses_price_reaction_and_ignores_future_events() -> None:
    contradictory = ClassifiedNewsEvent(
        event_id="news-1",
        published_at=NOW - timedelta(minutes=1),
        tickers=("SPY",),
        sentiment=0.80,
        importance=1.0,
        novelty=1.0,
        source_credibility=1.0,
        price_reaction=-0.05,
    )
    future = replace(
        contradictory,
        event_id="news-future",
        published_at=NOW + timedelta(minutes=1),
        sentiment=-1.0,
    )

    assessment = StructuredNewsIntelligence().assess(
        "SPY",
        (contradictory, future),
        as_of=NOW,
    )

    assert assessment.event_count == 1
    assert assessment.raw_sentiment > 0.0
    assert assessment.adjusted_sentiment < 0.0
    assert assessment.contradiction_score > 0.50
    assert assessment.event_ids == ("news-1",)


def test_adverse_news_reaction_can_trigger_the_risk_veto() -> None:
    event = ClassifiedNewsEvent(
        event_id="news-risk-veto",
        published_at=NOW - timedelta(minutes=1),
        tickers=("SPY",),
        sentiment=0.80,
        importance=1.0,
        novelty=1.0,
        source_credibility=1.0,
        price_reaction=-0.05,
    )
    system = AdaptiveTradingSystem(strategies=(LongStrategy(),))

    result = system.analyze_market(
        trending_market(),
        "SPY",
        empty_portfolio(),
        asset_class="ETF",
        news_events=(event,),
        timestamp=NOW,
        execution_estimate=cheap_execution(),
    )

    assert result.snapshot.news_sentiment < -0.40
    assert result.snapshot.news_relevance >= 0.85
    assert result.status is CycleStatus.RISK_REJECTED
    assert any("evento" in reason.lower() for reason in result.risk_decision.reasons)
