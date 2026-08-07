from datetime import datetime, timezone

import pytest

from adaptive.learning import (
    CausalRegimePerformanceTracker,
    RegimeLearningConfig,
)
from adaptive.models import Direction, MarketRegime, StrategyForecast


def forecast() -> StrategyForecast:
    return StrategyForecast(
        strategy_id="trend",
        ticker="SPY",
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        direction=Direction.LONG,
        confidence=0.8,
        expected_return=0.02,
        expected_volatility=0.15,
        horizon_bars=5,
        regime_fit=0.9,
    )


def fast_config() -> RegimeLearningConfig:
    return RegimeLearningConfig(
        skill_alpha=1.0,
        return_scale=0.01,
        minimum_observations=2,
        maximum_weight_step=1.0,
        quarantine_threshold=-0.10,
        release_threshold=0.05,
    )


def test_quarantine_is_causal_regime_specific_and_reversible() -> None:
    tracker = CausalRegimePerformanceTracker(fast_config())
    item = forecast()
    tracker.update(item, -0.03, regime=MarketRegime.TREND_DOWN)
    quarantined = tracker.update(
        item, -0.03, regime=MarketRegime.TREND_DOWN
    )
    assert quarantined.quarantined is True
    assert quarantined.weight == pytest.approx(0.25)
    assert tracker.weights(MarketRegime.TREND_UP) == {}

    released = tracker.update(
        item, 0.05, regime=MarketRegime.TREND_DOWN
    )
    assert released.quarantined is False
    assert released.weight > quarantined.weight


def test_tracker_stays_neutral_until_minimum_sample() -> None:
    tracker = CausalRegimePerformanceTracker(fast_config())
    state = tracker.update(
        forecast(), -0.50, regime=MarketRegime.TREND_DOWN
    )
    assert state.observations == 1
    assert state.weight == 1.0
    assert state.quarantined is False
