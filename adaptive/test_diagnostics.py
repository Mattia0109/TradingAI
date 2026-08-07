import pandas as pd
import pytest

from adaptive.diagnostics import summarize_forecasts


def observation(expected, realized, direction="LONG", regime="TREND_UP"):
    return {
        "strategy_id": "model",
        "regime": regime,
        "direction": direction,
        "direction_sign": 1 if direction == "LONG" else -1,
        "expected_return": expected,
        "realized_return": realized,
        "confidence": 0.7,
        "horizon_bars": 5,
        "estimated_cost_bps": 10.0,
    }


def test_summarizes_directional_outcomes_net_of_costs() -> None:
    active, by_strategy, by_regime = summarize_forecasts(
        [observation(0.02, 0.03), observation(-0.01, -0.02, "SHORT")]
    )
    row = by_strategy.loc["model"]
    assert len(active) == 2
    assert row["signals"] == 2
    assert row["hit_rate"] == 1.0
    assert row["mean_signed_return"] == pytest.approx(0.025)
    assert row["mean_net_return"] == pytest.approx(0.024)
    assert ("model", "TREND_UP") in by_regime.index


def test_excludes_flat_and_unmatured_forecasts() -> None:
    flat = observation(0.0, 0.01, "FLAT")
    pending = observation(0.02, None)
    active, by_strategy, _ = summarize_forecasts([flat, pending])
    assert active.empty
    assert by_strategy.empty


def test_correlation_is_zero_for_constant_forecasts() -> None:
    _, summary, _ = summarize_forecasts(
        [observation(0.01, 0.02), observation(0.01, -0.01)]
    )
    assert summary.loc["model", "forecast_correlation"] == 0.0
