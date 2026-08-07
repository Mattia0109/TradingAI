from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adaptive.backtester import AdaptiveBacktestConfig, AdaptiveBacktester
from adaptive.run_adaptive_backtest import parse_arguments


def market(seed: int = 1, rows: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=rows, freq="B", tz="UTC")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, rows)))
    return pd.DataFrame({"close": close, "volume": 1_000_000.0}, index=index)


def market_with_date_column(seed: int = 1, rows: int = 150) -> pd.DataFrame:
    frame = market(seed, rows).reset_index(names="date")
    frame.index = pd.RangeIndex(len(frame))
    return frame


def test_backtest_is_paper_only_and_returns_aligned_outputs() -> None:
    result = AdaptiveBacktester(config=AdaptiveBacktestConfig(rebalance_every=5)).run({"SPY": market()})
    assert result.paper_only is True
    assert len(result.equity_curve) == len(result.daily_returns)
    assert result.weights.index.equals(result.daily_returns.index)
    assert result.decision_count > 0
    assert np.isfinite(result.total_return)
    assert 0.0 <= result.invested_fraction <= 1.0
    assert np.isfinite(result.equal_weight_total_return)


def test_future_price_change_cannot_change_past_results() -> None:
    original = market(rows=170)
    changed = original.copy()
    cutoff = 145
    changed.iloc[cutoff:, changed.columns.get_loc("close")] *= 4.0
    first = AdaptiveBacktester().run({"SPY": original})
    second = AdaptiveBacktester().run({"SPY": changed})
    comparison_date = original.index[cutoff - 1]
    pd.testing.assert_series_equal(first.equity_curve.loc[:comparison_date], second.equity_curve.loc[:comparison_date])
    pd.testing.assert_frame_equal(first.weights.loc[:comparison_date], second.weights.loc[:comparison_date])


def test_costs_equal_turnover_times_configured_rate() -> None:
    result = AdaptiveBacktester(config=AdaptiveBacktestConfig(cost_bps_per_turnover=12.0)).run({"SPY": market()})
    assert result.costs.sum() == pytest.approx(result.turnover.sum() * 12.0 / 10_000.0)


def test_rejects_insufficient_common_history() -> None:
    with pytest.raises(ValueError, match="Storico comune insufficiente"):
        AdaptiveBacktester().run({"SPY": market(rows=40)})


def test_backtest_cli_defaults_are_explicit_and_paper_oriented() -> None:
    args = parse_arguments(["--tickers", "SPY", "QQQ"])
    assert args.period == "5y"
    assert args.rebalance_every == 5
    assert args.cost_bps == 5.0


def test_drawdown_is_measured_from_initial_capital_too() -> None:
    falling = market(rows=100)
    falling["close"] = np.linspace(100.0, 50.0, len(falling))
    result = AdaptiveBacktester().run({"SPY": falling})
    assert 0.0 <= result.max_drawdown <= 1.0


def test_date_column_becomes_the_reported_calendar() -> None:
    result = AdaptiveBacktester().run({"SPY": market_with_date_column()})
    assert isinstance(result.daily_returns.index, pd.DatetimeIndex)
    assert result.daily_returns.index.tz is not None
    assert result.decision_count > 0
    assert result.analysis_error_count == 0
