from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from adaptive.long_burst import (
    LongBurstConfig,
    LongBurstSignalEngine,
)
from adaptive.long_burst_backtester import (
    LongBurstBacktestConfig,
    LongBurstBacktester,
)
from adaptive.models import Direction
from adaptive.run_long_burst_backtest import main, parse_arguments


def burst_market(seed: int = 7, rows: int = 360) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0002, 0.006, rows)
    for start in (120, 190, 260, 320):
        if start + 6 <= rows:
            returns[start : start + 6] += np.linspace(0.002, 0.009, 6)
    close = 100.0 * np.exp(np.cumsum(returns))
    open_price = np.r_[
        close[0],
        close[:-1] * (1.0 + rng.normal(0.0, 0.001, rows - 1)),
    ]
    high = np.maximum(open_price, close) * (
        1.0 + rng.uniform(0.001, 0.005, rows)
    )
    low = np.minimum(open_price, close) * (
        1.0 - rng.uniform(0.001, 0.005, rows)
    )
    volume = 1_000_000.0 * np.exp(rng.normal(0.0, 0.25, rows))
    return pd.DataFrame(
        {
            "date": pd.date_range(
                "2024-01-01",
                periods=rows,
                freq="h",
                tz="UTC",
            ),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def permissive_config(**changes) -> LongBurstConfig:
    values = {
        "minimum_raw_score": 0.05,
        "minimum_confidence": 0.01,
        "minimum_net_edge_bps": 0.0,
        "minimum_opportunity_score": 0.0,
        "minimum_lorentzian_score": 0.0,
        "minimum_component_support": 1,
        "round_trip_cost_bps": 0.0,
    }
    values.update(changes)
    return LongBurstConfig(**values)


def test_signal_engine_is_long_only_and_normalized() -> None:
    engine = LongBurstSignalEngine(permissive_config())

    result = engine.evaluate_history(burst_market(), "test")

    assert set(result["direction"].unique()) <= {"LONG", "FLAT"}
    assert "LONG" in set(result["direction"])
    for column in (
        "lorentzian_score",
        "squeeze_score",
        "cmf_score",
        "breakout_strength",
        "price_acceleration",
        "raw_score",
    ):
        assert result[column].between(-1.0, 1.0).all()
    for column in (
        "agreement",
        "correlation_penalty",
        "regime_quality",
        "data_quality",
        "confidence",
    ):
        assert result[column].between(0.0, 1.0).all()
    assert result["paper_only"].all()


def test_future_prices_cannot_change_past_signals() -> None:
    original = burst_market(rows=340)
    changed = original.copy()
    cutoff = 270
    changed.loc[cutoff:, ["open", "high", "low", "close"]] *= 3.0
    engine = LongBurstSignalEngine(permissive_config())

    first = engine.evaluate_history(original, "TEST")
    second = engine.evaluate_history(changed, "TEST")

    pd.testing.assert_frame_equal(first.iloc[:cutoff], second.iloc[:cutoff])


def test_latest_signal_adapts_to_existing_forecast_contract() -> None:
    engine = LongBurstSignalEngine(permissive_config())
    signal = engine.analyze(burst_market(), "spy")

    forecast = signal.to_forecast()

    assert signal.paper_only is True
    assert signal.direction in (Direction.LONG, Direction.FLAT)
    assert forecast.direction in (Direction.LONG, Direction.FLAT)
    assert forecast.strategy_id == "long_burst_momentum_v1"
    assert 1 <= forecast.horizon_bars <= 5


def test_data_validation_rejects_duplicates_and_incoherent_bars() -> None:
    engine = LongBurstSignalEngine()
    duplicate = burst_market(rows=140)
    duplicate.loc[1, "date"] = duplicate.loc[0, "date"]
    with pytest.raises(ValueError, match="duplicati"):
        engine.evaluate_history(duplicate, "TEST")

    invalid = burst_market(rows=140)
    invalid.loc[20, "high"] = invalid.loc[20, "low"] - 1.0
    with pytest.raises(ValueError, match="OHLC"):
        engine.evaluate_history(invalid, "TEST")


def test_configuration_enforces_short_horizon() -> None:
    with pytest.raises(ValueError, match="compreso tra 1 e 5"):
        LongBurstConfig(forecast_horizon_bars=6)
    with pytest.raises(ValueError, match="compreso tra 1 e 5"):
        LongBurstBacktestConfig(maximum_holding_bars=6)


def test_backtester_enters_next_bar_and_never_holds_over_limit() -> None:
    engine = LongBurstSignalEngine(permissive_config())
    result = LongBurstBacktester(
        engine,
        LongBurstBacktestConfig(
            maximum_holding_bars=3,
            round_trip_cost_bps=0.0,
        ),
    ).run({"AAA": burst_market(1), "BBB": burst_market(2)})

    assert result.paper_only is True
    assert result.trade_count > 0
    assert result.maximum_duration_bars <= 3
    assert (result.trades["entry_date"] > result.trades["signal_date"]).all()
    assert result.asset_exposure.ge(0.0).all().all()
    assert result.asset_exposure.le(1.0).all().all()
    assert result.gross_exposure.between(0.0, 1.0).all()
    assert result.average_gross_exposure <= 1.0


def test_backtest_prefix_is_unchanged_by_future_data() -> None:
    original = burst_market(rows=340)
    changed = original.copy()
    cutoff = 285
    changed.loc[cutoff:, ["open", "high", "low", "close"]] *= 2.5
    engine = LongBurstSignalEngine(permissive_config())
    backtester = LongBurstBacktester(
        engine,
        LongBurstBacktestConfig(round_trip_cost_bps=0.0),
    )

    first = backtester.run({"TEST": original})
    second = backtester.run({"TEST": changed})
    comparison_date = original.loc[cutoff - 1, "date"]

    pd.testing.assert_series_equal(
        first.daily_returns.loc[:comparison_date],
        second.daily_returns.loc[:comparison_date],
    )
    pd.testing.assert_series_equal(
        first.gross_exposure.loc[:comparison_date],
        second.gross_exposure.loc[:comparison_date],
    )


def test_transaction_costs_reduce_same_simulated_path() -> None:
    engine = LongBurstSignalEngine(permissive_config())
    no_cost = LongBurstBacktester(
        engine,
        LongBurstBacktestConfig(round_trip_cost_bps=0.0),
    ).run({"TEST": burst_market()})
    with_cost = LongBurstBacktester(
        engine,
        LongBurstBacktestConfig(round_trip_cost_bps=20.0),
    ).run({"TEST": burst_market()})

    assert with_cost.trade_count == no_cost.trade_count
    assert with_cost.total_return < no_cost.total_return
    assert with_cost.trades["net_return"].mean() < no_cost.trades["net_return"].mean()


def test_only_fully_matured_horizons_enter_diagnostics() -> None:
    horizon = 5
    engine = LongBurstSignalEngine(
        permissive_config(forecast_horizon_bars=horizon)
    )
    result = LongBurstBacktester(
        engine,
        LongBurstBacktestConfig(maximum_holding_bars=5),
    ).run({"TEST": burst_market()})

    assert not result.signal_observations.empty
    assert (
        result.signal_observations["outcome_date"]
        <= result.daily_returns.index[-1]
    ).all()
    assert (result.signal_observations["horizon_bars"] == horizon).all()


def test_cost_configuration_does_not_mutate_signal_engine() -> None:
    base = permissive_config()
    engine = LongBurstSignalEngine(base)
    LongBurstBacktester(
        engine,
        LongBurstBacktestConfig(round_trip_cost_bps=25.0),
    )

    assert engine.config == base
    assert replace(base, round_trip_cost_bps=25.0) != engine.config


def test_cli_defaults_describe_short_research_only_challenger() -> None:
    arguments = parse_arguments(["--tickers", "SPY", "QQQ"])

    assert arguments.period == "5y"
    assert arguments.interval == "1d"
    assert arguments.horizon_bars == 3
    assert arguments.max_holding_bars == 5
    assert arguments.cost_bps == 10.0
    assert arguments.skip_champion_comparison is False


def test_cli_runs_without_external_execution(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "adaptive.run_long_burst_backtest.load_markets",
        lambda pipeline, tickers, period, interval: ({"TEST": burst_market()}, {}),
    )

    exit_code = main(
        [
            "--tickers",
            "TEST",
            "--skip-champion-comparison",
            "--diagnostic-min-signals",
            "1",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "CHALLENGER RESEARCH-ONLY" in output
    assert "Segnali ammessi: LONG / NO_TRADE" in output
    assert "Leva: assente" in output
    assert "Broker: assente" in output
