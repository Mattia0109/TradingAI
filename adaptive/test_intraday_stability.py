from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd
import pytest

from adaptive.intraday_stability import (
    DistributionShift,
    IntradayFeatureStabilityAnalyzer,
    IntradayStabilityConfig,
    add_dimensionless_squeeze_features,
    regular_session_frame,
    summarize_cross_asset_phase_consensus,
)


def feature_values(sessions: int = 20) -> pd.DataFrame:
    session_days = pd.bdate_range("2025-01-06", periods=sessions)
    timestamps = pd.DatetimeIndex(
        [
            pd.Timestamp(day.date(), tz="America/New_York")
            + pd.Timedelta(hours=9, minutes=30)
            + pd.Timedelta(minutes=15 * bar)
            for day in session_days
            for bar in range(26)
        ]
    )
    x = np.arange(len(timestamps), dtype=float)
    squeeze = np.sin(x / 13.0) + 0.1 * np.cos(x / 5.0)
    squeeze_change = np.r_[np.nan, np.diff(squeeze)]
    choppiness = 50.0 + 8.0 * np.sin(x / 17.0)
    cmf = 0.25 * np.sin(x / 11.0 + 0.8)
    return pd.DataFrame(
        {
            "squeeze_momentum": squeeze,
            "squeeze_momentum_change": squeeze_change,
            "squeeze_momentum_pct_close": squeeze / 100.0,
            "squeeze_momentum_change_pct_close": squeeze_change / 100.0,
            "choppiness": choppiness,
            "cmf": cmf,
            "squeeze_state": np.where(
                np.sin(x / 19.0) > 0.0,
                "SQUEEZE_ON",
                "SQUEEZE_OFF",
            ),
            "chop_segment": np.where(
                choppiness > 60.0,
                "CHOPPY",
                np.where(choppiness < 40.0, "TRENDING", "NEUTRAL"),
            ),
        },
        index=timestamps,
    )


def compact_config(**overrides) -> IntradayStabilityConfig:
    values = {
        "sessions_per_block": 5,
        "minimum_complete_blocks": 3,
        "minimum_valid_rows_per_block": 20,
    }
    values.update(overrides)
    return IntradayStabilityConfig(**values)


def test_configuration_rejects_unordered_session_phases() -> None:
    with pytest.raises(ValueError, match="fasi della sessione"):
        IntradayStabilityConfig(
            open_phase_end=time(15, 30),
            close_phase_start=time(10, 30),
        )


def test_regular_session_frame_excludes_extended_hours_and_weekends() -> None:
    dates = pd.DatetimeIndex(
        [
            "2025-01-06 08:00",
            "2025-01-06 09:30",
            "2025-01-06 15:45",
            "2025-01-06 16:00",
            "2025-01-11 10:00",
        ],
        tz="America/New_York",
    )
    market = pd.DataFrame(
        {
            "date": dates,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000.0,
        }
    )

    regular, excluded = regular_session_frame(market)

    assert len(regular) == 2
    assert excluded == 3
    assert list(regular["date"].dt.strftime("%H:%M")) == ["09:30", "15:45"]


def test_complete_block_assignments_do_not_change_when_future_is_appended() -> None:
    values = feature_values(20)
    sessions = pd.Series(values.index.date, index=values.index)
    first_fifteen = sorted(sessions.unique())[:15]
    prefix = values.loc[sessions.isin(first_fifteen)]
    analyzer = IntradayFeatureStabilityAnalyzer(compact_config())

    before = analyzer.analyze("spy", prefix)
    after = analyzer.analyze("SPY", values)

    pd.testing.assert_frame_equal(
        before.block_summary.reset_index(drop=True),
        after.block_summary.loc[
            after.block_summary["block"] <= 3
        ].reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        before.numeric_drift.reset_index(drop=True),
        after.numeric_drift.loc[
            after.numeric_drift["current_block"] <= 3
        ].reset_index(drop=True),
    )


def test_dimensionless_squeeze_is_invariant_to_price_unit_scaling() -> None:
    values = feature_values(5)
    market = pd.DataFrame(
        {
            "date": values.index,
            "close": np.linspace(100.0, 105.0, len(values)),
        }
    )
    scaled_values = values.copy()
    scaled_values["squeeze_momentum"] *= 10.0
    scaled_values["squeeze_momentum_change"] *= 10.0
    scaled_market = market.copy()
    scaled_market["close"] *= 10.0

    baseline = add_dimensionless_squeeze_features(values, market)
    scaled = add_dimensionless_squeeze_features(scaled_values, scaled_market)

    pd.testing.assert_series_equal(
        baseline["squeeze_momentum_pct_close"],
        scaled["squeeze_momentum_pct_close"],
    )
    pd.testing.assert_series_equal(
        baseline["squeeze_momentum_change_pct_close"],
        scaled["squeeze_momentum_change_pct_close"],
    )


def test_large_last_block_change_is_reported_as_high_shift() -> None:
    values = feature_values(20)
    sessions = pd.Series(values.index.date, index=values.index)
    last_five = sorted(sessions.unique())[-5:]
    changed = values.copy()
    changed.loc[sessions.isin(last_five), "choppiness"] += 40.0
    report = IntradayFeatureStabilityAnalyzer(compact_config()).analyze(
        "SPY",
        changed,
    )
    row = report.numeric_drift.loc[
        (report.numeric_drift["feature"] == "choppiness")
        & (report.numeric_drift["current_block"] == 4)
    ].iloc[0]

    assert row["shift"] == DistributionShift.HIGH_SHIFT.value
    assert row["median_shift_iqr"] > 1.5


def test_consecutive_elevated_transitions_are_marked_persistent() -> None:
    values = feature_values(20)
    sessions = pd.Series(values.index.date, index=values.index)
    ordered_sessions = sorted(sessions.unique())
    changed = values.copy()
    changed.loc[sessions.isin(ordered_sessions[10:15]), "choppiness"] += 20.0
    changed.loc[sessions.isin(ordered_sessions[15:20]), "choppiness"] += 40.0

    report = IntradayFeatureStabilityAnalyzer(compact_config()).analyze(
        "SPY",
        changed,
    )
    row = report.numeric_persistence.loc[
        report.numeric_persistence["feature"] == "choppiness"
    ].iloc[0]

    assert row["high_transitions"] >= 2
    assert row["longest_elevated_run"] >= 2
    assert row["pattern"] == "PERSISTENT_HIGH"


def test_phase_summary_uses_open_middle_and_close_buckets() -> None:
    report = IntradayFeatureStabilityAnalyzer(compact_config()).analyze(
        "SPY",
        feature_values(15),
    )
    squeeze = report.phase_summary.loc[
        report.phase_summary["feature"] == "squeeze_momentum_pct_close"
    ].set_index("phase")

    assert squeeze.loc["OPEN", "rows"] == 15 * 4
    assert squeeze.loc["MID_SESSION", "rows"] == 15 * 18
    assert squeeze.loc["CLOSE", "rows"] == 15 * 4


def test_phase_drift_keeps_separate_session_buckets() -> None:
    report = IntradayFeatureStabilityAnalyzer(compact_config()).analyze(
        "SPY",
        feature_values(15),
    )

    assert len(report.phase_drift) == 2 * 3 * 4
    assert len(report.phase_persistence) == 3 * 4
    assert set(report.phase_persistence["phase"]) == {
        "OPEN",
        "MID_SESSION",
        "CLOSE",
    }


def test_cross_asset_consensus_counts_phase_persistence_without_ranking() -> None:
    values = feature_values(20)
    bar = np.tile(np.arange(26), 20)
    values["cmf"] = 0.10 * np.sin(bar / 4.0)
    sessions = pd.Series(values.index.date, index=values.index)
    ordered_sessions = sorted(sessions.unique())
    close_phase = values.index.hour >= 15

    shifted = values.copy()
    block_three = sessions.isin(ordered_sessions[10:15]) & close_phase
    block_four = sessions.isin(ordered_sessions[15:20]) & close_phase
    shifted.loc[block_three, "cmf"] += 1.0
    shifted.loc[block_four, "cmf"] += 2.0
    analyzer = IntradayFeatureStabilityAnalyzer(compact_config())
    reports = [
        analyzer.analyze("AAA", shifted),
        analyzer.analyze("BBB", shifted),
        analyzer.analyze("CCC", values),
    ]

    consensus = summarize_cross_asset_phase_consensus(reports)
    row = consensus.loc[
        (consensus["feature"] == "cmf")
        & (consensus["phase"] == "CLOSE")
    ].iloc[0]

    assert row["assets"] == 3
    assert row["sufficient_assets"] == 3
    assert row["persistent_assets"] == 2
    assert row["persistent_high_assets"] == 2


def test_redundancy_report_flags_near_duplicate_features() -> None:
    values = feature_values(15)
    values["cmf"] = values["squeeze_momentum_pct_close"] * 0.5
    report = IntradayFeatureStabilityAnalyzer(compact_config()).analyze("SPY", values)
    row = report.redundancy.loc[
        (report.redundancy["left_feature"] == "squeeze_momentum_pct_close")
        & (report.redundancy["right_feature"] == "cmf")
    ].iloc[0]

    assert row["absolute_correlation"] == pytest.approx(1.0)
    assert row["relation"] == "HIGH_REDUNDANCY"


def test_report_contains_no_operational_or_future_outcome_fields() -> None:
    report = IntradayFeatureStabilityAnalyzer(compact_config()).analyze(
        "SPY",
        feature_values(15),
    )
    assert report.research_only is True
    forbidden = {
        "signal",
        "direction",
        "order",
        "position",
        "entry",
        "exit",
        "pnl",
        "return",
        "outcome",
        "leverage",
    }
    for table in (
        report.block_summary,
        report.numeric_drift,
        report.numeric_persistence,
        report.state_drift,
        report.state_persistence,
        report.phase_summary,
        report.phase_drift,
        report.phase_persistence,
        report.redundancy,
    ):
        assert forbidden.isdisjoint(str(column).lower() for column in table.columns)
