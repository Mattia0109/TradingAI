from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd
import pytest

from adaptive.intraday_stability import (
    DistributionShift,
    IntradayFeatureStabilityAnalyzer,
    IntradayStabilityConfig,
    regular_session_frame,
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
    choppiness = 50.0 + 8.0 * np.sin(x / 17.0)
    cmf = 0.25 * np.sin(x / 11.0 + 0.8)
    return pd.DataFrame(
        {
            "squeeze_momentum": squeeze,
            "squeeze_momentum_change": np.r_[np.nan, np.diff(squeeze)],
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


def test_phase_summary_uses_open_middle_and_close_buckets() -> None:
    report = IntradayFeatureStabilityAnalyzer(compact_config()).analyze(
        "SPY",
        feature_values(15),
    )
    squeeze = report.phase_summary.loc[
        report.phase_summary["feature"] == "squeeze_momentum"
    ].set_index("phase")

    assert squeeze.loc["OPEN", "rows"] == 15 * 4
    assert squeeze.loc["MID_SESSION", "rows"] == 15 * 18
    assert squeeze.loc["CLOSE", "rows"] == 15 * 4


def test_redundancy_report_flags_near_duplicate_features() -> None:
    values = feature_values(15)
    values["cmf"] = values["squeeze_momentum"] * 0.5
    report = IntradayFeatureStabilityAnalyzer(compact_config()).analyze("SPY", values)
    row = report.redundancy.loc[
        (report.redundancy["left_feature"] == "squeeze_momentum")
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
        report.state_drift,
        report.phase_summary,
        report.redundancy,
    ):
        assert forbidden.isdisjoint(str(column).lower() for column in table.columns)
