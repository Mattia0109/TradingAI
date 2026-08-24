from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adaptive.lorentzian_neighborhood_stability import (
    DETAIL_COLUMNS,
    SUMMARY_COLUMNS,
    LorentzianNeighborhoodStabilityConfig,
    analyze_lorentzian_neighborhood_stability,
    write_lorentzian_neighborhood_stability,
)


def synthetic_inputs(rows: int = 520) -> tuple[pd.DataFrame, pd.DataFrame]:
    sessions = (rows + 25) // 26
    timestamps: list[pd.Timestamp] = []
    for day in pd.bdate_range("2025-01-06", periods=sessions):
        start = pd.Timestamp(day.date(), tz="America/New_York") + pd.Timedelta(
            hours=9, minutes=30
        )
        timestamps.extend(pd.date_range(start, periods=26, freq="15min"))
    index = pd.DatetimeIndex(timestamps[:rows], name="timestamp")
    x = np.arange(rows, dtype=float)
    normalized = pd.DataFrame(
        {
            "sqz": np.sin(x / 11.0) + 0.0007 * x,
            "sqz_change": np.cos(x / 7.0) - 0.0003 * x,
            "chop": np.sin(x / 19.0) * 0.8,
            "cmf": np.cos(x / 23.0) * 0.6,
        },
        index=index,
    )
    states = pd.DataFrame(
        {
            "chop_segment": np.where(
                (x.astype(int) // 9) % 3 == 0,
                "TRENDING",
                np.where((x.astype(int) // 9) % 3 == 1, "NEUTRAL", "CHOPPY"),
            ),
            "squeeze_state": np.where(
                (x.astype(int) // 13) % 2 == 0,
                "SQUEEZE_ON",
                "SQUEEZE_OFF",
            ),
        },
        index=index,
    )
    return normalized, states


def compact_config(**overrides) -> LorentzianNeighborhoodStabilityConfig:
    values = {
        "history_limits": (80, 160, 240),
        "neighbors": 4,
        "minimum_candidates": 8,
        "embargo_bars": 3,
        "sample_stride": 2,
        "query_stride": 5,
        "minimum_overlap_queries": 5,
    }
    values.update(overrides)
    return LorentzianNeighborhoodStabilityConfig(**values)


def test_config_rejects_unsorted_or_impossible_windows() -> None:
    with pytest.raises(ValueError, match="crescente"):
        compact_config(history_limits=(160, 80, 240))
    with pytest.raises(ValueError, match="troppo corta"):
        compact_config(history_limits=(4, 80), minimum_candidates=8)


def test_report_contains_both_filter_modes_and_all_windows() -> None:
    normalized, states = synthetic_inputs()
    report = analyze_lorentzian_neighborhood_stability(
        "spy", normalized, states, compact_config()
    )

    assert not report.details.empty
    assert set(report.details["filter_mode"]) == {"PHASE_ONLY", "JOINT_CONTEXT"}
    assert set(report.details["history_limit"]) == {80, 160, 240}
    assert set(report.summary["history_limit"]) == {80, 160, 240}
    assert report.research_only


def test_every_selected_neighbor_respects_embargo_and_window() -> None:
    normalized, states = synthetic_inputs()
    config = compact_config()
    report = analyze_lorentzian_neighborhood_stability(
        "SPY", normalized, states, config
    )
    complete = report.details.loc[
        report.details["neighbor_count"].eq(config.neighbors)
    ]

    assert not complete.empty
    assert (
        complete["neighbor_age_min_bars"] >= config.embargo_bars + 1
    ).all()
    assert (
        complete["neighbor_age_max_bars"] <= complete["history_limit"]
    ).all()


def test_reference_window_has_unit_overlap_and_distance_ratio() -> None:
    normalized, states = synthetic_inputs()
    config = compact_config()
    report = analyze_lorentzian_neighborhood_stability(
        "SPY", normalized, states, config
    )
    reference = report.details.loc[
        report.details["history_limit"].eq(max(config.history_limits))
        & report.details["neighbor_count"].eq(config.neighbors)
    ]

    assert not reference.empty
    np.testing.assert_allclose(reference["overlap_reference"], 1.0)
    np.testing.assert_allclose(reference["distance_ratio_reference"], 1.0)


def test_future_mutation_cannot_change_past_queries() -> None:
    normalized, states = synthetic_inputs(650)
    config = compact_config()
    baseline = analyze_lorentzian_neighborhood_stability(
        "SPY", normalized, states, config
    ).details
    cutoff = normalized.index[500]
    changed = normalized.copy()
    changed.loc[changed.index > cutoff, :] += np.array([50.0, -30.0, 80.0, -60.0])

    observed = analyze_lorentzian_neighborhood_stability(
        "SPY", changed, states, config
    ).details

    pd.testing.assert_frame_equal(
        baseline.loc[baseline["timestamp"] <= cutoff].reset_index(drop=True),
        observed.loc[observed["timestamp"] <= cutoff].reset_index(drop=True),
    )


def test_sparse_joint_context_is_explicit_not_filled() -> None:
    normalized, states = synthetic_inputs()
    states.loc[:, "chop_segment"] = [f"STATE_{i % 60}" for i in range(len(states))]
    config = compact_config(minimum_candidates=10)
    report = analyze_lorentzian_neighborhood_stability(
        "SPY", normalized, states, config
    )
    phase = report.summary.loc[
        report.summary["filter_mode"].eq("PHASE_ONLY")
        & report.summary["history_limit"].eq(80)
    ].iloc[0]
    joint = report.summary.loc[
        report.summary["filter_mode"].eq("JOINT_CONTEXT")
        & report.summary["history_limit"].eq(80)
    ].iloc[0]

    assert joint["coverage"] < phase["coverage"]
    assert joint["overlap_state"] == "INSUFFICIENT_COMPARISON"


def test_output_contract_contains_no_operational_fields() -> None:
    forbidden = {
        "signal",
        "direction",
        "prediction",
        "order",
        "position",
        "entry",
        "exit",
        "pnl",
        "outcome",
        "leverage",
    }

    assert all(
        not any(token in column.lower() for token in forbidden)
        for column in (*DETAIL_COLUMNS, *SUMMARY_COLUMNS)
    )


def test_csv_writer_preserves_declared_schema(tmp_path) -> None:
    normalized, states = synthetic_inputs()
    report = analyze_lorentzian_neighborhood_stability(
        "SPY", normalized, states, compact_config()
    )
    path = tmp_path / "lorentzian_stability.csv"

    destination = write_lorentzian_neighborhood_stability(report.summary, path)
    observed = pd.read_csv(destination)

    assert tuple(observed.columns) == SUMMARY_COLUMNS
    assert len(observed) == len(report.summary)
    with pytest.raises(ValueError, match="csv"):
        write_lorentzian_neighborhood_stability(report.summary, tmp_path / "x.txt")
