from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adaptive.lorentzian_soft_surface import (
    DETAIL_COLUMNS,
    SUMMARY_COLUMNS,
    LorentzianSoftSurfaceConfig,
    analyze_lorentzian_soft_surface,
    summarize_soft_surface_across_assets,
    write_lorentzian_soft_surface,
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


def compact_config(**overrides) -> LorentzianSoftSurfaceConfig:
    values = {
        "history_limit": 240,
        "recency_half_lives": (80, 160),
        "context_penalties": (0.0, 0.5, 1.0),
        "neighbors": 4,
        "minimum_candidates": 8,
        "embargo_bars": 3,
        "sample_stride": 2,
        "query_stride": 5,
    }
    values.update(overrides)
    return LorentzianSoftSurfaceConfig(**values)


def test_config_rejects_bad_half_lives_and_penalties() -> None:
    with pytest.raises(ValueError, match="half_lives"):
        compact_config(recency_half_lives=(160, 80))
    with pytest.raises(ValueError, match="context_penalties"):
        compact_config(context_penalties=(-0.1, 0.5))


def test_surface_contains_reference_and_declared_grid() -> None:
    normalized, states = synthetic_inputs()
    report = analyze_lorentzian_soft_surface(
        "spy", normalized, states, compact_config()
    )

    assert not report.details.empty
    assert report.summary["profile"].nunique() == 7
    assert "REFERENCE_RAW" in set(report.summary["profile"])
    assert report.research_only


def test_reference_profile_is_identity() -> None:
    normalized, states = synthetic_inputs()
    report = analyze_lorentzian_soft_surface(
        "SPY", normalized, states, compact_config()
    )
    reference = report.details.loc[
        report.details["profile"].eq("REFERENCE_RAW")
        & report.details["neighbor_count"].eq(4)
    ]

    assert not reference.empty
    np.testing.assert_allclose(
        reference["raw_distance_median"],
        reference["adjusted_distance_median"],
    )
    np.testing.assert_allclose(reference["overlap_raw_reference"], 1.0)


def test_recency_penalty_reduces_aggregate_neighbor_age() -> None:
    normalized, states = synthetic_inputs()
    report = analyze_lorentzian_soft_surface(
        "SPY", normalized, states, compact_config()
    ).summary
    reference_age = report.loc[
        report["profile"].eq("REFERENCE_RAW"), "median_neighbor_age_bars"
    ].iloc[0]
    short_age = report.loc[
        report["profile"].eq("HL_80_CTX_0"), "median_neighbor_age_bars"
    ].iloc[0]

    assert short_age < reference_age


def test_context_penalty_increases_aggregate_context_agreement() -> None:
    normalized, states = synthetic_inputs()
    report = analyze_lorentzian_soft_surface(
        "SPY", normalized, states, compact_config()
    ).summary.set_index("profile")

    assert (
        report.loc["HL_160_CTX_1", "median_joint_context_match_fraction"]
        >= report.loc["HL_160_CTX_0", "median_joint_context_match_fraction"]
    )
    assert (
        report.loc["HL_160_CTX_1", "median_context_mismatches"]
        <= report.loc["HL_160_CTX_0", "median_context_mismatches"]
    )


def test_session_diversity_metrics_are_bounded() -> None:
    normalized, states = synthetic_inputs()
    report = analyze_lorentzian_soft_surface(
        "SPY", normalized, states, compact_config()
    ).details
    complete = report.loc[report["neighbor_count"].eq(4)]

    assert not complete.empty
    assert complete["neighbor_session_count"].between(1, 4).all()
    assert complete["neighbor_session_effective_count"].between(1, 4).all()
    assert complete["neighbor_max_session_share"].between(0.25, 1.0).all()


def test_future_mutation_cannot_change_past_surface() -> None:
    normalized, states = synthetic_inputs(650)
    config = compact_config()
    baseline = analyze_lorentzian_soft_surface(
        "SPY", normalized, states, config
    ).details
    cutoff = normalized.index[500]
    changed = normalized.copy()
    changed.loc[changed.index > cutoff, :] += np.array([50.0, -30.0, 80.0, -60.0])

    observed = analyze_lorentzian_soft_surface(
        "SPY", changed, states, config
    ).details

    pd.testing.assert_frame_equal(
        baseline.loc[baseline["timestamp"] <= cutoff].reset_index(drop=True),
        observed.loc[observed["timestamp"] <= cutoff].reset_index(drop=True),
    )


def test_embargo_and_history_limit_are_enforced() -> None:
    normalized, states = synthetic_inputs()
    config = compact_config()
    report = analyze_lorentzian_soft_surface(
        "SPY", normalized, states, config
    )
    complete = report.details.loc[report.details["neighbor_count"].eq(4)]

    assert not complete.empty
    assert (complete["neighbor_age_min_bars"] >= 4).all()
    assert (complete["neighbor_age_max_bars"] <= 240).all()


def test_cross_asset_summary_and_csv_schema(tmp_path) -> None:
    normalized, states = synthetic_inputs()
    first = analyze_lorentzian_soft_surface(
        "SPY", normalized, states, compact_config()
    ).summary
    second = first.copy()
    second["ticker"] = "QQQ"
    combined = pd.concat([first, second], ignore_index=True)

    cross = summarize_soft_surface_across_assets(combined)
    destination = write_lorentzian_soft_surface(
        combined, tmp_path / "soft_surface.csv"
    )

    assert (cross["assets"] == 2).all()
    assert tuple(pd.read_csv(destination).columns) == SUMMARY_COLUMNS


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
