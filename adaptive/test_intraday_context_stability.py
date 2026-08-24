from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from adaptive.intraday_context_stability import (
    DRIFT_COLUMNS,
    PERSISTENCE_COLUMNS,
    ContextDistributionShift,
    IntradayContextStabilityConfig,
    analyze_intraday_context_stability,
    summarize_context_stability,
    write_context_stability,
)
from adaptive.intraday_regime_atlas import (
    IntradayRegimeAtlasConfig,
    build_intraday_regime_atlas,
)


def audit(status: str = "READY_DESCRIPTIVE") -> SimpleNamespace:
    return SimpleNamespace(
        status=SimpleNamespace(value=status),
        has_volume=True,
    )


def feature_frame(block_states: tuple[str, ...]) -> pd.DataFrame:
    """Crea due sedute da quattro barre per ogni blocco dichiarato."""

    indexes: list[pd.DatetimeIndex] = []
    chop: list[str] = []
    squeeze: list[str] = []
    first_date = pd.Timestamp("2023-08-01")
    session_number = 0
    for block_state in block_states:
        for _ in range(2):
            date = first_date + pd.offsets.BDay(session_number)
            indexes.append(
                pd.date_range(
                    f"{date.date()} 09:30",
                    periods=4,
                    freq="15min",
                    tz="America/New_York",
                )
            )
            session_number += 1
            if block_state == "A":
                chop.extend(("TRENDING", "TRENDING", "NEUTRAL", "NEUTRAL"))
                squeeze.extend(
                    ("SQUEEZE_OFF", "SQUEEZE_OFF", "SQUEEZE_ON", "SQUEEZE_ON")
                )
            else:
                chop.extend(("CHOPPY", "CHOPPY", "CHOPPY", "NEUTRAL"))
                squeeze.extend(
                    ("SQUEEZE_ON", "SQUEEZE_ON", "SQUEEZE_ON", "SQUEEZE_OFF")
                )
    index = indexes[0].append(indexes[1:])
    return pd.DataFrame(
        {"chop_segment": chop, "squeeze_state": squeeze},
        index=index,
    )


def atlas(values: pd.DataFrame):
    return build_intraday_regime_atlas(
        {"SPY": values},
        {"SPY": audit()},
        IntradayRegimeAtlasConfig(
            sessions_per_block=2,
            minimum_observations=1,
            minimum_sessions=1,
            minimum_complete_blocks=1,
        ),
    )


def compact_config(**overrides) -> IntradayContextStabilityConfig:
    values = {
        "minimum_complete_blocks": 4,
        "minimum_occupancy_rows_per_block": 1,
        "minimum_transition_rows_per_block": 1,
    }
    values.update(overrides)
    return IntradayContextStabilityConfig(**values)


def test_identical_blocks_have_low_occupancy_and_transition_shift() -> None:
    report = analyze_intraday_context_stability(
        atlas(feature_frame(("A", "A", "A", "A"))),
        compact_config(),
    )

    assert tuple(report.occupancy_drift.columns) == DRIFT_COLUMNS
    assert tuple(report.transition_drift.columns) == DRIFT_COLUMNS
    assert set(report.occupancy_drift["shift"]) == {
        ContextDistributionShift.LOW_SHIFT.value
    }
    assert set(report.transition_drift["shift"]) == {
        ContextDistributionShift.LOW_SHIFT.value
    }
    assert report.occupancy_drift["total_variation"].eq(0.0).all()
    assert report.transition_drift["jensen_shannon_bits"].eq(0.0).all()


def test_alternating_blocks_are_persistent_high() -> None:
    report = analyze_intraday_context_stability(
        atlas(feature_frame(("A", "B", "A", "B"))),
        compact_config(),
    )
    occupancy = report.persistence.loc[
        report.persistence["component"].eq("OCCUPANCY")
    ]
    transitions = report.persistence.loc[
        report.persistence["component"].eq("TRANSITIONS")
    ]

    assert set(occupancy["pattern"]) == {"PERSISTENT_HIGH"}
    assert set(transitions["pattern"]) == {"PERSISTENT_HIGH"}
    assert occupancy.iloc[0]["high_transitions"] == 3


def test_transition_observations_exclude_overnight_boundaries() -> None:
    report = analyze_intraday_context_stability(
        atlas(feature_frame(("A", "A", "A", "A"))),
        compact_config(),
    )

    # Due sedute x tre coppie intraseduta = sei transizioni per blocco.
    assert report.transition_drift["reference_observations"].eq(6).all()
    assert report.transition_drift["current_observations"].eq(6).all()


def test_future_complete_block_does_not_change_past_comparisons() -> None:
    prefix = analyze_intraday_context_stability(
        atlas(feature_frame(("A", "B", "A"))),
        compact_config(minimum_complete_blocks=3),
    )
    full = analyze_intraday_context_stability(
        atlas(feature_frame(("A", "B", "A", "B"))),
        compact_config(minimum_complete_blocks=3),
    )
    columns = [
        "ticker",
        "component",
        "scope",
        "reference_block",
        "current_block",
        "reference_observations",
        "current_observations",
        "total_variation",
        "jensen_shannon_bits",
        "shift",
    ]

    pd.testing.assert_frame_equal(
        prefix.occupancy_drift.loc[:, columns].reset_index(drop=True),
        full.occupancy_drift.loc[
            full.occupancy_drift["current_block"].le(3), columns
        ].reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        prefix.transition_drift.loc[:, columns].reset_index(drop=True),
        full.transition_drift.loc[
            full.transition_drift["current_block"].le(3), columns
        ].reset_index(drop=True),
    )


def test_insufficient_complete_blocks_are_explicit() -> None:
    report = analyze_intraday_context_stability(
        atlas(feature_frame(("A", "A", "A"))),
        compact_config(minimum_complete_blocks=4),
    )

    assert set(report.occupancy_drift["shift"]) == {"INSUFFICIENT"}
    assert set(report.persistence["pattern"]) == {"INSUFFICIENT"}


def test_summary_and_csv_have_no_operational_contract(tmp_path) -> None:
    report = analyze_intraday_context_stability(
        atlas(feature_frame(("A", "A", "A", "A"))),
        compact_config(),
    )
    summary = summarize_context_stability(report)
    destination = tmp_path / "reports" / "context_stability.csv"

    assert tuple(report.persistence.columns) == PERSISTENCE_COLUMNS
    assert summary.iloc[0]["low_or_none"] == 2
    assert write_context_stability(report.persistence, destination) == destination
    assert tuple(pd.read_csv(destination).columns) == PERSISTENCE_COLUMNS
    forbidden = {"signal", "direction", "order", "entry", "exit", "pnl"}
    assert forbidden.isdisjoint(report.persistence.columns)
    with pytest.raises(ValueError, match=".csv"):
        write_context_stability(report.persistence, tmp_path / "report.txt")


def test_permutation_calibration_is_deterministic_and_auditable() -> None:
    source = atlas(feature_frame(("A", "B", "A", "B")))
    config = compact_config(calibration_permutations=24, calibration_seed=17)

    first = analyze_intraday_context_stability(source, config)
    second = analyze_intraday_context_stability(source, config)

    pd.testing.assert_frame_equal(first.occupancy_drift, second.occupancy_drift)
    pd.testing.assert_frame_equal(first.transition_drift, second.transition_drift)
    for table in (first.occupancy_drift, first.transition_drift):
        assert table["null_total_variation_quantile"].notna().all()
        assert table["null_jensen_shannon_quantile"].notna().all()
        assert table["total_variation_excess"].ge(0.0).all()
        assert table["jensen_shannon_excess"].ge(0.0).all()
    with pytest.raises(ValueError, match="almeno 20"):
        compact_config(calibration_permutations=19)
