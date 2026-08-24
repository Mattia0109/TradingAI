from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from adaptive.intraday_regime_atlas import (
    ATLAS_COLUMNS,
    TRANSITION_COLUMNS,
    DescriptiveContextCoverage,
    IntradayRegimeAtlasConfig,
    build_intraday_regime_atlas,
    summarize_regime_atlas,
    write_regime_atlas,
    write_regime_transitions,
)


def audit(
    status: str = "READY_DESCRIPTIVE",
    has_volume: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        status=SimpleNamespace(value=status),
        has_volume=has_volume,
    )


def feature_frame(
    dates: tuple[str, ...] = ("2023-08-01", "2023-08-02"),
    *,
    bars_per_session: int = 4,
) -> pd.DataFrame:
    indexes: list[pd.DatetimeIndex] = []
    for date in dates:
        indexes.append(
            pd.date_range(
                f"{date} 09:30",
                periods=bars_per_session,
                freq="15min",
                tz="America/New_York",
            )
        )
    index = indexes[0].append(indexes[1:])
    chop_cycle = ("TRENDING", "TRENDING", "NEUTRAL", "CHOPPY")
    squeeze_cycle = (
        "SQUEEZE_ON",
        "SQUEEZE_ON",
        "SQUEEZE_OFF",
        "NEUTRAL",
    )
    return pd.DataFrame(
        {
            "chop_segment": [
                chop_cycle[position % len(chop_cycle)]
                for position in range(len(index))
            ],
            "squeeze_state": [
                squeeze_cycle[position % len(squeeze_cycle)]
                for position in range(len(index))
            ],
        },
        index=index,
    )


def compact_config(**overrides) -> IntradayRegimeAtlasConfig:
    values = {
        "sessions_per_block": 2,
        "minimum_observations": 2,
        "minimum_sessions": 1,
        "minimum_complete_blocks": 1,
    }
    values.update(overrides)
    return IntradayRegimeAtlasConfig(**values)


def test_atlas_is_descriptive_and_marks_price_only_cmf() -> None:
    report = build_intraday_regime_atlas(
        {"SPX": feature_frame()},
        {"SPX": audit("PRICE_ONLY", has_volume=False)},
        compact_config(),
    )

    assert tuple(report.atlas.columns) == ATLAS_COLUMNS
    assert tuple(report.transitions.columns) == TRANSITION_COLUMNS
    assert report.research_only is True
    assert set(report.assignments["cmf_availability"]) == {"NOT_AVAILABLE"}
    assert set(report.assignments["data_mode"]) == {"PRICE"}
    assert (
        DescriptiveContextCoverage.ADEQUATE_DESCRIPTIVE_COVERAGE.value
        in set(report.atlas["coverage"])
    )
    forbidden = {"signal", "direction", "order", "entry", "exit", "pnl"}
    assert forbidden.isdisjoint(report.atlas.columns)
    assert forbidden.isdisjoint(report.transitions.columns)


def test_transitions_never_cross_sessions_or_missing_bars() -> None:
    values = feature_frame(bars_per_session=4)
    # Rimuove la terza barra della prima seduta: 09:45 -> 10:15 non deve
    # diventare una transizione artificiale.
    values = values.drop(pd.Timestamp("2023-08-01 10:00", tz="America/New_York"))
    report = build_intraday_regime_atlas(
        {"SPY": values},
        {"SPY": audit()},
        compact_config(minimum_observations=1),
    )

    # Prima seduta: una transizione valida; seconda: tre. Nessuna overnight.
    assert int(report.transitions["observations"].sum()) == 4
    assignments = report.assignments
    expected_max = sum(
        max(int(len(group)) - 1, 0)
        for _, group in assignments.groupby("session_date")
    )
    assert int(report.transitions["observations"].sum()) <= expected_max


def test_invalid_warmup_state_cannot_bridge_a_gap() -> None:
    values = feature_frame(dates=("2023-08-01",), bars_per_session=4)
    values.loc[
        pd.Timestamp("2023-08-01 10:00", tz="America/New_York"),
        "chop_segment",
    ] = "INSUFFICIENT"
    report = build_intraday_regime_atlas(
        {"SPY": values},
        {"SPY": audit()},
        compact_config(sessions_per_block=1, minimum_observations=1),
    )

    assert len(report.assignments) == 3
    assert int(report.transitions["observations"].sum()) == 1


def test_future_session_does_not_reassign_past_contexts_or_blocks() -> None:
    prefix_values = feature_frame(dates=("2023-08-01", "2023-08-02"))
    full_values = feature_frame(
        dates=("2023-08-01", "2023-08-02", "2023-08-03")
    )
    prefix = build_intraday_regime_atlas(
        {"SPY": prefix_values},
        {"SPY": audit()},
        compact_config(),
    ).assignments
    full = build_intraday_regime_atlas(
        {"SPY": full_values},
        {"SPY": audit()},
        compact_config(),
    ).assignments

    pd.testing.assert_frame_equal(
        prefix.reset_index(drop=True),
        full.iloc[: len(prefix)].reset_index(drop=True),
    )


def test_cross_asset_summary_counts_observation_without_independence_claim() -> None:
    report = build_intraday_regime_atlas(
        {
            "SPY": feature_frame(),
            "SPX": feature_frame(),
        },
        {
            "SPY": audit(),
            "SPX": audit("PRICE_ONLY", has_volume=False),
        },
        compact_config(),
    )

    assert report.cross_asset["assets_observed"].max() == 2
    assert report.cross_asset["ohlcv_assets"].max() == 1
    assert report.cross_asset["price_only_assets"].max() == 1
    summary = summarize_regime_atlas(report).set_index("ticker")
    assert summary.loc["SPY", "assignments"] == 8
    assert summary.loc["SPX", "source_status"] == "PRICE_ONLY"


def test_exports_are_deterministic_csv_contracts(tmp_path) -> None:
    report = build_intraday_regime_atlas(
        {"SPY": feature_frame()},
        {"SPY": audit()},
        compact_config(),
    )
    atlas_path = tmp_path / "reports" / "atlas.csv"
    transition_path = tmp_path / "reports" / "transitions.csv"

    assert write_regime_atlas(report.atlas, atlas_path) == atlas_path
    assert write_regime_transitions(
        report.transitions,
        transition_path,
    ) == transition_path
    assert tuple(pd.read_csv(atlas_path).columns) == ATLAS_COLUMNS
    assert tuple(pd.read_csv(transition_path).columns) == TRANSITION_COLUMNS
    with pytest.raises(ValueError, match=".csv"):
        write_regime_atlas(report.atlas, tmp_path / "atlas.txt")


def test_configuration_rejects_invalid_temporal_contract() -> None:
    with pytest.raises(ValueError, match="positivo"):
        IntradayRegimeAtlasConfig(interval_minutes=0)
    with pytest.raises(ValueError, match="non divide"):
        IntradayRegimeAtlasConfig(interval_minutes=17)
