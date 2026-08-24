from __future__ import annotations

import pandas as pd
import pytest

from adaptive.lorentzian_soft_block_stability import (
    BLOCK_COLUMNS,
    SUMMARY_COLUMNS,
    LorentzianSoftBlockStabilityConfig,
    analyze_lorentzian_soft_block_stability,
    summarize_soft_block_stability_across_assets,
    write_lorentzian_soft_block_details,
    write_lorentzian_soft_block_stability,
)
from adaptive.lorentzian_soft_surface import analyze_lorentzian_soft_surface
from adaptive.test_lorentzian_soft_surface import (
    compact_config,
    synthetic_inputs,
)


def stability_config(**overrides) -> LorentzianSoftBlockStabilityConfig:
    values = {
        "profile": "HL_160_CTX_0.5",
        "sessions_per_block": 5,
        "minimum_queries_per_block": 5,
        "minimum_complete_blocks": 2,
        "neighbors": 4,
        "moderate_minimum_effective_sessions": 3.0,
        "high_minimum_effective_sessions": 2.0,
    }
    values.update(overrides)
    return LorentzianSoftBlockStabilityConfig(**values)


def surface(rows: int = 780):
    normalized, states = synthetic_inputs(rows)
    report = analyze_lorentzian_soft_surface(
        "SPY", normalized, states, compact_config()
    )
    return normalized, report


def test_config_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="raw_distance_change"):
        stability_config(
            moderate_raw_distance_change=0.4,
            high_raw_distance_change=0.3,
        )
    with pytest.raises(ValueError, match="effective_sessions"):
        stability_config(
            moderate_minimum_effective_sessions=5.0,
            high_minimum_effective_sessions=2.0,
        )


def test_block_report_freezes_declared_profile() -> None:
    normalized, soft = surface()
    report = analyze_lorentzian_soft_block_stability(
        "spy", soft.details, normalized.index, stability_config()
    )

    assert not report.blocks.empty
    assert tuple(report.blocks.columns) == BLOCK_COLUMNS
    assert tuple(report.summary.columns) == SUMMARY_COLUMNS
    assert report.blocks["profile"].eq("HL_160_CTX_0.5").all()
    assert report.summary["eligible_blocks"].iloc[0] >= 2
    assert report.research_only


def test_block_boundaries_do_not_change_when_future_sessions_are_added() -> None:
    normalized, soft = surface(650)
    config = stability_config()
    baseline = analyze_lorentzian_soft_block_stability(
        "SPY", soft.details, normalized.index, config
    ).blocks

    last_day = normalized.index[-1].tz_convert("America/New_York").date()
    future_days = pd.bdate_range(last_day + pd.Timedelta(days=1), periods=8)
    future_timestamps = []
    for day in future_days:
        start = pd.Timestamp(day.date(), tz="America/New_York") + pd.Timedelta(
            hours=9, minutes=30
        )
        future_timestamps.extend(pd.date_range(start, periods=26, freq="15min"))
    extended_index = normalized.index.append(pd.DatetimeIndex(future_timestamps))
    observed = analyze_lorentzian_soft_block_stability(
        "SPY", soft.details, extended_index, config
    ).blocks

    frozen = baseline.loc[baseline["block_sessions"].eq(5)]
    observed_frozen = observed.loc[observed["block_id"].isin(frozen["block_id"])]
    pd.testing.assert_frame_equal(
        frozen.reset_index(drop=True),
        observed_frozen.reset_index(drop=True),
    )


def test_shuffled_detail_rows_are_deterministic() -> None:
    normalized, soft = surface()
    config = stability_config()
    baseline = analyze_lorentzian_soft_block_stability(
        "SPY", soft.details, normalized.index, config
    )
    shuffled = soft.details.sample(frac=1.0, random_state=7)
    observed = analyze_lorentzian_soft_block_stability(
        "SPY", shuffled, normalized.index, config
    )

    pd.testing.assert_frame_equal(baseline.blocks, observed.blocks)
    pd.testing.assert_frame_equal(baseline.summary, observed.summary)


def test_cross_asset_summary_and_csv_contract(tmp_path) -> None:
    normalized, soft = surface()
    report = analyze_lorentzian_soft_block_stability(
        "SPY", soft.details, normalized.index, stability_config()
    )
    second_summary = report.summary.copy()
    second_summary["ticker"] = "QQQ"
    summaries = pd.concat([report.summary, second_summary], ignore_index=True)
    cross = summarize_soft_block_stability_across_assets(summaries)

    blocks_path = write_lorentzian_soft_block_details(
        report.blocks, tmp_path / "blocks.csv"
    )
    summary_path = write_lorentzian_soft_block_stability(
        summaries, tmp_path / "summary.csv"
    )

    assert cross["assets"].iloc[0] == 2
    assert tuple(pd.read_csv(blocks_path).columns) == BLOCK_COLUMNS
    assert tuple(pd.read_csv(summary_path).columns) == SUMMARY_COLUMNS


def test_missing_profile_and_timezone_are_rejected() -> None:
    normalized, soft = surface()
    with pytest.raises(ValueError, match="non disponibile"):
        analyze_lorentzian_soft_block_stability(
            "SPY",
            soft.details,
            normalized.index,
            stability_config(profile="HL_999_CTX_9"),
        )
    with pytest.raises(ValueError, match="timezone"):
        analyze_lorentzian_soft_block_stability(
            "SPY",
            soft.details,
            normalized.index.tz_localize(None),
            stability_config(),
        )


def test_output_contract_has_no_operational_fields() -> None:
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
        for column in (*BLOCK_COLUMNS, *SUMMARY_COLUMNS)
    )
