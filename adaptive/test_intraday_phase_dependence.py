from __future__ import annotations

import pandas as pd
import pytest

from adaptive.intraday_cross_asset_dependence import (
    PHASE_FACTOR_BLOCK_COLUMNS,
    PHASE_FACTOR_CONTRAST_COLUMNS,
    PHASE_FACTOR_LOADING_COLUMNS,
    PHASE_FACTOR_SUMMARY_COLUMNS,
    analyze_intraday_cross_asset_dependence,
    write_cross_asset_phase_factor_blocks,
    write_cross_asset_phase_factor_contrast,
    write_cross_asset_phase_factor_loadings,
    write_cross_asset_phase_factor_summary,
)
from adaptive.test_intraday_cross_asset_dependence import (
    compact_config,
    synthetic_inputs,
)


def phase_config(**overrides):
    values = {"minimum_phase_block_observations": 8}
    values.update(overrides)
    return compact_config(**values)


def test_phase_contract_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="minimum_phase_block_observations"):
        phase_config(minimum_phase_block_observations=0)
    with pytest.raises(ValueError, match="phase_factor_share_range"):
        phase_config(
            moderate_phase_factor_share_range=0.20,
            high_phase_factor_share_range=0.10,
        )
    with pytest.raises(ValueError, match="phase_effective_fraction_range"):
        phase_config(
            moderate_phase_effective_fraction_range=0.30,
            high_phase_effective_fraction_range=0.20,
        )


def test_phase_changes_exclude_cross_phase_boundaries() -> None:
    report = analyze_intraday_cross_asset_dependence(
        *synthetic_inputs(), phase_config()
    )
    summary = report.phase_factor_summary.set_index("session_phase")

    assert summary.loc["OPEN", "observations"] == 36
    assert summary.loc["MID_SESSION", "observations"] == 204
    assert summary.loc["CLOSE", "observations"] == 36


def test_phase_factor_schemas_and_loading_normalization() -> None:
    report = analyze_intraday_cross_asset_dependence(
        *synthetic_inputs(), phase_config()
    )

    assert tuple(report.phase_factor_loadings.columns) == (
        PHASE_FACTOR_LOADING_COLUMNS
    )
    assert tuple(report.phase_factor_blocks.columns) == PHASE_FACTOR_BLOCK_COLUMNS
    assert tuple(report.phase_factor_summary.columns) == (
        PHASE_FACTOR_SUMMARY_COLUMNS
    )
    assert tuple(report.phase_factor_contrast.columns) == (
        PHASE_FACTOR_CONTRAST_COLUMNS
    )
    shares = report.phase_factor_loadings.groupby("session_phase")[
        "loading_share"
    ].sum()
    assert list(shares) == pytest.approx([1.0, 1.0, 1.0])


def test_phase_contrast_uses_the_complete_declared_grid() -> None:
    report = analyze_intraday_cross_asset_dependence(
        *synthetic_inputs(),
        phase_config(
            moderate_phase_factor_share_range=0.80,
            high_phase_factor_share_range=0.90,
            moderate_phase_effective_fraction_range=0.80,
            high_phase_effective_fraction_range=0.90,
        ),
    )
    row = report.phase_factor_contrast.iloc[0]

    assert row.phases_available == 3
    assert row.minimum_eligible_blocks == 3
    assert row.phase_structure_state == "LOW_PHASE_HETEROGENEITY"
    assert row.maximum_common_factor_session_phase in {
        "OPEN",
        "MID_SESSION",
        "CLOSE",
    }


def test_completed_phase_blocks_are_frozen_when_future_is_added() -> None:
    baseline = analyze_intraday_cross_asset_dependence(
        *synthetic_inputs(8), phase_config()
    ).phase_factor_blocks
    observed = analyze_intraday_cross_asset_dependence(
        *synthetic_inputs(12), phase_config()
    ).phase_factor_blocks
    frozen = baseline.loc[baseline["block_complete"]]
    observed_frozen = observed.loc[
        observed["scope"].eq("ALL_ASSETS_DESCRIPTIVE")
        & observed["block_id"].isin(frozen["block_id"])
    ]

    pd.testing.assert_frame_equal(
        frozen.reset_index(drop=True), observed_frozen.reset_index(drop=True)
    )


def test_phase_outputs_are_deterministic_under_row_shuffle() -> None:
    markets, features, audits = synthetic_inputs()
    config = phase_config()
    baseline = analyze_intraday_cross_asset_dependence(
        markets, features, audits, config
    )
    shuffled_markets = {
        ticker: frame.sample(frac=1.0, random_state=51).reset_index(drop=True)
        for ticker, frame in markets.items()
    }
    shuffled_features = {
        ticker: frame.sample(frac=1.0, random_state=52)
        for ticker, frame in features.items()
    }
    observed = analyze_intraday_cross_asset_dependence(
        shuffled_markets, shuffled_features, audits, config
    )

    pd.testing.assert_frame_equal(
        baseline.phase_factor_loadings, observed.phase_factor_loadings
    )
    pd.testing.assert_frame_equal(
        baseline.phase_factor_blocks, observed.phase_factor_blocks
    )
    pd.testing.assert_frame_equal(
        baseline.phase_factor_summary, observed.phase_factor_summary
    )
    pd.testing.assert_frame_equal(
        baseline.phase_factor_contrast, observed.phase_factor_contrast
    )


def test_special_context_asset_has_separate_phase_scope() -> None:
    markets, features, audits = synthetic_inputs()
    markets["VXX"] = markets["CCC"].copy()
    features["VXX"] = features["CCC"].copy()
    audits["VXX"] = audits["CCC"]
    report = analyze_intraday_cross_asset_dependence(
        markets, features, audits, phase_config()
    )
    ordinary = report.phase_factor_loadings.loc[
        report.phase_factor_loadings["scope"].eq("ORDINARY_ASSETS")
    ]

    assert "VXX" not in set(ordinary["ticker"])
    assert set(report.phase_factor_contrast["scope"]) == {
        "ORDINARY_ASSETS",
        "ALL_ASSETS_DESCRIPTIVE",
    }


def test_incomplete_phase_block_remains_explicit() -> None:
    report = analyze_intraday_cross_asset_dependence(
        *synthetic_inputs(10), phase_config()
    )
    final = report.phase_factor_blocks.loc[
        report.phase_factor_blocks["block_id"].eq(2)
    ]

    assert len(final) == 3
    assert not final["block_complete"].any()
    assert final["block_state"].eq("INCOMPLETE_BLOCK").all()
    assert final["common_factor_share"].isna().all()


def test_phase_csv_contract(tmp_path) -> None:
    report = analyze_intraday_cross_asset_dependence(
        *synthetic_inputs(), phase_config()
    )
    paths = (
        write_cross_asset_phase_factor_loadings(
            report.phase_factor_loadings, tmp_path / "phase_loadings.csv"
        ),
        write_cross_asset_phase_factor_blocks(
            report.phase_factor_blocks, tmp_path / "phase_blocks.csv"
        ),
        write_cross_asset_phase_factor_summary(
            report.phase_factor_summary, tmp_path / "phase_summary.csv"
        ),
        write_cross_asset_phase_factor_contrast(
            report.phase_factor_contrast, tmp_path / "phase_contrast.csv"
        ),
    )
    expected = (
        PHASE_FACTOR_LOADING_COLUMNS,
        PHASE_FACTOR_BLOCK_COLUMNS,
        PHASE_FACTOR_SUMMARY_COLUMNS,
        PHASE_FACTOR_CONTRAST_COLUMNS,
    )

    for path, columns in zip(paths, expected):
        assert tuple(pd.read_csv(path).columns) == columns


def test_phase_outputs_have_no_operational_fields() -> None:
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
        for column in (
            *PHASE_FACTOR_LOADING_COLUMNS,
            *PHASE_FACTOR_BLOCK_COLUMNS,
            *PHASE_FACTOR_SUMMARY_COLUMNS,
            *PHASE_FACTOR_CONTRAST_COLUMNS,
        )
    )
