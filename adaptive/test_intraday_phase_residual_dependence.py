from __future__ import annotations

import pandas as pd
import pytest

from adaptive.intraday_cross_asset_dependence import (
    PHASE_RESIDUAL_CONTRAST_COLUMNS,
    PHASE_RESIDUAL_PAIR_BLOCK_COLUMNS,
    PHASE_RESIDUAL_PAIR_COLUMNS,
    analyze_intraday_cross_asset_dependence,
    write_cross_asset_phase_residual_contrast,
    write_cross_asset_phase_residual_pair_blocks,
    write_cross_asset_phase_residual_pairs,
)
from adaptive.test_intraday_cross_asset_dependence import (
    compact_config,
    synthetic_inputs,
)


def residual_phase_config(**overrides):
    values = {"minimum_phase_block_observations": 8}
    values.update(overrides)
    return compact_config(**values)


def test_phase_residual_contract_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="phase_residual_absolute_range"):
        residual_phase_config(
            moderate_phase_residual_absolute_range=0.40,
            high_phase_residual_absolute_range=0.30,
        )


def test_phase_residual_schemas_and_complete_pair_grid() -> None:
    report = analyze_intraday_cross_asset_dependence(
        *synthetic_inputs(), residual_phase_config()
    )

    assert tuple(report.phase_residual_pair_blocks.columns) == (
        PHASE_RESIDUAL_PAIR_BLOCK_COLUMNS
    )
    assert tuple(report.phase_residual_pairs.columns) == (
        PHASE_RESIDUAL_PAIR_COLUMNS
    )
    assert tuple(report.phase_residual_contrast.columns) == (
        PHASE_RESIDUAL_CONTRAST_COLUMNS
    )
    assert report.phase_residual_pair_blocks.shape[0] == 27
    assert report.phase_residual_pairs.shape[0] == 9
    assert report.phase_residual_contrast.shape[0] == 3


def test_phase_residual_changes_never_cross_phase_boundaries() -> None:
    report = analyze_intraday_cross_asset_dependence(
        *synthetic_inputs(), residual_phase_config()
    )
    observations = (
        report.phase_residual_pairs.groupby("session_phase")["observations"]
        .first()
        .to_dict()
    )

    assert observations == {"OPEN": 36, "MID_SESSION": 204, "CLOSE": 36}


def test_phase_residual_contrast_keeps_stable_proxy_explicit() -> None:
    report = analyze_intraday_cross_asset_dependence(
        *synthetic_inputs(), residual_phase_config()
    )
    row = report.phase_residual_contrast.loc[
        report.phase_residual_contrast["ticker_a"].eq("AAA")
        & report.phase_residual_contrast["ticker_b"].eq("BBB")
    ].iloc[0]

    assert row.phases_available == 3
    assert row.stable_proxy_phases == 3
    assert row.phase_residual_state == "STABLE_PROXY_ACROSS_PHASES"


def test_completed_phase_residual_blocks_are_frozen_with_future_sessions() -> None:
    baseline = analyze_intraday_cross_asset_dependence(
        *synthetic_inputs(8), residual_phase_config()
    ).phase_residual_pair_blocks
    observed = analyze_intraday_cross_asset_dependence(
        *synthetic_inputs(12), residual_phase_config()
    ).phase_residual_pair_blocks
    frozen = baseline.loc[baseline["block_complete"]]
    observed_frozen = observed.loc[
        observed["scope"].eq("ALL_ASSETS_DESCRIPTIVE")
        & observed["block_id"].isin(frozen["block_id"])
    ]

    pd.testing.assert_frame_equal(
        frozen.reset_index(drop=True), observed_frozen.reset_index(drop=True)
    )


def test_phase_residual_outputs_are_deterministic_under_row_shuffle() -> None:
    markets, features, audits = synthetic_inputs()
    config = residual_phase_config()
    baseline = analyze_intraday_cross_asset_dependence(
        markets, features, audits, config
    )
    shuffled_markets = {
        ticker: frame.sample(frac=1.0, random_state=71).reset_index(drop=True)
        for ticker, frame in markets.items()
    }
    shuffled_features = {
        ticker: frame.sample(frac=1.0, random_state=72)
        for ticker, frame in features.items()
    }
    observed = analyze_intraday_cross_asset_dependence(
        shuffled_markets, shuffled_features, audits, config
    )

    pd.testing.assert_frame_equal(
        baseline.phase_residual_pair_blocks,
        observed.phase_residual_pair_blocks,
    )
    pd.testing.assert_frame_equal(
        baseline.phase_residual_pairs,
        observed.phase_residual_pairs,
    )
    pd.testing.assert_frame_equal(
        baseline.phase_residual_contrast,
        observed.phase_residual_contrast,
    )


def test_special_context_asset_remains_outside_ordinary_phase_residual_scope() -> None:
    markets, features, audits = synthetic_inputs()
    markets["VXX"] = markets["CCC"].copy()
    features["VXX"] = features["CCC"].copy()
    audits["VXX"] = audits["CCC"]
    report = analyze_intraday_cross_asset_dependence(
        markets, features, audits, residual_phase_config()
    )
    ordinary = report.phase_residual_pairs.loc[
        report.phase_residual_pairs["scope"].eq("ORDINARY_ASSETS")
    ]

    assert "VXX" not in set(ordinary["ticker_a"]) | set(ordinary["ticker_b"])
    assert set(report.phase_residual_contrast["scope"]) == {
        "ORDINARY_ASSETS",
        "ALL_ASSETS_DESCRIPTIVE",
    }


def test_incomplete_phase_residual_block_is_explicit() -> None:
    report = analyze_intraday_cross_asset_dependence(
        *synthetic_inputs(10), residual_phase_config()
    )
    final = report.phase_residual_pair_blocks.loc[
        report.phase_residual_pair_blocks["block_id"].eq(2)
    ]

    assert len(final) == 9
    assert not final["block_complete"].any()
    assert final["block_state"].eq("INCOMPLETE_BLOCK").all()
    assert final["raw_correlation"].isna().all()
    assert final["residual_correlation"].isna().all()


def test_phase_residual_csv_contract(tmp_path) -> None:
    report = analyze_intraday_cross_asset_dependence(
        *synthetic_inputs(), residual_phase_config()
    )
    paths = (
        write_cross_asset_phase_residual_pair_blocks(
            report.phase_residual_pair_blocks,
            tmp_path / "phase_residual_blocks.csv",
        ),
        write_cross_asset_phase_residual_pairs(
            report.phase_residual_pairs,
            tmp_path / "phase_residual_pairs.csv",
        ),
        write_cross_asset_phase_residual_contrast(
            report.phase_residual_contrast,
            tmp_path / "phase_residual_contrast.csv",
        ),
    )
    expected = (
        PHASE_RESIDUAL_PAIR_BLOCK_COLUMNS,
        PHASE_RESIDUAL_PAIR_COLUMNS,
        PHASE_RESIDUAL_CONTRAST_COLUMNS,
    )

    for path, columns in zip(paths, expected):
        assert tuple(pd.read_csv(path).columns) == columns


def test_phase_residual_outputs_have_no_operational_fields() -> None:
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
            *PHASE_RESIDUAL_PAIR_BLOCK_COLUMNS,
            *PHASE_RESIDUAL_PAIR_COLUMNS,
            *PHASE_RESIDUAL_CONTRAST_COLUMNS,
        )
    )
