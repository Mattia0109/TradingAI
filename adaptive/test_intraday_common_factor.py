from __future__ import annotations

import pandas as pd
import pytest

from adaptive.intraday_cross_asset_dependence import (
    FACTOR_BLOCK_COLUMNS,
    FACTOR_LOADING_COLUMNS,
    FACTOR_SUMMARY_COLUMNS,
    analyze_intraday_cross_asset_dependence,
    write_cross_asset_factor_blocks,
    write_cross_asset_factor_loadings,
    write_cross_asset_factor_summary,
)
from adaptive.test_intraday_cross_asset_dependence import (
    compact_config,
    synthetic_inputs,
)


def test_config_rejects_invalid_common_factor_thresholds() -> None:
    with pytest.raises(ValueError, match="loading_cosine"):
        compact_config(
            variable_minimum_loading_cosine=0.9,
            stable_minimum_loading_cosine=0.8,
        )
    with pytest.raises(ValueError, match="factor_share_change"):
        compact_config(
            stable_maximum_factor_share_change=0.4,
            variable_maximum_factor_share_change=0.3,
        )


def test_common_factor_loading_contract_is_normalized() -> None:
    report = analyze_intraday_cross_asset_dependence(
        *synthetic_inputs(), compact_config()
    )
    loadings = report.factor_loadings

    assert tuple(loadings.columns) == FACTOR_LOADING_COLUMNS
    assert loadings["absolute_loading_rank"].tolist() == [1, 2, 3]
    assert loadings["loading_share"].sum() == pytest.approx(1.0)
    assert loadings["common_variance_share"].between(0.0, 1.0).all()


def test_common_factor_exposes_nominal_and_residual_breadth() -> None:
    report = analyze_intraday_cross_asset_dependence(
        *synthetic_inputs(), compact_config()
    )
    summary = report.factor_summary.iloc[0]

    assert tuple(report.factor_summary.columns) == FACTOR_SUMMARY_COLUMNS
    assert summary.common_factor_share > 0.60
    assert summary.raw_effective_asset_count < summary.assets
    assert (
        summary.residual_effective_asset_count
        > summary.raw_effective_asset_count
    )
    assert summary.common_mode_state == "STABLE_COMMON_MODE"


def test_completed_factor_blocks_are_frozen_when_future_is_added() -> None:
    baseline = analyze_intraday_cross_asset_dependence(
        *synthetic_inputs(8), compact_config()
    ).factor_blocks
    observed = analyze_intraday_cross_asset_dependence(
        *synthetic_inputs(12), compact_config()
    ).factor_blocks
    observed_frozen = observed.loc[
        observed["scope"].eq("ALL_ASSETS_DESCRIPTIVE")
        & observed["block_id"].isin(baseline["block_id"])
    ]

    pd.testing.assert_frame_equal(
        baseline.reset_index(drop=True), observed_frozen.reset_index(drop=True)
    )


def test_factor_decomposition_is_deterministic_under_row_shuffle() -> None:
    markets, features, audits = synthetic_inputs()
    config = compact_config()
    baseline = analyze_intraday_cross_asset_dependence(
        markets, features, audits, config
    )
    shuffled_markets = {
        ticker: frame.sample(frac=1.0, random_state=14).reset_index(drop=True)
        for ticker, frame in markets.items()
    }
    shuffled_features = {
        ticker: frame.sample(frac=1.0, random_state=15)
        for ticker, frame in features.items()
    }
    observed = analyze_intraday_cross_asset_dependence(
        shuffled_markets, shuffled_features, audits, config
    )

    pd.testing.assert_frame_equal(
        baseline.factor_loadings, observed.factor_loadings
    )
    pd.testing.assert_frame_equal(baseline.factor_blocks, observed.factor_blocks)
    pd.testing.assert_frame_equal(
        baseline.factor_summary, observed.factor_summary
    )


def test_special_context_asset_has_separate_factor_scope() -> None:
    markets, features, audits = synthetic_inputs()
    markets["VXX"] = markets["CCC"].copy()
    features["VXX"] = features["CCC"].copy()
    audits["VXX"] = audits["CCC"]
    report = analyze_intraday_cross_asset_dependence(
        markets, features, audits, compact_config()
    )

    assert set(report.factor_summary["scope"]) == {
        "ORDINARY_ASSETS",
        "ALL_ASSETS_DESCRIPTIVE",
    }
    ordinary = report.factor_loadings.loc[
        report.factor_loadings["scope"].eq("ORDINARY_ASSETS")
    ]
    assert "VXX" not in set(ordinary["ticker"])


def test_common_factor_csv_contract(tmp_path) -> None:
    report = analyze_intraday_cross_asset_dependence(
        *synthetic_inputs(), compact_config()
    )
    loadings_path = write_cross_asset_factor_loadings(
        report.factor_loadings, tmp_path / "loadings.csv"
    )
    blocks_path = write_cross_asset_factor_blocks(
        report.factor_blocks, tmp_path / "blocks.csv"
    )
    summary_path = write_cross_asset_factor_summary(
        report.factor_summary, tmp_path / "summary.csv"
    )

    assert tuple(pd.read_csv(loadings_path).columns) == FACTOR_LOADING_COLUMNS
    assert tuple(pd.read_csv(blocks_path).columns) == FACTOR_BLOCK_COLUMNS
    assert tuple(pd.read_csv(summary_path).columns) == FACTOR_SUMMARY_COLUMNS


def test_common_factor_output_has_no_operational_fields() -> None:
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
            *FACTOR_LOADING_COLUMNS,
            *FACTOR_BLOCK_COLUMNS,
            *FACTOR_SUMMARY_COLUMNS,
        )
    )
