from __future__ import annotations

import pandas as pd
import pytest

from adaptive.intraday_cross_asset_dependence import (
    PROXY_BLOCK_SENSITIVITY_COLUMNS,
    PROXY_BLOCK_STABILITY_COLUMNS,
    PROXY_CLUSTER_COLUMNS,
    PROXY_REPRESENTATIVE_INVARIANCE_COLUMNS,
    PROXY_SENSITIVITY_COLUMNS,
    RESIDUAL_PAIR_BLOCK_COLUMNS,
    RESIDUAL_PAIR_COLUMNS,
    analyze_intraday_cross_asset_dependence,
    write_cross_asset_proxy_block_sensitivity,
    write_cross_asset_proxy_block_stability,
    write_cross_asset_proxy_clusters,
    write_cross_asset_proxy_representative_invariance,
    write_cross_asset_proxy_sensitivity,
    write_cross_asset_residual_pair_blocks,
    write_cross_asset_residual_pairs,
)
from adaptive.test_intraday_cross_asset_dependence import (
    compact_config,
    synthetic_inputs,
)


def test_proxy_threshold_contract_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="proxy raw"):
        compact_config(
            proxy_minimum_raw_correlation=0.90,
            proxy_minimum_raw_block_correlation=0.95,
        )
    with pytest.raises(ValueError, match="maximum_proxy_combinations"):
        compact_config(maximum_proxy_combinations=0)


def test_residual_pair_contract_is_complete_and_descriptive() -> None:
    report = analyze_intraday_cross_asset_dependence(
        *synthetic_inputs(), compact_config()
    )

    assert tuple(report.residual_pairs.columns) == RESIDUAL_PAIR_COLUMNS
    assert tuple(report.residual_pair_blocks.columns) == RESIDUAL_PAIR_BLOCK_COLUMNS
    assert len(report.residual_pairs) == 3
    assert report.residual_pairs["eligible_blocks"].eq(3).all()


def test_stable_raw_proxy_pair_forms_group_without_representative() -> None:
    report = analyze_intraday_cross_asset_dependence(
        *synthetic_inputs(), compact_config()
    )
    pair = report.residual_pairs.loc[
        report.residual_pairs["ticker_a"].eq("AAA")
        & report.residual_pairs["ticker_b"].eq("BBB")
    ].iloc[0]
    group = report.proxy_clusters.loc[
        report.proxy_clusters["member_count"].eq(2)
    ]

    assert bool(pair.stable_proxy_link)
    assert pair.residual_pair_state == "STABLE_PROXY_LINK"
    assert set(group["ticker"]) == {"AAA", "BBB"}
    assert group["cluster_members"].eq("AAA;BBB").all()


def test_proxy_collapse_enumerates_every_member_choice() -> None:
    report = analyze_intraday_cross_asset_dependence(
        *synthetic_inputs(), compact_config()
    )
    collapse = report.proxy_sensitivity.loc[
        report.proxy_sensitivity["scenario_type"].eq(
            "COLLAPSE_STABLE_PROXY_GROUPS"
        )
    ]

    assert tuple(report.proxy_sensitivity.columns) == PROXY_SENSITIVITY_COLUMNS
    assert len(collapse) == 2
    assert set(collapse["retained_assets"]) == {"AAA;CCC", "BBB;CCC"}
    assert collapse["scenario_state"].eq(
        "ENUMERATED_WITHOUT_REPRESENTATIVE_SELECTION"
    ).all()


def test_proxy_combination_limit_never_selects_a_partial_grid() -> None:
    report = analyze_intraday_cross_asset_dependence(
        *synthetic_inputs(), compact_config(maximum_proxy_combinations=1)
    )
    collapse = report.proxy_sensitivity.loc[
        report.proxy_sensitivity["scenario_type"].eq(
            "COLLAPSE_STABLE_PROXY_GROUPS"
        )
    ]

    assert len(collapse) == 1
    assert collapse.iloc[0].scenario_id == "COLLAPSE_LIMIT"
    assert collapse.iloc[0].scenario_state == "COMBINATION_LIMIT_EXCEEDED"


def test_completed_residual_blocks_are_frozen_when_future_is_added() -> None:
    baseline = analyze_intraday_cross_asset_dependence(
        *synthetic_inputs(8), compact_config()
    ).residual_pair_blocks
    observed = analyze_intraday_cross_asset_dependence(
        *synthetic_inputs(12), compact_config()
    ).residual_pair_blocks
    frozen = baseline.loc[baseline["block_complete"]]
    observed_frozen = observed.loc[
        observed["scope"].eq("ALL_ASSETS_DESCRIPTIVE")
        & observed["block_id"].isin(frozen["block_id"])
    ]

    pd.testing.assert_frame_equal(
        frozen.reset_index(drop=True), observed_frozen.reset_index(drop=True)
    )


def test_residual_outputs_are_deterministic_under_row_shuffle() -> None:
    markets, features, audits = synthetic_inputs()
    config = compact_config()
    baseline = analyze_intraday_cross_asset_dependence(
        markets, features, audits, config
    )
    shuffled_markets = {
        ticker: frame.sample(frac=1.0, random_state=31).reset_index(drop=True)
        for ticker, frame in markets.items()
    }
    shuffled_features = {
        ticker: frame.sample(frac=1.0, random_state=32)
        for ticker, frame in features.items()
    }
    observed = analyze_intraday_cross_asset_dependence(
        shuffled_markets, shuffled_features, audits, config
    )

    pd.testing.assert_frame_equal(
        baseline.residual_pair_blocks, observed.residual_pair_blocks
    )
    pd.testing.assert_frame_equal(baseline.residual_pairs, observed.residual_pairs)
    pd.testing.assert_frame_equal(baseline.proxy_clusters, observed.proxy_clusters)
    pd.testing.assert_frame_equal(
        baseline.proxy_sensitivity, observed.proxy_sensitivity
    )
    pd.testing.assert_frame_equal(
        baseline.proxy_block_sensitivity,
        observed.proxy_block_sensitivity,
    )
    pd.testing.assert_frame_equal(
        baseline.proxy_block_stability,
        observed.proxy_block_stability,
    )
    pd.testing.assert_frame_equal(
        baseline.proxy_representative_invariance,
        observed.proxy_representative_invariance,
    )


def test_special_context_asset_remains_outside_ordinary_proxy_scope() -> None:
    markets, features, audits = synthetic_inputs()
    markets["VXX"] = markets["CCC"].copy()
    features["VXX"] = features["CCC"].copy()
    audits["VXX"] = audits["CCC"]
    report = analyze_intraday_cross_asset_dependence(
        markets, features, audits, compact_config()
    )
    ordinary = report.proxy_clusters.loc[
        report.proxy_clusters["scope"].eq("ORDINARY_ASSETS")
    ]

    assert "VXX" not in set(ordinary["ticker"])
    assert set(report.proxy_clusters["scope"]) == {
        "ORDINARY_ASSETS",
        "ALL_ASSETS_DESCRIPTIVE",
    }
    assert set(report.proxy_block_sensitivity["scope"]) == {
        "ORDINARY_ASSETS",
        "ALL_ASSETS_DESCRIPTIVE",
    }
    ordinary_blocks = report.proxy_block_sensitivity.loc[
        report.proxy_block_sensitivity["scope"].eq("ORDINARY_ASSETS")
    ]
    assert ordinary_blocks["retained_assets"].str.split(";").map(
        lambda assets: "VXX" not in assets
    ).all()


def test_baseline_sensitivity_reconciles_to_factor_summary() -> None:
    report = analyze_intraday_cross_asset_dependence(
        *synthetic_inputs(), compact_config()
    )
    baseline = report.proxy_sensitivity.loc[
        report.proxy_sensitivity["scenario_type"].eq("BASELINE")
    ].iloc[0]
    factor = report.factor_summary.iloc[0]

    assert baseline.effective_asset_count == pytest.approx(
        factor.raw_effective_asset_count
    )
    assert baseline.common_factor_share == pytest.approx(
        factor.common_factor_share
    )
    assert baseline.residual_effective_asset_count == pytest.approx(
        factor.residual_effective_asset_count
    )


def test_residual_cluster_csv_contract_and_non_operational_fields(tmp_path) -> None:
    report = analyze_intraday_cross_asset_dependence(
        *synthetic_inputs(), compact_config()
    )
    paths = (
        write_cross_asset_residual_pair_blocks(
            report.residual_pair_blocks, tmp_path / "residual_blocks.csv"
        ),
        write_cross_asset_residual_pairs(
            report.residual_pairs, tmp_path / "residual_pairs.csv"
        ),
        write_cross_asset_proxy_clusters(
            report.proxy_clusters, tmp_path / "clusters.csv"
        ),
        write_cross_asset_proxy_sensitivity(
            report.proxy_sensitivity, tmp_path / "sensitivity.csv"
        ),
        write_cross_asset_proxy_block_sensitivity(
            report.proxy_block_sensitivity,
            tmp_path / "block_sensitivity.csv",
        ),
        write_cross_asset_proxy_block_stability(
            report.proxy_block_stability,
            tmp_path / "block_stability.csv",
        ),
        write_cross_asset_proxy_representative_invariance(
            report.proxy_representative_invariance,
            tmp_path / "representative_invariance.csv",
        ),
    )
    expected = (
        RESIDUAL_PAIR_BLOCK_COLUMNS,
        RESIDUAL_PAIR_COLUMNS,
        PROXY_CLUSTER_COLUMNS,
        PROXY_SENSITIVITY_COLUMNS,
        PROXY_BLOCK_SENSITIVITY_COLUMNS,
        PROXY_BLOCK_STABILITY_COLUMNS,
        PROXY_REPRESENTATIVE_INVARIANCE_COLUMNS,
    )
    for path, columns in zip(paths, expected):
        assert tuple(pd.read_csv(path).columns) == columns

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
            *RESIDUAL_PAIR_BLOCK_COLUMNS,
            *RESIDUAL_PAIR_COLUMNS,
            *PROXY_CLUSTER_COLUMNS,
            *PROXY_SENSITIVITY_COLUMNS,
            *PROXY_BLOCK_SENSITIVITY_COLUMNS,
            *PROXY_BLOCK_STABILITY_COLUMNS,
            *PROXY_REPRESENTATIVE_INVARIANCE_COLUMNS,
        )
    )


def test_proxy_block_tables_use_declared_schemas() -> None:
    report = analyze_intraday_cross_asset_dependence(
        *synthetic_inputs(), compact_config()
    )

    assert tuple(report.proxy_block_sensitivity.columns) == (
        PROXY_BLOCK_SENSITIVITY_COLUMNS
    )
    assert tuple(report.proxy_block_stability.columns) == (
        PROXY_BLOCK_STABILITY_COLUMNS
    )
    assert tuple(report.proxy_representative_invariance.columns) == (
        PROXY_REPRESENTATIVE_INVARIANCE_COLUMNS
    )


def test_proxy_scenarios_are_repeated_in_every_complete_block() -> None:
    report = analyze_intraday_cross_asset_dependence(
        *synthetic_inputs(), compact_config()
    )
    blocks = report.proxy_block_sensitivity

    assert len(blocks) == 12
    counts = blocks.groupby("scenario_id").size().to_dict()
    assert counts == {
        "BASELINE": 3,
        "COLLAPSE_001": 3,
        "COLLAPSE_002": 3,
        "LEAVE_CLUSTER_01": 3,
    }


def test_block_baseline_reconciles_to_factor_decomposition() -> None:
    report = analyze_intraday_cross_asset_dependence(
        *synthetic_inputs(), compact_config()
    )
    baseline = report.proxy_block_sensitivity.loc[
        report.proxy_block_sensitivity["scenario_type"].eq("BASELINE")
        & report.proxy_block_sensitivity["block_state"].eq("BLOCK_AVAILABLE")
    ].sort_values("block_id")
    factor = report.factor_blocks.loc[
        report.factor_blocks["block_state"].eq("BLOCK_AVAILABLE")
    ].sort_values("block_id")

    assert list(baseline["block_id"]) == list(factor["block_id"])
    assert list(baseline["effective_asset_count"]) == pytest.approx(
        list(factor["raw_effective_asset_count"])
    )
    assert list(baseline["common_factor_share"]) == pytest.approx(
        list(factor["common_factor_share"])
    )
    assert list(baseline["residual_effective_asset_count"]) == pytest.approx(
        list(factor["residual_effective_asset_count"])
    )


def test_representative_invariance_requires_the_complete_grid() -> None:
    report = analyze_intraday_cross_asset_dependence(
        *synthetic_inputs(), compact_config()
    )
    invariance = report.proxy_representative_invariance

    assert len(invariance) == 3
    assert invariance["expected_scenarios"].eq(2).all()
    assert invariance["observed_scenarios"].eq(2).all()
    assert invariance["invariance_state"].eq(
        "COMPLETE_REPRESENTATIVE_GRID"
    ).all()


def test_incomplete_final_block_is_never_used_for_proxy_invariance() -> None:
    report = analyze_intraday_cross_asset_dependence(
        *synthetic_inputs(10), compact_config()
    )
    final = report.proxy_representative_invariance.sort_values("block_id").iloc[-1]

    assert not bool(final.block_complete)
    assert final.observed_scenarios == 0
    assert final.invariance_state == "INCOMPLETE_BLOCK"
    assert pd.isna(final.effective_asset_count_range)


def test_completed_proxy_blocks_are_frozen_when_future_is_added() -> None:
    baseline = analyze_intraday_cross_asset_dependence(
        *synthetic_inputs(8), compact_config()
    ).proxy_block_sensitivity
    observed = analyze_intraday_cross_asset_dependence(
        *synthetic_inputs(12), compact_config()
    ).proxy_block_sensitivity
    frozen_ids = set(baseline.loc[baseline["block_complete"], "block_id"])
    observed_frozen = observed.loc[observed["block_id"].isin(frozen_ids)]

    pd.testing.assert_frame_equal(
        baseline.reset_index(drop=True), observed_frozen.reset_index(drop=True)
    )


def test_combination_limit_is_explicit_in_every_block() -> None:
    report = analyze_intraday_cross_asset_dependence(
        *synthetic_inputs(), compact_config(maximum_proxy_combinations=1)
    )
    limited = report.proxy_block_sensitivity.loc[
        report.proxy_block_sensitivity["scenario_id"].eq("COLLAPSE_LIMIT")
    ]

    assert len(limited) == 3
    assert limited["block_state"].eq("COMBINATION_LIMIT_EXCEEDED").all()
    assert report.proxy_representative_invariance["invariance_state"].eq(
        "COMBINATION_LIMIT_EXCEEDED"
    ).all()


def test_representative_ranges_are_non_negative_and_descriptive() -> None:
    report = analyze_intraday_cross_asset_dependence(
        *synthetic_inputs(), compact_config()
    )
    invariance = report.proxy_representative_invariance
    range_columns = [
        column
        for column in PROXY_REPRESENTATIVE_INVARIANCE_COLUMNS
        if column.endswith("_range")
    ]

    assert invariance[range_columns].ge(0.0).all().all()
    assert invariance["effective_asset_count_range"].max() < 0.001


def test_proxy_block_stability_counts_only_available_blocks() -> None:
    report = analyze_intraday_cross_asset_dependence(
        *synthetic_inputs(10), compact_config()
    )
    baseline = report.proxy_block_stability.loc[
        report.proxy_block_stability["scenario_id"].eq("BASELINE")
    ].iloc[0]

    assert baseline.eligible_blocks == 2
    assert baseline.block_stability_state == "SUFFICIENT_COMPLETE_BLOCKS"
    assert baseline.minimum_block_observations >= 20


def test_proxy_block_csv_outputs_have_no_duplicate_rows(tmp_path) -> None:
    report = analyze_intraday_cross_asset_dependence(
        *synthetic_inputs(), compact_config()
    )
    paths = (
        write_cross_asset_proxy_block_sensitivity(
            report.proxy_block_sensitivity, tmp_path / "blocks.csv"
        ),
        write_cross_asset_proxy_block_stability(
            report.proxy_block_stability, tmp_path / "stability.csv"
        ),
        write_cross_asset_proxy_representative_invariance(
            report.proxy_representative_invariance,
            tmp_path / "invariance.csv",
        ),
    )

    for path in paths:
        assert not pd.read_csv(path).duplicated().any()
