from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from adaptive.intraday_cross_asset_dependence import (
    BLOCK_BREADTH_COLUMNS,
    BREADTH_COLUMNS,
    PAIR_COLUMNS,
    IntradayCrossAssetDependenceConfig,
    analyze_intraday_cross_asset_dependence,
    write_cross_asset_block_breadth,
    write_cross_asset_dependence_pairs,
    write_cross_asset_effective_breadth,
)


def compact_config(**overrides) -> IntradayCrossAssetDependenceConfig:
    values = {
        "sessions_per_block": 4,
        "minimum_pair_observations": 20,
        "minimum_pair_sessions": 3,
        "minimum_block_observations": 20,
        "minimum_complete_blocks": 2,
        "special_context_tickers": ("VXX",),
    }
    values.update(overrides)
    return IntradayCrossAssetDependenceConfig(**values)


def synthetic_inputs(sessions: int = 12):
    days = pd.bdate_range("2023-01-03", periods=sessions)
    timestamps = []
    for day in days:
        start = pd.Timestamp(day.date(), tz="America/New_York") + pd.Timedelta(
            hours=9, minutes=30
        )
        timestamps.extend(pd.date_range(start, periods=26, freq="15min"))
    index = pd.DatetimeIndex(timestamps, name="date")
    position = np.arange(len(index), dtype=float)
    base = 0.0007 * np.sin(position / 7.0) + 0.0003 * np.cos(position / 3.0)
    independent = 0.0006 * np.sin(position / 2.1 + 1.3)
    changes = {
        "AAA": base,
        "BBB": base * 1.03 + 0.00001 * np.sin(position / 5.0),
        "CCC": independent,
    }
    markets = {}
    features = {}
    audits = {}
    for ticker, series in changes.items():
        close = 100.0 * np.exp(np.cumsum(series))
        markets[ticker] = pd.DataFrame({"date": index, "close": close})
        if ticker == "CCC":
            chop = np.where((position.astype(int) // 5) % 3 == 0, "CHOPPY", "TRENDING")
            squeeze = np.where((position.astype(int) // 4) % 2 == 0, "SQUEEZE_ON", "SQUEEZE_OFF")
        else:
            chop = np.where((position.astype(int) // 9) % 2 == 0, "TRENDING", "NEUTRAL")
            squeeze = np.where((position.astype(int) // 11) % 2 == 0, "SQUEEZE_OFF", "SQUEEZE_ON")
        features[ticker] = pd.DataFrame(
            {
                "choppiness": 50.0 + series * 10_000.0,
                "squeeze_momentum_pct_close": series,
                "cmf": np.tanh(series * 1_000.0),
                "chop_segment": chop,
                "squeeze_state": squeeze,
            },
            index=index,
        )
        audits[ticker] = SimpleNamespace(
            status=SimpleNamespace(value="READY_DESCRIPTIVE"),
            has_volume=True,
        )
    return markets, features, audits


def test_config_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="absolute_correlation"):
        compact_config(
            moderate_absolute_correlation=0.8,
            high_absolute_correlation=0.7,
        )
    with pytest.raises(ValueError, match="dominant_share"):
        compact_config(
            distributed_dominant_share=0.7,
            concentrated_dominant_share=0.6,
        )


def test_pair_report_detects_shared_evidence_without_future_outcomes() -> None:
    markets, features, audits = synthetic_inputs()
    report = analyze_intraday_cross_asset_dependence(
        markets, features, audits, compact_config()
    )
    pair = report.pairs.loc[
        report.pairs["ticker_a"].eq("AAA")
        & report.pairs["ticker_b"].eq("BBB")
    ].iloc[0]

    assert tuple(report.pairs.columns) == PAIR_COLUMNS
    assert pair.close_change_correlation > 0.99
    assert pair.joint_context_nmi > 0.99
    assert pair.dependence_state == "HIGH_DEPENDENCE"
    assert report.research_only


def test_effective_breadth_is_spectral_and_bounded() -> None:
    markets, features, audits = synthetic_inputs()
    report = analyze_intraday_cross_asset_dependence(
        markets, features, audits, compact_config()
    )
    row = report.breadth.loc[
        report.breadth["scope"].eq("ALL_ASSETS_DESCRIPTIVE")
    ].iloc[0]

    assert tuple(report.breadth.columns) == BREADTH_COLUMNS
    assert 1.0 <= row.effective_asset_count <= row.assets
    assert 0.0 < row.effective_asset_fraction <= 1.0
    assert 0.0 < row.dominant_component_share <= 1.0
    assert row.effective_asset_count < row.assets


def test_completed_block_metrics_do_not_change_with_future_sessions() -> None:
    markets, features, audits = synthetic_inputs(8)
    baseline = analyze_intraday_cross_asset_dependence(
        markets, features, audits, compact_config()
    ).block_breadth
    extended_inputs = synthetic_inputs(12)
    observed = analyze_intraday_cross_asset_dependence(
        *extended_inputs, compact_config()
    ).block_breadth

    frozen = baseline.loc[baseline["block_complete"]]
    observed_frozen = observed.loc[
        observed["scope"].eq("ALL_ASSETS_DESCRIPTIVE")
        & observed["block_id"].isin(frozen["block_id"])
    ]
    pd.testing.assert_frame_equal(
        frozen.reset_index(drop=True), observed_frozen.reset_index(drop=True)
    )


def test_input_row_order_does_not_change_results() -> None:
    markets, features, audits = synthetic_inputs()
    config = compact_config()
    baseline = analyze_intraday_cross_asset_dependence(
        markets, features, audits, config
    )
    shuffled_markets = {
        ticker: frame.sample(frac=1.0, random_state=4).reset_index(drop=True)
        for ticker, frame in markets.items()
    }
    shuffled_features = {
        ticker: frame.sample(frac=1.0, random_state=5)
        for ticker, frame in features.items()
    }
    observed = analyze_intraday_cross_asset_dependence(
        shuffled_markets, shuffled_features, audits, config
    )

    pd.testing.assert_frame_equal(baseline.pairs, observed.pairs)
    pd.testing.assert_frame_equal(baseline.block_breadth, observed.block_breadth)
    pd.testing.assert_frame_equal(baseline.breadth, observed.breadth)


def test_special_context_asset_is_excluded_from_ordinary_scope() -> None:
    markets, features, audits = synthetic_inputs()
    markets["VXX"] = markets["CCC"].copy()
    features["VXX"] = features["CCC"].copy()
    audits["VXX"] = audits["CCC"]
    report = analyze_intraday_cross_asset_dependence(
        markets, features, audits, compact_config()
    )
    ordinary = report.breadth.loc[
        report.breadth["scope"].eq("ORDINARY_ASSETS")
    ].iloc[0]

    assert "VXX" not in ordinary.asset_list.split(";")
    assert report.breadth["scope"].eq("ALL_ASSETS_DESCRIPTIVE").any()


def test_csv_contract_and_missing_timezone_validation(tmp_path) -> None:
    markets, features, audits = synthetic_inputs()
    report = analyze_intraday_cross_asset_dependence(
        markets, features, audits, compact_config()
    )
    pair_path = write_cross_asset_dependence_pairs(
        report.pairs, tmp_path / "pairs.csv"
    )
    block_path = write_cross_asset_block_breadth(
        report.block_breadth, tmp_path / "blocks.csv"
    )
    breadth_path = write_cross_asset_effective_breadth(
        report.breadth, tmp_path / "breadth.csv"
    )

    assert tuple(pd.read_csv(pair_path).columns) == PAIR_COLUMNS
    assert tuple(pd.read_csv(block_path).columns) == BLOCK_BREADTH_COLUMNS
    assert tuple(pd.read_csv(breadth_path).columns) == BREADTH_COLUMNS

    bad_features = dict(features)
    bad_features["AAA"] = features["AAA"].copy()
    bad_features["AAA"].index = bad_features["AAA"].index.tz_localize(None)
    with pytest.raises(ValueError, match="timezone"):
        analyze_intraday_cross_asset_dependence(
            markets, bad_features, audits, compact_config()
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
        for column in (*PAIR_COLUMNS, *BLOCK_BREADTH_COLUMNS, *BREADTH_COLUMNS)
    )
