from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from adaptive.intraday_feature_readiness import (
    MATRIX_COLUMNS,
    DescriptiveFeatureStatus,
    build_descriptive_feature_matrix,
    non_stable_evidence,
    summarize_asset_matrix,
    write_descriptive_feature_matrix,
)


def persistence(
    feature: str,
    pattern: str,
    latest: str,
    *,
    phase: str | None = None,
) -> pd.DataFrame:
    row = {
        "ticker": "SPY",
        "feature": feature,
        "transitions": 7,
        "elevated_transitions": int(pattern != "LOW_OR_NONE"),
        "high_transitions": int(pattern == "PERSISTENT_HIGH"),
        "longest_elevated_run": int(pattern.startswith("PERSISTENT")),
        "latest_shift": latest,
        "pattern": pattern,
    }
    if phase is not None:
        row["phase"] = phase
    return pd.DataFrame([row])


def report(ticker: str = "SPY") -> SimpleNamespace:
    numeric = persistence("choppiness", "LOW_OR_NONE", "LOW_SHIFT")
    state = persistence(
        "squeeze_state",
        "PERSISTENT_HIGH",
        "HIGH_SHIFT",
    )
    phase = persistence(
        "cmf",
        "ISOLATED_SHIFT",
        "LOW_SHIFT",
        phase="OPEN",
    )
    for table in (numeric, state, phase):
        table["ticker"] = ticker
    return SimpleNamespace(
        ticker=ticker,
        numeric_persistence=numeric,
        state_persistence=state,
        phase_persistence=phase,
    )


def audit(status: str = "READY_DESCRIPTIVE", has_volume: bool = True):
    return SimpleNamespace(
        status=SimpleNamespace(value=status),
        has_volume=has_volume,
    )


def test_matrix_maps_patterns_without_operational_fields() -> None:
    matrix = build_descriptive_feature_matrix(
        [report()],
        {"SPY": audit()},
    )

    assert tuple(matrix.columns) == MATRIX_COLUMNS
    assert set(matrix["readiness"]) == {
        DescriptiveFeatureStatus.STABLE_DESCRIPTIVE.value,
        DescriptiveFeatureStatus.MONITOR_ISOLATED_SHIFT.value,
        DescriptiveFeatureStatus.CONTEXT_REQUIRED.value,
    }
    forbidden = {"signal", "direction", "order", "entry", "exit", "pnl"}
    assert forbidden.isdisjoint(matrix.columns)


def test_price_only_adds_explicit_cmf_not_available() -> None:
    matrix = build_descriptive_feature_matrix(
        [report("SPX")],
        {"SPX": audit(status="PRICE_ONLY", has_volume=False)},
    )
    cmf = matrix.loc[
        matrix["feature"].eq("cmf")
        & matrix["scope"].eq("ALL_SESSION")
    ]

    assert len(cmf) == 1
    assert cmf.iloc[0]["readiness"] == "NOT_AVAILABLE"
    assert cmf.iloc[0]["data_mode"] == "PRICE"


def test_asset_summary_prioritizes_limited_source_and_context() -> None:
    matrix = build_descriptive_feature_matrix(
        [report("VXX"), report("SPY")],
        {
            "VXX": audit(status="LIMITED"),
            "SPY": audit(),
        },
    )
    summary = summarize_asset_matrix(matrix).set_index("ticker")

    assert summary.loc["VXX", "overall"] == "SOURCE_LIMITED"
    assert summary.loc["SPY", "overall"] == "CONTEXT_REQUIRED"
    assert summary.loc["SPY", "context_required"] == 1


def test_non_stable_evidence_excludes_stable_and_insufficient() -> None:
    matrix = build_descriptive_feature_matrix(
        [report()],
        {"SPY": audit()},
    )

    evidence = non_stable_evidence(matrix)

    assert set(evidence["feature"]) == {"cmf", "squeeze_state"}


def test_csv_export_preserves_tidy_matrix(tmp_path) -> None:
    matrix = build_descriptive_feature_matrix(
        [report()],
        {"SPY": audit()},
    )
    destination = tmp_path / "reports" / "matrix.csv"

    observed = write_descriptive_feature_matrix(matrix, destination)

    assert observed == destination
    pd.testing.assert_frame_equal(pd.read_csv(destination), matrix)
    with pytest.raises(ValueError, match=".csv"):
        write_descriptive_feature_matrix(matrix, tmp_path / "matrix.txt")
