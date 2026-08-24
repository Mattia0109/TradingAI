from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from adaptive.intraday_report_audit import (
    REPORT_AUDIT_COLUMNS,
    REPORT_MANIFEST_FILENAME,
    STANDARD_INTRADAY_REPORT_FILENAMES,
    audit_intraday_report_directory,
    write_intraday_report_manifest,
)
from adaptive.run_intraday_report_audit import main, parse_arguments
from adaptive.run_local_intraday_research import STANDARD_REPORT_OUTPUTS


def _write_frame(path: Path, **columns) -> None:
    pd.DataFrame(columns).to_csv(path, index=False)


def test_standard_report_contract_is_unique_and_matches_main_runner() -> None:
    assert len(STANDARD_INTRADAY_REPORT_FILENAMES) == 28
    assert len(set(STANDARD_INTRADAY_REPORT_FILENAMES)) == 28
    assert set(STANDARD_REPORT_OUTPUTS.values()) == set(
        STANDARD_INTRADAY_REPORT_FILENAMES
    )


def test_complete_clean_directory_passes_and_manifest_is_ignored(tmp_path) -> None:
    _write_frame(tmp_path / "alpha.csv", ticker=["AAA"], value=[1.0])
    _write_frame(tmp_path / "beta.csv", ticker=["BBB"], value=[2.0])
    _write_frame(
        tmp_path / REPORT_MANIFEST_FILENAME,
        report_name=["old"],
        audit_state=["PASS"],
    )

    report = audit_intraday_report_directory(
        tmp_path,
        expected_reports=("alpha.csv", "beta.csv"),
    )

    assert report.overall_state == "PASS"
    assert report.expected_reports == 2
    assert report.observed_expected_reports == 2
    assert report.unexpected_reports == 0
    assert report.total_rows == 2
    assert list(report.details.columns) == list(REPORT_AUDIT_COLUMNS)
    assert report.details["sha256"].str.len().eq(64).all()


def test_missing_and_unexpected_csv_require_review(tmp_path) -> None:
    _write_frame(tmp_path / "extra.csv", ticker=["AAA"], value=[1.0])

    report = audit_intraday_report_directory(
        tmp_path,
        expected_reports=("missing.csv",),
    )

    states = dict(
        zip(report.details["report_name"], report.details["audit_state"])
    )
    assert report.overall_state == "REVIEW_REQUIRED"
    assert report.observed_expected_reports == 0
    assert report.unexpected_reports == 1
    assert states == {
        "missing.csv": "MISSING",
        "extra.csv": "UNEXPECTED_CSV",
    }


def test_content_failures_are_reported_without_hiding_multiple_causes(
    tmp_path,
) -> None:
    _write_frame(
        tmp_path / "bad.csv",
        ticker=["AAA", "AAA"],
        signal_score=[np.inf, np.inf],
        unused=[np.nan, np.nan],
    )

    report = audit_intraday_report_directory(
        tmp_path,
        expected_reports=("bad.csv",),
    )

    row = report.details.iloc[0]
    assert report.overall_state == "REVIEW_REQUIRED"
    assert row["duplicate_rows"] == 1
    assert row["nonfinite_numeric_values"] == 2
    assert row["all_nan_columns"] == "unused"
    assert row["prohibited_columns"] == "signal_score"
    assert row["audit_state"] == (
        "DUPLICATE_ROWS+NONFINITE_VALUES+ALL_NAN_COLUMNS+"
        "PROHIBITED_COLUMNS"
    )


def test_zero_row_csv_with_schema_is_valid(tmp_path) -> None:
    pd.DataFrame(columns=["ticker", "state"]).to_csv(
        tmp_path / "empty_rows.csv", index=False
    )

    report = audit_intraday_report_directory(
        tmp_path,
        expected_reports=("empty_rows.csv",),
    )

    assert report.overall_state == "PASS"
    assert report.total_rows == 0


def test_manifest_writer_preserves_exact_schema(tmp_path) -> None:
    _write_frame(tmp_path / "alpha.csv", ticker=["AAA"], value=[1.0])
    report = audit_intraday_report_directory(
        tmp_path,
        expected_reports=("alpha.csv",),
    )

    destination = write_intraday_report_manifest(
        report,
        tmp_path / "nested" / REPORT_MANIFEST_FILENAME,
    )
    restored = pd.read_csv(destination)

    assert destination.exists()
    assert list(restored.columns) == list(REPORT_AUDIT_COLUMNS)
    assert restored.loc[0, "audit_state"] == "PASS"


def test_cli_defaults_manifest_to_report_directory(tmp_path, capsys) -> None:
    for filename in STANDARD_INTRADAY_REPORT_FILENAMES:
        _write_frame(tmp_path / filename, ticker=["AAA"], value=[1.0])

    exit_code = main(["--report-dir", str(tmp_path)])

    assert exit_code == 0
    assert (tmp_path / REPORT_MANIFEST_FILENAME).exists()
    assert "28/28" in capsys.readouterr().out


def test_cli_returns_nonzero_for_incomplete_report_set(tmp_path) -> None:
    _write_frame(tmp_path / STANDARD_INTRADAY_REPORT_FILENAMES[0], value=[1])

    assert main(["--report-dir", str(tmp_path)]) == 1


def test_audit_cli_argument_contract() -> None:
    arguments = parse_arguments(
        ["--report-dir", "reports/v15", "--output", "audit.csv"]
    )

    assert arguments.report_dir == "reports/v15"
    assert arguments.output == "audit.csv"
