"""Audit riproducibile dei CSV descrittivi prodotti dalla ricerca intraday."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


STANDARD_INTRADAY_REPORT_FILENAMES = (
    "intraday_feature_stability_matrix.csv",
    "intraday_regime_atlas.csv",
    "intraday_regime_transitions.csv",
    "intraday_context_stability.csv",
    "lorentzian_neighborhood_stability.csv",
    "lorentzian_soft_surface.csv",
    "lorentzian_soft_block_details.csv",
    "lorentzian_soft_block_stability.csv",
    "intraday_cross_asset_dependence.csv",
    "intraday_cross_asset_block_breadth.csv",
    "intraday_cross_asset_effective_breadth.csv",
    "intraday_common_factor_loadings.csv",
    "intraday_common_factor_blocks.csv",
    "intraday_common_factor_summary.csv",
    "intraday_residual_pair_blocks.csv",
    "intraday_residual_pairs.csv",
    "intraday_stable_proxy_clusters.csv",
    "intraday_proxy_sensitivity.csv",
    "intraday_proxy_block_sensitivity.csv",
    "intraday_proxy_block_stability.csv",
    "intraday_proxy_representative_invariance.csv",
    "intraday_phase_factor_loadings.csv",
    "intraday_phase_factor_blocks.csv",
    "intraday_phase_factor_summary.csv",
    "intraday_phase_factor_contrast.csv",
    "intraday_phase_residual_pair_blocks.csv",
    "intraday_phase_residual_pairs.csv",
    "intraday_phase_residual_contrast.csv",
)

REPORT_MANIFEST_FILENAME = "intraday_report_manifest.csv"

REPORT_AUDIT_COLUMNS = (
    "report_name",
    "relative_path",
    "expected",
    "file_bytes",
    "sha256",
    "rows",
    "columns",
    "duplicate_rows",
    "nonfinite_numeric_values",
    "all_nan_columns",
    "prohibited_columns",
    "audit_state",
)

# I nomi vengono controllati per token/sottostringa. Non si vietano misure
# descrittive di prezzo o variazione contemporanea.
PROHIBITED_COLUMN_TOKENS = (
    "signal",
    "order",
    "position",
    "entry",
    "exit",
    "stop",
    "leverage",
    "pnl",
    "profit",
    "forecast",
    "prediction",
    "outcome",
    "trade",
)


@dataclass(frozen=True)
class IntradayReportAudit:
    """Risultato immutabile dell'audit di una directory di report."""

    details: pd.DataFrame
    expected_reports: int
    observed_expected_reports: int
    unexpected_reports: int
    total_rows: int
    overall_state: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prohibited_columns(columns: Iterable[object]) -> list[str]:
    flagged = []
    for column in columns:
        normalized = str(column).strip().lower()
        if any(token in normalized for token in PROHIBITED_COLUMN_TOKENS):
            flagged.append(str(column))
    return sorted(flagged)


def _audit_existing_csv(
    path: Path,
    *,
    report_name: str,
    expected: bool,
) -> dict:
    base = {
        "report_name": report_name,
        "relative_path": path.name,
        "expected": expected,
        "file_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
        "rows": 0,
        "columns": 0,
        "duplicate_rows": 0,
        "nonfinite_numeric_values": 0,
        "all_nan_columns": "",
        "prohibited_columns": "",
        "audit_state": "PASS" if expected else "UNEXPECTED_CSV",
    }
    try:
        frame = pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
        base["audit_state"] = "READ_ERROR"
        return base

    numeric = frame.select_dtypes(include=[np.number])
    nonfinite = (
        int(np.isinf(numeric.to_numpy(dtype=float)).sum())
        if not numeric.empty
        else 0
    )
    all_nan = (
        sorted(
            str(column)
            for column in frame.columns
            if frame[column].isna().all()
        )
        if len(frame)
        else []
    )
    prohibited = _prohibited_columns(frame.columns)
    duplicate_rows = int(frame.duplicated().sum())
    base.update(
        {
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "duplicate_rows": duplicate_rows,
            "nonfinite_numeric_values": nonfinite,
            "all_nan_columns": ";".join(all_nan),
            "prohibited_columns": ";".join(prohibited),
        }
    )

    failures = []
    if not expected:
        failures.append("UNEXPECTED_CSV")
    if base["file_bytes"] == 0 or len(frame.columns) == 0:
        failures.append("EMPTY_SCHEMA")
    if duplicate_rows:
        failures.append("DUPLICATE_ROWS")
    if nonfinite:
        failures.append("NONFINITE_VALUES")
    if all_nan:
        failures.append("ALL_NAN_COLUMNS")
    if prohibited:
        failures.append("PROHIBITED_COLUMNS")
    base["audit_state"] = "+".join(failures) if failures else "PASS"
    return base


def audit_intraday_report_directory(
    report_directory: str | Path,
    *,
    expected_reports: Iterable[str] = STANDARD_INTRADAY_REPORT_FILENAMES,
    manifest_filename: str = REPORT_MANIFEST_FILENAME,
) -> IntradayReportAudit:
    """Verifica completezza, integrita' e natura descrittiva dei CSV."""

    root = Path(report_directory)
    if not root.is_dir():
        raise ValueError(f"Directory report non valida: {root}")

    expected = tuple(sorted({Path(name).name for name in expected_reports}))
    if not expected:
        raise ValueError("expected_reports non puo' essere vuoto.")
    observed = {
        path.name: path
        for path in sorted(root.glob("*.csv"))
        if path.name != manifest_filename
    }

    rows: list[dict] = []
    for name in expected:
        path = observed.pop(name, None)
        if path is None:
            rows.append(
                {
                    "report_name": name,
                    "relative_path": name,
                    "expected": True,
                    "file_bytes": 0,
                    "sha256": "",
                    "rows": 0,
                    "columns": 0,
                    "duplicate_rows": 0,
                    "nonfinite_numeric_values": 0,
                    "all_nan_columns": "",
                    "prohibited_columns": "",
                    "audit_state": "MISSING",
                }
            )
            continue
        rows.append(
            _audit_existing_csv(path, report_name=name, expected=True)
        )

    for name, path in sorted(observed.items()):
        rows.append(
            _audit_existing_csv(path, report_name=name, expected=False)
        )

    details = pd.DataFrame(rows, columns=REPORT_AUDIT_COLUMNS).sort_values(
        ["expected", "report_name"],
        ascending=[False, True],
        kind="mergesort",
        ignore_index=True,
    )
    passed = details["audit_state"].eq("PASS")
    expected_mask = details["expected"].eq(True)
    expected_observed = int(
        (expected_mask & ~details["audit_state"].eq("MISSING")).sum()
    )
    unexpected_count = int((~expected_mask).sum())
    overall_state = "PASS" if bool(passed.all()) else "REVIEW_REQUIRED"
    return IntradayReportAudit(
        details=details,
        expected_reports=len(expected),
        observed_expected_reports=expected_observed,
        unexpected_reports=unexpected_count,
        total_rows=int(details.loc[expected_mask, "rows"].sum()),
        overall_state=overall_state,
    )


def write_intraday_report_manifest(
    report: IntradayReportAudit,
    output_path: str | Path,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    report.details.to_csv(destination, index=False)
    return destination
