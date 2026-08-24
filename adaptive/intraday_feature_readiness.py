"""Matrice descrittiva asset x feature x fase.

La matrice consolida evidenze di disponibilita' e drift gia' calcolate senza
aggiungere etichette direzionali, outcome futuri, segnali, operazioni, size,
stop, leva o P&L.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd


class DescriptiveFeatureStatus(str, Enum):
    STABLE_DESCRIPTIVE = "STABLE_DESCRIPTIVE"
    MONITOR_ISOLATED_SHIFT = "MONITOR_ISOLATED_SHIFT"
    CONTEXT_REQUIRED = "CONTEXT_REQUIRED"
    INSUFFICIENT = "INSUFFICIENT"
    NOT_AVAILABLE = "NOT_AVAILABLE"


MATRIX_COLUMNS = (
    "ticker",
    "source_status",
    "data_mode",
    "feature",
    "feature_kind",
    "scope",
    "transitions",
    "elevated_transitions",
    "high_transitions",
    "longest_elevated_run",
    "latest_shift",
    "pattern",
    "readiness",
)


def _readiness_from_pattern(pattern: str) -> DescriptiveFeatureStatus:
    normalized = str(pattern).upper().strip()
    if normalized == "LOW_OR_NONE":
        return DescriptiveFeatureStatus.STABLE_DESCRIPTIVE
    if normalized == "ISOLATED_SHIFT":
        return DescriptiveFeatureStatus.MONITOR_ISOLATED_SHIFT
    if normalized in {"PERSISTENT_ELEVATED", "PERSISTENT_HIGH"}:
        return DescriptiveFeatureStatus.CONTEXT_REQUIRED
    if normalized == "NOT_AVAILABLE":
        return DescriptiveFeatureStatus.NOT_AVAILABLE
    return DescriptiveFeatureStatus.INSUFFICIENT


def _audit_status(audit) -> tuple[str, str]:
    source_status = getattr(audit.status, "value", str(audit.status))
    mode = "OHLCV" if bool(audit.has_volume) else "PRICE"
    return str(source_status), mode


def _persistence_rows(
    ticker: str,
    source_status: str,
    data_mode: str,
    table: pd.DataFrame,
    feature_kind: str,
    phase_table: bool = False,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if table.empty:
        return rows
    required = {
        "feature",
        "transitions",
        "elevated_transitions",
        "high_transitions",
        "longest_elevated_run",
        "latest_shift",
        "pattern",
    }
    if phase_table:
        required.add("phase")
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Colonne persistence mancanti: {missing}.")
    for item in table.itertuples(index=False):
        pattern = str(item.pattern)
        rows.append(
            {
                "ticker": ticker,
                "source_status": source_status,
                "data_mode": data_mode,
                "feature": str(item.feature),
                "feature_kind": feature_kind,
                "scope": str(item.phase) if phase_table else "ALL_SESSION",
                "transitions": int(item.transitions),
                "elevated_transitions": int(item.elevated_transitions),
                "high_transitions": int(item.high_transitions),
                "longest_elevated_run": int(item.longest_elevated_run),
                "latest_shift": str(item.latest_shift),
                "pattern": pattern,
                "readiness": _readiness_from_pattern(pattern).value,
            }
        )
    return rows


def build_descriptive_feature_matrix(
    stability_reports: Iterable,
    source_audits: Mapping[str, object],
) -> pd.DataFrame:
    """Unisce persistence numerica, di stato e per fase in forma tidy."""

    reports = {str(report.ticker).upper(): report for report in stability_reports}
    rows: list[dict[str, object]] = []
    for ticker, audit in sorted(source_audits.items()):
        normalized_ticker = str(ticker).upper()
        report = reports.get(normalized_ticker)
        if report is None:
            continue
        source_status, data_mode = _audit_status(audit)
        rows.extend(
            _persistence_rows(
                normalized_ticker,
                source_status,
                data_mode,
                report.numeric_persistence,
                "NUMERIC",
            )
        )
        rows.extend(
            _persistence_rows(
                normalized_ticker,
                source_status,
                data_mode,
                report.state_persistence,
                "STATE",
            )
        )
        rows.extend(
            _persistence_rows(
                normalized_ticker,
                source_status,
                data_mode,
                report.phase_persistence,
                "NUMERIC",
                phase_table=True,
            )
        )
        if not bool(audit.has_volume):
            rows.append(
                {
                    "ticker": normalized_ticker,
                    "source_status": source_status,
                    "data_mode": data_mode,
                    "feature": "cmf",
                    "feature_kind": "NUMERIC",
                    "scope": "ALL_SESSION",
                    "transitions": 0,
                    "elevated_transitions": 0,
                    "high_transitions": 0,
                    "longest_elevated_run": 0,
                    "latest_shift": "NOT_AVAILABLE",
                    "pattern": "NOT_AVAILABLE",
                    "readiness": DescriptiveFeatureStatus.NOT_AVAILABLE.value,
                }
            )
    if not rows:
        return pd.DataFrame(columns=MATRIX_COLUMNS)
    matrix = pd.DataFrame(rows, columns=MATRIX_COLUMNS)
    return matrix.sort_values(
        ["ticker", "feature", "scope", "feature_kind"],
        kind="mergesort",
    ).reset_index(drop=True)


def summarize_asset_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """Conta gli stati per asset senza trasformarli in approvazioni."""

    if matrix.empty:
        return pd.DataFrame(
            columns=(
                "ticker",
                "source_status",
                "stable",
                "isolated",
                "context_required",
                "insufficient",
                "not_available",
                "overall",
            )
        )
    rows: list[dict[str, object]] = []
    for ticker, group in matrix.groupby("ticker", sort=True):
        source_status = str(group["source_status"].iloc[0])
        counts = group["readiness"].value_counts()
        context_required = int(
            counts.get(DescriptiveFeatureStatus.CONTEXT_REQUIRED.value, 0)
        )
        isolated = int(
            counts.get(DescriptiveFeatureStatus.MONITOR_ISOLATED_SHIFT.value, 0)
        )
        insufficient = int(
            counts.get(DescriptiveFeatureStatus.INSUFFICIENT.value, 0)
        )
        if source_status == "LIMITED":
            overall = "SOURCE_LIMITED"
        elif context_required:
            overall = DescriptiveFeatureStatus.CONTEXT_REQUIRED.value
        elif isolated:
            overall = DescriptiveFeatureStatus.MONITOR_ISOLATED_SHIFT.value
        elif insufficient:
            overall = DescriptiveFeatureStatus.INSUFFICIENT.value
        else:
            overall = DescriptiveFeatureStatus.STABLE_DESCRIPTIVE.value
        rows.append(
            {
                "ticker": ticker,
                "source_status": source_status,
                "stable": int(
                    counts.get(
                        DescriptiveFeatureStatus.STABLE_DESCRIPTIVE.value,
                        0,
                    )
                ),
                "isolated": isolated,
                "context_required": context_required,
                "insufficient": insufficient,
                "not_available": int(
                    counts.get(DescriptiveFeatureStatus.NOT_AVAILABLE.value, 0)
                ),
                "overall": overall,
            }
        )
    return pd.DataFrame(rows)


def non_stable_evidence(matrix: pd.DataFrame) -> pd.DataFrame:
    """Restituisce solo drift isolati/persistenti; esclude insufficienza e NaN."""

    if matrix.empty:
        return matrix.copy()
    selected = matrix["readiness"].isin(
        {
            DescriptiveFeatureStatus.MONITOR_ISOLATED_SHIFT.value,
            DescriptiveFeatureStatus.CONTEXT_REQUIRED.value,
        }
    )
    return matrix.loc[selected].reset_index(drop=True)


def write_descriptive_feature_matrix(
    matrix: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Esporta il CSV dichiarato, creando soltanto la directory necessaria."""

    destination = Path(path).expanduser()
    if destination.suffix.lower() != ".csv":
        raise ValueError("matrix_output deve terminare con .csv.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(destination, index=False)
    return destination
