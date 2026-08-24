"""Audit temporale tra due freeze di ricerca intraday.

`PASS` indica esclusivamente che esiste un nuovo blocco dati intatto e
sufficientemente esteso per una successiva revisione scientifica cieca.  Non
indica accuratezza, redditivita' o idoneita' operativa.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping

import pandas as pd

from adaptive.research_freeze import (
    canonical_json_bytes,
    load_research_freeze,
    verify_research_freeze,
)


FORWARD_READINESS_SCHEMA_VERSION = 1
FORWARD_READINESS_DETAILS_FILENAME = "intraday_forward_readiness_details.csv"
FORWARD_READINESS_SUMMARY_FILENAME = "intraday_forward_readiness_summary.json"


class ForwardReadinessState(str, Enum):
    PASS = "PASS"
    INCONCLUSIVE = "INCONCLUSIVE"
    FAIL = "FAIL"


@dataclass(frozen=True)
class ForwardReadinessConfig:
    """Soglie di copertura, non soglie di performance."""

    minimum_common_forward_sessions: int = 60
    sessions_per_checkpoint: int = 20
    minimum_checkpoints: int = 3
    minimum_eligible_sources: int = 5

    def __post_init__(self) -> None:
        for name in (
            "minimum_common_forward_sessions",
            "sessions_per_checkpoint",
            "minimum_checkpoints",
            "minimum_eligible_sources",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} deve essere positivo.")
        required = self.sessions_per_checkpoint * self.minimum_checkpoints
        if self.minimum_common_forward_sessions < required:
            raise ValueError(
                "minimum_common_forward_sessions deve coprire tutti i checkpoint."
            )


@dataclass(frozen=True)
class ForwardReadinessAudit:
    state: ForwardReadinessState
    baseline_freeze_id: str
    candidate_freeze_id: str
    evidence_id: str
    specification_unchanged: bool
    implementation_unchanged: bool
    eligible_sources: int
    common_forward_sessions: int
    complete_checkpoints: int
    details: pd.DataFrame
    reasons: tuple[str, ...]
    research_only: bool = True


DETAIL_COLUMNS = (
    "ticker",
    "source_status_baseline",
    "source_status_candidate",
    "comparison_mode",
    "baseline_sessions",
    "candidate_sessions",
    "overlap_sessions",
    "unchanged_overlap_sessions",
    "changed_overlap_sessions",
    "missing_baseline_sessions",
    "historical_insertions",
    "forward_sessions",
    "first_forward_session",
    "last_forward_session",
    "eligible_for_common_window",
    "source_state",
)


def _source_map(freeze: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    return {
        str(record["ticker"]): record
        for record in freeze["sources"]
    }


def _session_map(record: Mapping[str, object]) -> dict[str, str]:
    sessions = record.get("session_fingerprints")
    if not isinstance(sessions, Mapping):
        raise ValueError("Impronte per seduta mancanti.")
    return {str(date): str(value) for date, value in sessions.items()}


def _sha256_object(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def compare_research_freezes(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    config: ForwardReadinessConfig | None = None,
) -> ForwardReadinessAudit:
    """Confronta cronologia e integrita' senza leggere outcome futuri."""

    config = config or ForwardReadinessConfig()
    verify_research_freeze(baseline)
    verify_research_freeze(candidate)

    fatal: list[str] = []
    pending: list[str] = []
    specification_unchanged = (
        baseline["specification_sha256"] == candidate["specification_sha256"]
    )
    if not specification_unchanged:
        fatal.append("La specifica di ricerca e' cambiata dopo il freeze.")
    baseline_implementation = baseline.get("implementation", {})
    candidate_implementation = candidate.get("implementation", {})
    implementation_unchanged = (
        baseline_implementation.get("sha256")
        == candidate_implementation.get("sha256")
    )
    if not implementation_unchanged:
        fatal.append("L'implementazione descrittiva e' cambiata dopo il freeze.")

    baseline_sources = _source_map(baseline)
    candidate_sources = _source_map(candidate)
    baseline_tickers = set(baseline_sources)
    candidate_tickers = set(candidate_sources)
    if baseline_tickers != candidate_tickers:
        missing = sorted(baseline_tickers - candidate_tickers)
        added = sorted(candidate_tickers - baseline_tickers)
        fatal.append(
            "Universo sorgenti cambiato; mancanti="
            f"{','.join(missing) or '-'}; aggiunti={','.join(added) or '-'} ."
        )

    rows: list[dict[str, object]] = []
    forward_sets: list[set[str]] = []
    eligible_sources = 0
    for ticker in sorted(baseline_tickers & candidate_tickers):
        baseline_record = baseline_sources[ticker]
        candidate_record = candidate_sources[ticker]
        baseline_sessions = _session_map(baseline_record)
        candidate_sessions = _session_map(candidate_record)
        baseline_dates = set(baseline_sessions)
        candidate_dates = set(candidate_sessions)
        overlap = baseline_dates & candidate_dates
        changed = {
            date
            for date in overlap
            if baseline_sessions[date] != candidate_sessions[date]
        }
        unchanged = overlap - changed
        missing_baseline = baseline_dates - candidate_dates if overlap else set()
        cutoff = max(baseline_dates)
        new_dates = candidate_dates - baseline_dates
        historical_insertions = {date for date in new_dates if date <= cutoff}
        forward_dates = {date for date in new_dates if date > cutoff}
        comparison_mode = "CUMULATIVE" if overlap else "FORWARD_ONLY"

        source_failures: list[str] = []
        if changed:
            source_failures.append("HISTORICAL_DATA_CHANGED")
            fatal.append(
                f"{ticker}: {len(changed)} sedute storiche sono cambiate."
            )
        if missing_baseline:
            source_failures.append("BASELINE_SESSIONS_MISSING")
            fatal.append(
                f"{ticker}: {len(missing_baseline)} sedute baseline mancanti "
                "nel campione cumulativo."
            )
        if historical_insertions:
            source_failures.append("PRE_CUTOFF_INSERTIONS")
            fatal.append(
                f"{ticker}: {len(historical_insertions)} sedute sono state "
                "inserite prima del cutoff congelato."
            )

        baseline_status = str(baseline_record.get("source_status", ""))
        candidate_status = str(candidate_record.get("source_status", ""))
        eligible = (
            not source_failures
            and baseline_status in {"READY_DESCRIPTIVE", "PRICE_ONLY"}
            and candidate_status in {"READY_DESCRIPTIVE", "PRICE_ONLY"}
        )
        if eligible:
            eligible_sources += 1
            forward_sets.append(forward_dates)

        rows.append(
            {
                "ticker": ticker,
                "source_status_baseline": baseline_status,
                "source_status_candidate": candidate_status,
                "comparison_mode": comparison_mode,
                "baseline_sessions": len(baseline_dates),
                "candidate_sessions": len(candidate_dates),
                "overlap_sessions": len(overlap),
                "unchanged_overlap_sessions": len(unchanged),
                "changed_overlap_sessions": len(changed),
                "missing_baseline_sessions": len(missing_baseline),
                "historical_insertions": len(historical_insertions),
                "forward_sessions": len(forward_dates),
                "first_forward_session": min(forward_dates) if forward_dates else "",
                "last_forward_session": max(forward_dates) if forward_dates else "",
                "eligible_for_common_window": eligible,
                "source_state": "+".join(source_failures) if source_failures else "PASS",
            }
        )

    details = pd.DataFrame(rows, columns=DETAIL_COLUMNS)
    common_forward_dates = (
        set.intersection(*forward_sets) if forward_sets else set()
    )
    common_forward_sessions = len(common_forward_dates)
    complete_checkpoints = math.floor(
        common_forward_sessions / config.sessions_per_checkpoint
    )

    if eligible_sources < config.minimum_eligible_sources:
        pending.append(
            f"Sorgenti idonee {eligible_sources}/{config.minimum_eligible_sources}."
        )
    if common_forward_sessions < config.minimum_common_forward_sessions:
        pending.append(
            "Sedute forward comuni "
            f"{common_forward_sessions}/{config.minimum_common_forward_sessions}."
        )
    if complete_checkpoints < config.minimum_checkpoints:
        pending.append(
            "Checkpoint temporali completi "
            f"{complete_checkpoints}/{config.minimum_checkpoints}."
        )

    baseline_report_state = baseline.get("report_audit", {}).get("state")
    candidate_report_state = candidate.get("report_audit", {}).get("state")
    if baseline_report_state != "PASS" or candidate_report_state != "PASS":
        fatal.append("Almeno un set di report non supera l'audit strutturale.")

    if fatal:
        state = ForwardReadinessState.FAIL
        reasons = tuple(dict.fromkeys(fatal + pending))
    elif pending:
        state = ForwardReadinessState.INCONCLUSIVE
        reasons = tuple(dict.fromkeys(pending))
    else:
        state = ForwardReadinessState.PASS
        reasons = (
            "Nuova finestra intatta e sufficiente per una revisione "
            "scientifica cieca; nessuna conclusione sulla strategia.",
        )

    evidence_payload = {
        "baseline_freeze_id": baseline["freeze_id"],
        "candidate_freeze_id": candidate["freeze_id"],
        "config": asdict(config),
        "details": details.to_dict(orient="records"),
    }
    return ForwardReadinessAudit(
        state=state,
        baseline_freeze_id=str(baseline["freeze_id"]),
        candidate_freeze_id=str(candidate["freeze_id"]),
        evidence_id=_sha256_object(evidence_payload),
        specification_unchanged=specification_unchanged,
        implementation_unchanged=implementation_unchanged,
        eligible_sources=eligible_sources,
        common_forward_sessions=common_forward_sessions,
        complete_checkpoints=complete_checkpoints,
        details=details,
        reasons=reasons,
    )


def compare_research_freeze_files(
    baseline_path: str | Path,
    candidate_path: str | Path,
    config: ForwardReadinessConfig | None = None,
) -> ForwardReadinessAudit:
    return compare_research_freezes(
        load_research_freeze(baseline_path),
        load_research_freeze(candidate_path),
        config,
    )


def write_forward_readiness_details(
    audit: ForwardReadinessAudit,
    output_path: str | Path,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    audit.details.to_csv(destination, index=False)
    return destination


def write_forward_readiness_summary(
    audit: ForwardReadinessAudit,
    output_path: str | Path,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": FORWARD_READINESS_SCHEMA_VERSION,
        "artifact_type": "INTRADAY_FORWARD_RESEARCH_READINESS",
        "research_only": True,
        "state": audit.state.value,
        "baseline_freeze_id": audit.baseline_freeze_id,
        "candidate_freeze_id": audit.candidate_freeze_id,
        "evidence_id": audit.evidence_id,
        "specification_unchanged": audit.specification_unchanged,
        "implementation_unchanged": audit.implementation_unchanged,
        "eligible_sources": audit.eligible_sources,
        "common_forward_sessions": audit.common_forward_sessions,
        "complete_checkpoints": audit.complete_checkpoints,
        "reasons": list(audit.reasons),
        "interpretation": (
            "PASS significa soltanto readiness per revisione scientifica "
            "cieca; non prova accuratezza o redditivita'."
        ),
    }
    destination.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return destination
