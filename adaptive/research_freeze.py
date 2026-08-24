"""Congelamento riproducibile della ricerca intraday descrittiva.

Il modulo non calcola direzioni, segnali, outcome futuri o performance.  Crea
soltanto una prova verificabile di quali dati, parametri, implementazioni e
report descrittivi appartenevano a una specifica versione della ricerca.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from adaptive.intraday_report_audit import (
    REPORT_MANIFEST_FILENAME,
    STANDARD_INTRADAY_REPORT_FILENAMES,
    audit_intraday_report_directory,
)
from adaptive.local_intraday_data import LocalIntradaySourceAudit


RESEARCH_FREEZE_SCHEMA_VERSION = 1
RESEARCH_FREEZE_FILENAME = "intraday_research_freeze.json"

RESEARCH_ONLY_CONTRACT = (
    "NO_DIRECTION",
    "NO_SIGNAL",
    "NO_ORDER",
    "NO_POSITION_SIZE",
    "NO_STOP",
    "NO_LEVERAGE",
    "NO_FUTURE_OUTCOME",
    "NO_PNL",
)

IMPLEMENTATION_FILES = (
    "adaptive/research_freeze.py",
    "adaptive/forward_research_readiness.py",
    "adaptive/intraday_report_audit.py",
    "adaptive/run_local_intraday_research.py",
    "adaptive/intraday_features.py",
    "adaptive/intraday_stability.py",
    "adaptive/intraday_feature_readiness.py",
    "adaptive/intraday_regime_atlas.py",
    "adaptive/intraday_context_stability.py",
    "adaptive/intraday_cross_asset_dependence.py",
    "adaptive/local_intraday_data.py",
    "adaptive/lorentzian_research.py",
    "adaptive/lorentzian_neighborhood_stability.py",
    "adaptive/lorentzian_soft_surface.py",
    "adaptive/lorentzian_soft_block_stability.py",
)


@dataclass(frozen=True)
class IntradayResearchSpecification:
    """Parametri che devono restare identici durante la finestra forward."""

    interval: str = "15m"
    source_interval: str = "1m"
    market_timezone: str = "America/New_York"
    regular_session: str = "09:30-16:00"
    sessions_per_block: int = 30
    minimum_complete_blocks: int = 4
    normalization_window: int = 520
    normalization_min_periods: int = 260
    neighbors: int = 8
    minimum_candidates: int = 16
    embargo_bars: int = 4
    sample_stride: int = 4
    history_limit: int = 4_000
    context_minimum_observations: int = 60
    context_minimum_sessions: int = 10
    context_minimum_complete_blocks: int = 3
    feature_contract: str = "INTRADAY_REFERENCE_FEATURES_V1"
    lorentzian_distance_contract: str = "SUM_LOG1P_ABS_DELTA"
    frozen_soft_profile: str = "HL_2080_CTX_0.5"

    def __post_init__(self) -> None:
        if self.interval != "15m" or self.source_interval != "1m":
            raise ValueError("La specifica V1 richiede 1m -> 15m.")
        positive = (
            "sessions_per_block",
            "minimum_complete_blocks",
            "normalization_window",
            "normalization_min_periods",
            "neighbors",
            "minimum_candidates",
            "embargo_bars",
            "sample_stride",
            "history_limit",
            "context_minimum_observations",
            "context_minimum_sessions",
            "context_minimum_complete_blocks",
        )
        for name in positive:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} deve essere positivo.")
        if self.normalization_min_periods > self.normalization_window:
            raise ValueError(
                "normalization_min_periods non puo' superare la finestra."
            )
        if self.history_limit < self.minimum_candidates + self.embargo_bars:
            raise ValueError("history_limit insufficiente per candidati ed embargo.")


def specification_from_namespace(arguments) -> IntradayResearchSpecification:
    """Estrae soltanto i parametri scientifici dalla namespace CLI."""

    names = (
        "interval",
        "sessions_per_block",
        "minimum_complete_blocks",
        "normalization_window",
        "normalization_min_periods",
        "neighbors",
        "minimum_candidates",
        "embargo_bars",
        "sample_stride",
        "history_limit",
        "context_minimum_observations",
        "context_minimum_sessions",
        "context_minimum_complete_blocks",
    )
    return IntradayResearchSpecification(
        **{name: getattr(arguments, name) for name in names}
    )


def canonical_json_bytes(value: object) -> bytes:
    """Serializzazione stabile usata da tutte le impronte del protocollo."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_object(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _git_commit(repository_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "UNKNOWN"
    value = result.stdout.strip()
    return value if len(value) == 40 else "UNKNOWN"


def implementation_fingerprint(
    repository_root: str | Path,
    files: Iterable[str] = IMPLEMENTATION_FILES,
) -> tuple[str, dict[str, str]]:
    """Impronta soltanto il codice che costruisce l'evidenza descrittiva."""

    root = Path(repository_root)
    hashes: dict[str, str] = {}
    for relative in sorted(set(files)):
        path = root / relative
        if not path.is_file():
            raise ValueError(f"File di implementazione mancante: {relative}")
        hashes[relative] = sha256_file(path)
    return _sha256_object(hashes), hashes


def _float_token(value: object) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Le barre da congelare contengono valori non finiti.")
    return number.hex()


def _session_fingerprints(frame: pd.DataFrame) -> dict[str, str]:
    required = ("date", "open", "high", "low", "close")
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"Colonne mancanti per il freeze: {missing}.")
    columns = [*required[1:]]
    if "volume" in frame.columns:
        columns.append("volume")
    ordered = frame.loc[:, ["date", *columns]].copy()
    ordered["date"] = pd.to_datetime(ordered["date"], errors="raise")
    ordered = ordered.sort_values("date", kind="mergesort")
    if ordered["date"].duplicated().any():
        raise ValueError("Timestamp duplicati nel freeze della sorgente.")

    fingerprints: dict[str, str] = {}
    session_keys = ordered["date"].dt.date.astype(str)
    for session, block in ordered.groupby(session_keys, sort=True):
        digest = hashlib.sha256()
        digest.update(("date|" + "|".join(columns) + "\n").encode("ascii"))
        for row in block.itertuples(index=False, name=None):
            timestamp = pd.Timestamp(row[0]).isoformat()
            tokens = [timestamp, *(_float_token(value) for value in row[1:])]
            digest.update(("|".join(tokens) + "\n").encode("ascii"))
        fingerprints[str(session)] = digest.hexdigest()
    if not fingerprints:
        raise ValueError("Nessuna seduta disponibile per il freeze.")
    return fingerprints


def _source_record(
    ticker: str,
    market: pd.DataFrame,
    audit: LocalIntradaySourceAudit,
) -> dict[str, object]:
    source = Path(audit.source_path)
    if not source.is_file():
        raise ValueError(f"Archivio sorgente non trovato per {ticker}: {source}")
    sessions = _session_fingerprints(market)
    mode = "OHLCV" if audit.has_volume else "PRICE_ONLY"
    record = {
        "ticker": ticker,
        "archive_name": source.name,
        "archive_sha256": sha256_file(source),
        "data_mode": mode,
        "source_status": audit.status.value,
        "first_timestamp": pd.Timestamp(audit.first_timestamp).isoformat(),
        "last_timestamp": pd.Timestamp(audit.last_timestamp).isoformat(),
        "observed_sessions": int(audit.observed_sessions),
        "output_bars": int(audit.output_bars),
        "session_fingerprints": sessions,
    }
    record["bars_fingerprint"] = _sha256_object(sessions)
    return record


def _reported_tickers(report_directory: Path) -> tuple[str, ...]:
    scalar_columns = ("ticker", "ticker_a", "ticker_b")
    list_columns = (
        "asset_list",
        "cluster_members",
        "omitted_assets",
        "retained_assets",
    )
    tickers: set[str] = set()
    for filename in STANDARD_INTRADAY_REPORT_FILENAMES:
        frame = pd.read_csv(report_directory / filename)
        for column in scalar_columns:
            if column not in frame:
                continue
            tickers.update(
                str(value).strip().upper()
                for value in frame[column].dropna()
                if str(value).strip()
            )
        for column in list_columns:
            if column not in frame:
                continue
            for value in frame[column].dropna():
                tickers.update(
                    token.strip().upper()
                    for token in re.split(r"[;,]", str(value))
                    if token.strip()
                )
    return tuple(sorted(tickers))


def _report_records(report_directory: Path) -> tuple[dict, list[dict]]:
    audit = audit_intraday_report_directory(
        report_directory,
        expected_reports=STANDARD_INTRADAY_REPORT_FILENAMES,
    )
    if audit.overall_state != "PASS":
        failed = audit.details.loc[audit.details["audit_state"].ne("PASS")]
        problems = ", ".join(
            f"{row.report_name}={row.audit_state}"
            for row in failed.itertuples(index=False)
        )
        raise ValueError(f"Report descrittivi non congelabili: {problems}")
    records = [
        {
            "report_name": str(row.report_name),
            "sha256": str(row.sha256),
            "rows": int(row.rows),
            "columns": int(row.columns),
        }
        for row in audit.details.itertuples(index=False)
    ]
    summary = {
        "state": audit.overall_state,
        "expected_reports": int(audit.expected_reports),
        "observed_expected_reports": int(audit.observed_expected_reports),
        "unexpected_reports": int(audit.unexpected_reports),
        "total_rows": int(audit.total_rows),
        "reports_fingerprint": _sha256_object(records),
        "reported_tickers": list(_reported_tickers(report_directory)),
    }
    manifest = report_directory / REPORT_MANIFEST_FILENAME
    summary["manifest_sha256"] = sha256_file(manifest) if manifest.is_file() else ""
    return summary, records


def build_research_freeze(
    *,
    markets: Mapping[str, pd.DataFrame],
    source_audits: Mapping[str, LocalIntradaySourceAudit],
    report_directory: str | Path,
    specification: IntradayResearchSpecification,
    source_errors: Mapping[str, str] | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, object]:
    """Costruisce un freeze deterministico e auto-verificabile."""

    if source_errors:
        details = ", ".join(
            f"{ticker}={error}" for ticker, error in sorted(source_errors.items())
        )
        raise ValueError(f"Sorgenti con errori non congelabili: {details}")
    if not markets:
        raise ValueError("Nessun mercato da congelare.")
    if set(markets) != set(source_audits):
        raise ValueError("Mercati e audit sorgente non coincidono.")

    root = (
        Path(repository_root)
        if repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    implementation_sha, implementation_files = implementation_fingerprint(root)
    specification_record = asdict(specification)
    specification_sha = _sha256_object(specification_record)
    report_summary, report_records = _report_records(Path(report_directory))
    sources = [
        _source_record(ticker, markets[ticker], source_audits[ticker])
        for ticker in sorted(markets)
    ]
    source_tickers = {str(record["ticker"]) for record in sources}
    report_tickers = set(report_summary["reported_tickers"])
    if report_tickers != source_tickers:
        missing = sorted(report_tickers - source_tickers)
        extra = sorted(source_tickers - report_tickers)
        raise ValueError(
            "Universo report e sorgenti non coincide; archivi mancanti="
            f"{','.join(missing) or '-'}; archivi extra="
            f"{','.join(extra) or '-'} ."
        )
    payload: dict[str, object] = {
        "schema_version": RESEARCH_FREEZE_SCHEMA_VERSION,
        "artifact_type": "INTRADAY_DESCRIPTIVE_RESEARCH_FREEZE",
        "research_only": True,
        "contract": list(RESEARCH_ONLY_CONTRACT),
        "specification": specification_record,
        "specification_sha256": specification_sha,
        "implementation": {
            "git_commit": _git_commit(root),
            "sha256": implementation_sha,
            "files": implementation_files,
        },
        "sources": sources,
        "report_audit": report_summary,
        "reports": report_records,
    }
    payload["freeze_id"] = _sha256_object(payload)
    return payload


def verify_research_freeze(payload: Mapping[str, object]) -> None:
    """Rifiuta freeze alterati, incompleti o non research-only."""

    if int(payload.get("schema_version", -1)) != RESEARCH_FREEZE_SCHEMA_VERSION:
        raise ValueError("Versione schema freeze non supportata.")
    if payload.get("artifact_type") != "INTRADAY_DESCRIPTIVE_RESEARCH_FREEZE":
        raise ValueError("Tipo artefatto freeze non valido.")
    if payload.get("research_only") is not True:
        raise ValueError("Il freeze deve essere esplicitamente research-only.")
    if tuple(payload.get("contract", ())) != RESEARCH_ONLY_CONTRACT:
        raise ValueError("Contratto research-only alterato.")
    stored_id = str(payload.get("freeze_id", ""))
    unsigned = dict(payload)
    unsigned.pop("freeze_id", None)
    if stored_id != _sha256_object(unsigned):
        raise ValueError("Freeze alterato: freeze_id non corrisponde.")
    specification = payload.get("specification")
    if not isinstance(specification, Mapping):
        raise ValueError("Specificazione freeze mancante.")
    if payload.get("specification_sha256") != _sha256_object(specification):
        raise ValueError("Impronta della specificazione non valida.")
    implementation = payload.get("implementation")
    if not isinstance(implementation, Mapping):
        raise ValueError("Impronta implementazione mancante.")
    implementation_files = implementation.get("files")
    if not isinstance(implementation_files, Mapping) or not implementation_files:
        raise ValueError("Elenco file implementazione mancante.")
    if implementation.get("sha256") != _sha256_object(implementation_files):
        raise ValueError("Impronta implementazione non valida.")
    sources = payload.get("sources")
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        raise ValueError("Elenco sorgenti freeze mancante.")
    if not sources:
        raise ValueError("Il freeze non contiene sorgenti.")
    tickers = [str(record.get("ticker", "")) for record in sources]
    if any(not ticker for ticker in tickers) or len(tickers) != len(set(tickers)):
        raise ValueError("Ticker freeze mancanti o duplicati.")
    for record in sources:
        sessions = record.get("session_fingerprints")
        if not isinstance(sessions, Mapping) or not sessions:
            raise ValueError("Impronte per seduta mancanti nel freeze.")
        if record.get("bars_fingerprint") != _sha256_object(sessions):
            raise ValueError("Impronta barre sorgente non valida.")
    report_audit = payload.get("report_audit")
    if not isinstance(report_audit, Mapping) or report_audit.get("state") != "PASS":
        raise ValueError("Audit report del freeze non valido.")
    reports = payload.get("reports")
    if not isinstance(reports, Sequence) or isinstance(reports, (str, bytes)):
        raise ValueError("Elenco report del freeze mancante.")
    if report_audit.get("reports_fingerprint") != _sha256_object(reports):
        raise ValueError("Impronta del set di report non valida.")
    reported_tickers = set(report_audit.get("reported_tickers", ()))
    if reported_tickers != set(tickers):
        raise ValueError("Universo report e sorgenti del freeze non coincide.")


def load_research_freeze(path: str | Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Freeze JSON non leggibile: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Il freeze JSON deve contenere un oggetto.")
    verify_research_freeze(payload)
    return payload


def write_research_freeze(
    payload: Mapping[str, object],
    output_path: str | Path,
) -> Path:
    """Scrive una volta; un contenuto diverso non puo' sovrascrivere il freeze."""

    verify_research_freeze(payload)
    destination = Path(output_path)
    rendered = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    if destination.exists():
        existing = load_research_freeze(destination)
        if existing.get("freeze_id") != payload.get("freeze_id"):
            raise FileExistsError(
                "Freeze gia' presente con contenuto diverso; usa una nuova "
                "directory di checkpoint."
            )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered, encoding="utf-8", newline="\n")
    return destination
