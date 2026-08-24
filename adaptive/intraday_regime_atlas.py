"""Atlante descrittivo dei contesti intraday osservati.

Il modulo combina esclusivamente stati contemporanei gia' calcolati:
fase della seduta, segmento CHOP e stato Squeeze. Misura frequenze, copertura
storica e transizioni tra barre adiacenti della stessa seduta. Non usa
rendimenti futuri, non stima una direzione e non produce segnali, ordini,
size, stop, leva o P&L.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from enum import Enum
from pathlib import Path
from typing import Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


class DescriptiveContextCoverage(str, Enum):
    """Copertura del campione, mai idoneita' a operare."""

    ADEQUATE_DESCRIPTIVE_COVERAGE = "ADEQUATE_DESCRIPTIVE_COVERAGE"
    SPARSE_CONTEXT = "SPARSE_CONTEXT"
    INSUFFICIENT = "INSUFFICIENT"


@dataclass(frozen=True)
class IntradayRegimeAtlasConfig:
    """Contratto temporale e soglie dichiarate per l'atlante."""

    market_timezone: str = "America/New_York"
    regular_session_start: time = time(9, 30)
    open_phase_end: time = time(10, 30)
    close_phase_start: time = time(15, 0)
    regular_session_end: time = time(16, 0)
    interval_minutes: int = 15
    sessions_per_block: int = 30
    minimum_observations: int = 60
    minimum_sessions: int = 10
    minimum_complete_blocks: int = 3

    def __post_init__(self) -> None:
        ZoneInfo(self.market_timezone)
        minutes = tuple(
            self._minutes(value)
            for value in (
                self.regular_session_start,
                self.open_phase_end,
                self.close_phase_start,
                self.regular_session_end,
            )
        )
        if tuple(sorted(minutes)) != minutes or len(set(minutes)) != len(minutes):
            raise ValueError("Le fasi della sessione non sono ordinate.")
        for name in (
            "interval_minutes",
            "sessions_per_block",
            "minimum_observations",
            "minimum_sessions",
            "minimum_complete_blocks",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} deve essere positivo.")
        session_minutes = minutes[-1] - minutes[0]
        if session_minutes % int(self.interval_minutes):
            raise ValueError("L'intervallo non divide la sessione regolare.")

    @staticmethod
    def _minutes(value: time) -> int:
        return value.hour * 60 + value.minute


ASSIGNMENT_COLUMNS = (
    "ticker",
    "source_status",
    "data_mode",
    "cmf_availability",
    "timestamp",
    "session_date",
    "block",
    "block_complete",
    "session_phase",
    "chop_segment",
    "squeeze_state",
    "context_key",
)

ATLAS_COLUMNS = (
    "ticker",
    "source_status",
    "data_mode",
    "cmf_availability",
    "session_phase",
    "chop_segment",
    "squeeze_state",
    "observations",
    "sessions",
    "blocks_present",
    "complete_blocks_present",
    "observation_share_phase",
    "session_share_phase",
    "first_timestamp",
    "last_timestamp",
    "coverage",
)

TRANSITION_COLUMNS = (
    "ticker",
    "source_status",
    "data_mode",
    "cmf_availability",
    "from_phase",
    "from_chop_segment",
    "from_squeeze_state",
    "to_phase",
    "to_chop_segment",
    "to_squeeze_state",
    "observations",
    "sessions",
    "blocks_present",
    "complete_blocks_present",
    "transition_share_from_context",
    "first_timestamp",
    "last_timestamp",
    "coverage",
)

CROSS_ASSET_COLUMNS = (
    "session_phase",
    "chop_segment",
    "squeeze_state",
    "assets_observed",
    "assets_adequate",
    "ohlcv_assets",
    "price_only_assets",
    "total_observations",
    "median_observation_share_phase",
    "minimum_observation_share_phase",
    "maximum_observation_share_phase",
)


@dataclass(frozen=True)
class IntradayRegimeAtlasReport:
    """Tabelle descrittive separate dagli oggetti operativi del progetto."""

    assignments: pd.DataFrame
    atlas: pd.DataFrame
    transitions: pd.DataFrame
    cross_asset: pd.DataFrame
    caveats: tuple[str, ...]
    research_only: bool = True


def _empty(columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _audit_metadata(audit: object) -> tuple[str, str, str]:
    if audit is None:
        raise ValueError("Audit sorgente mancante.")
    status_value = getattr(audit, "status", None)
    source_status = getattr(status_value, "value", status_value)
    if source_status is None:
        raise ValueError("Stato sorgente mancante nell'audit.")
    has_volume = bool(getattr(audit, "has_volume", False))
    return (
        str(source_status),
        "OHLCV" if has_volume else "PRICE",
        "AVAILABLE" if has_volume else "NOT_AVAILABLE",
    )


def _localized_index(index: pd.DatetimeIndex, timezone: str) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("Le feature devono avere un DatetimeIndex.")
    parsed = pd.DatetimeIndex(pd.to_datetime(index, errors="coerce"))
    if parsed.isna().any():
        raise ValueError("Sono presenti timestamp non validi.")
    if parsed.tz is None:
        parsed = parsed.tz_localize(
            timezone,
            ambiguous="NaT",
            nonexistent="NaT",
        )
        if parsed.isna().any():
            raise ValueError("Timestamp locale ambiguo o inesistente.")
    return parsed.tz_convert(timezone)


def _coverage(
    observations: int,
    sessions: int,
    complete_blocks: int,
    config: IntradayRegimeAtlasConfig,
) -> str:
    if observations <= 0 or sessions <= 0:
        return DescriptiveContextCoverage.INSUFFICIENT.value
    if (
        observations >= config.minimum_observations
        and sessions >= config.minimum_sessions
        and complete_blocks >= config.minimum_complete_blocks
    ):
        return DescriptiveContextCoverage.ADEQUATE_DESCRIPTIVE_COVERAGE.value
    return DescriptiveContextCoverage.SPARSE_CONTEXT.value


def _prepare_assignments(
    ticker: str,
    values: pd.DataFrame,
    audit: object,
    config: IntradayRegimeAtlasConfig,
) -> pd.DataFrame:
    normalized_ticker = str(ticker).upper().strip()
    if not normalized_ticker:
        raise ValueError("ticker non puo' essere vuoto.")
    if not isinstance(values, pd.DataFrame):
        raise TypeError("values deve essere un pandas DataFrame.")
    if values.empty:
        return _empty(ASSIGNMENT_COLUMNS)
    required = {"chop_segment", "squeeze_state"}
    missing = sorted(required.difference(values.columns))
    if missing:
        raise ValueError(f"Stati intraday mancanti: {missing}.")

    source_status, data_mode, cmf_availability = _audit_metadata(audit)
    frame = values.loc[:, ["chop_segment", "squeeze_state"]].copy()
    frame.index = _localized_index(frame.index, config.market_timezone)
    if frame.index.duplicated().any():
        raise ValueError("Sono presenti timestamp feature duplicati.")
    frame = frame.sort_index()
    local = frame.index
    minutes = local.hour * 60 + local.minute
    start = config._minutes(config.regular_session_start)
    end = config._minutes(config.regular_session_end)
    regular = (local.dayofweek < 5) & (minutes >= start) & (minutes < end)
    frame = frame.loc[regular].copy()
    if frame.empty:
        return _empty(ASSIGNMENT_COLUMNS)

    minutes = frame.index.hour * 60 + frame.index.minute
    offset = minutes - start
    if ((offset < 0) | (offset % config.interval_minutes != 0)).any():
        raise ValueError("Timestamp fuori dalla griglia RTH dichiarata.")

    frame["session_date"] = frame.index.date
    sessions = sorted(frame["session_date"].unique())
    session_ordinals = {
        session: position for position, session in enumerate(sessions)
    }
    frame["block"] = (
        frame["session_date"].map(session_ordinals).astype(int)
        // config.sessions_per_block
        + 1
    )
    block_session_counts = frame.groupby("block")["session_date"].nunique()
    complete_blocks = frozenset(
        int(block)
        for block, count in block_session_counts.items()
        if int(count) == config.sessions_per_block
    )
    frame["block_complete"] = frame["block"].isin(complete_blocks)

    minutes = frame.index.hour * 60 + frame.index.minute
    open_end = config._minutes(config.open_phase_end)
    close_start = config._minutes(config.close_phase_start)
    frame["session_phase"] = np.select(
        [minutes < open_end, minutes >= close_start],
        ["OPEN", "CLOSE"],
        default="MID_SESSION",
    )
    frame["chop_segment"] = frame["chop_segment"].astype(str).str.upper()
    frame["squeeze_state"] = frame["squeeze_state"].astype(str).str.upper()
    valid_chop = {"TRENDING", "NEUTRAL", "CHOPPY"}
    valid_squeeze = {"SQUEEZE_ON", "SQUEEZE_OFF", "NEUTRAL"}
    valid = frame["chop_segment"].isin(valid_chop) & frame[
        "squeeze_state"
    ].isin(valid_squeeze)
    frame = frame.loc[valid].copy()
    if frame.empty:
        return _empty(ASSIGNMENT_COLUMNS)

    frame.insert(0, "timestamp", frame.index)
    frame.insert(0, "cmf_availability", cmf_availability)
    frame.insert(0, "data_mode", data_mode)
    frame.insert(0, "source_status", source_status)
    frame.insert(0, "ticker", normalized_ticker)
    frame["context_key"] = (
        frame["session_phase"]
        + "|"
        + frame["chop_segment"]
        + "|"
        + frame["squeeze_state"]
    )
    return frame.loc[:, ASSIGNMENT_COLUMNS].reset_index(drop=True)


def _build_atlas(
    assignments: pd.DataFrame,
    config: IntradayRegimeAtlasConfig,
) -> pd.DataFrame:
    if assignments.empty:
        return _empty(ATLAS_COLUMNS)
    group_columns = [
        "ticker",
        "source_status",
        "data_mode",
        "cmf_availability",
        "session_phase",
        "chop_segment",
        "squeeze_state",
    ]
    rows: list[dict[str, object]] = []
    phase_observations = assignments.groupby(
        ["ticker", "session_phase"], sort=True
    ).size()
    phase_sessions = assignments.groupby(
        ["ticker", "session_phase"], sort=True
    )["session_date"].nunique()
    for keys, group in assignments.groupby(group_columns, sort=True):
        key_values = dict(zip(group_columns, keys))
        ticker = str(key_values["ticker"])
        phase = str(key_values["session_phase"])
        observations = int(len(group))
        sessions = int(group["session_date"].nunique())
        complete_blocks = int(
            group.loc[group["block_complete"], "block"].nunique()
        )
        rows.append(
            {
                **key_values,
                "observations": observations,
                "sessions": sessions,
                "blocks_present": int(group["block"].nunique()),
                "complete_blocks_present": complete_blocks,
                "observation_share_phase": float(
                    observations / int(phase_observations.loc[(ticker, phase)])
                ),
                "session_share_phase": float(
                    sessions / int(phase_sessions.loc[(ticker, phase)])
                ),
                "first_timestamp": group["timestamp"].min(),
                "last_timestamp": group["timestamp"].max(),
                "coverage": _coverage(
                    observations,
                    sessions,
                    complete_blocks,
                    config,
                ),
            }
        )
    return pd.DataFrame(rows, columns=ATLAS_COLUMNS).sort_values(
        ["ticker", "session_phase", "chop_segment", "squeeze_state"],
        kind="mergesort",
    ).reset_index(drop=True)


def _build_transitions(
    assignments: pd.DataFrame,
    config: IntradayRegimeAtlasConfig,
) -> pd.DataFrame:
    if assignments.empty:
        return _empty(TRANSITION_COLUMNS)
    ordered = assignments.sort_values(
        ["ticker", "timestamp"], kind="mergesort"
    ).copy()
    group_keys = [ordered["ticker"], ordered["session_date"]]
    for column in ("timestamp", "session_phase", "chop_segment", "squeeze_state"):
        ordered[f"previous_{column}"] = ordered.groupby(
            group_keys, sort=False
        )[column].shift(1)
    elapsed = ordered["timestamp"] - ordered["previous_timestamp"]
    consecutive = elapsed.eq(pd.Timedelta(minutes=config.interval_minutes))
    transitions = ordered.loc[consecutive].copy()
    if transitions.empty:
        return _empty(TRANSITION_COLUMNS)
    transitions = transitions.rename(
        columns={
            "previous_session_phase": "from_phase",
            "previous_chop_segment": "from_chop_segment",
            "previous_squeeze_state": "from_squeeze_state",
            "session_phase": "to_phase",
            "chop_segment": "to_chop_segment",
            "squeeze_state": "to_squeeze_state",
        }
    )
    group_columns = [
        "ticker",
        "source_status",
        "data_mode",
        "cmf_availability",
        "from_phase",
        "from_chop_segment",
        "from_squeeze_state",
        "to_phase",
        "to_chop_segment",
        "to_squeeze_state",
    ]
    denominators = transitions.groupby(
        [
            "ticker",
            "from_phase",
            "from_chop_segment",
            "from_squeeze_state",
        ],
        sort=True,
    ).size()
    rows: list[dict[str, object]] = []
    for keys, group in transitions.groupby(group_columns, sort=True):
        key_values = dict(zip(group_columns, keys))
        denominator_key = (
            str(key_values["ticker"]),
            str(key_values["from_phase"]),
            str(key_values["from_chop_segment"]),
            str(key_values["from_squeeze_state"]),
        )
        observations = int(len(group))
        sessions = int(group["session_date"].nunique())
        complete_blocks = int(
            group.loc[group["block_complete"], "block"].nunique()
        )
        rows.append(
            {
                **key_values,
                "observations": observations,
                "sessions": sessions,
                "blocks_present": int(group["block"].nunique()),
                "complete_blocks_present": complete_blocks,
                "transition_share_from_context": float(
                    observations / int(denominators.loc[denominator_key])
                ),
                "first_timestamp": group["timestamp"].min(),
                "last_timestamp": group["timestamp"].max(),
                "coverage": _coverage(
                    observations,
                    sessions,
                    complete_blocks,
                    config,
                ),
            }
        )
    return pd.DataFrame(rows, columns=TRANSITION_COLUMNS).sort_values(
        [
            "ticker",
            "from_phase",
            "from_chop_segment",
            "from_squeeze_state",
            "to_phase",
            "to_chop_segment",
            "to_squeeze_state",
        ],
        kind="mergesort",
    ).reset_index(drop=True)


def _build_cross_asset(atlas: pd.DataFrame) -> pd.DataFrame:
    if atlas.empty:
        return _empty(CROSS_ASSET_COLUMNS)
    rows: list[dict[str, object]] = []
    for keys, group in atlas.groupby(
        ["session_phase", "chop_segment", "squeeze_state"],
        sort=True,
    ):
        phase, chop, squeeze = keys
        shares = pd.to_numeric(
            group["observation_share_phase"], errors="coerce"
        ).dropna()
        rows.append(
            {
                "session_phase": phase,
                "chop_segment": chop,
                "squeeze_state": squeeze,
                "assets_observed": int(group["ticker"].nunique()),
                "assets_adequate": int(
                    group.loc[
                        group["coverage"].eq(
                            DescriptiveContextCoverage.ADEQUATE_DESCRIPTIVE_COVERAGE.value
                        ),
                        "ticker",
                    ].nunique()
                ),
                "ohlcv_assets": int(
                    group.loc[group["data_mode"].eq("OHLCV"), "ticker"].nunique()
                ),
                "price_only_assets": int(
                    group.loc[group["data_mode"].eq("PRICE"), "ticker"].nunique()
                ),
                "total_observations": int(group["observations"].sum()),
                "median_observation_share_phase": float(shares.median()),
                "minimum_observation_share_phase": float(shares.min()),
                "maximum_observation_share_phase": float(shares.max()),
            }
        )
    return pd.DataFrame(rows, columns=CROSS_ASSET_COLUMNS).sort_values(
        ["session_phase", "chop_segment", "squeeze_state"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_intraday_regime_atlas(
    feature_frames: Mapping[str, pd.DataFrame],
    source_audits: Mapping[str, object],
    config: IntradayRegimeAtlasConfig | None = None,
) -> IntradayRegimeAtlasReport:
    """Costruisce l'atlante multi-asset senza collegarlo ad alcuna strategia."""

    settings = config or IntradayRegimeAtlasConfig()
    assignment_frames: list[pd.DataFrame] = []
    for ticker, values in sorted(feature_frames.items()):
        normalized_ticker = str(ticker).upper().strip()
        if normalized_ticker not in source_audits:
            raise ValueError(f"Audit sorgente mancante per {normalized_ticker}.")
        prepared = _prepare_assignments(
            normalized_ticker,
            values,
            source_audits[normalized_ticker],
            settings,
        )
        if not prepared.empty:
            assignment_frames.append(prepared)
    assignments = (
        pd.concat(assignment_frames, ignore_index=True)
        if assignment_frames
        else _empty(ASSIGNMENT_COLUMNS)
    )
    if not assignments.empty:
        assignments = assignments.sort_values(
            ["ticker", "timestamp"], kind="mergesort"
        ).reset_index(drop=True)
    atlas = _build_atlas(assignments, settings)
    transitions = _build_transitions(assignments, settings)
    cross_asset = _build_cross_asset(atlas)
    return IntradayRegimeAtlasReport(
        assignments=assignments,
        atlas=atlas,
        transitions=transitions,
        cross_asset=cross_asset,
        caveats=(
            "Gli stati sono contemporanei e descrittivi; non sono etichette direzionali.",
            "Le transizioni richiedono barre adiacenti della stessa seduta e non attraversano la notte.",
            "Gli asset correlati non costituiscono prove statistiche indipendenti.",
            "CMF e' soltanto marcato disponibile o non disponibile; il suo segno non entra negli stati.",
        ),
    )


def summarize_regime_atlas(report: IntradayRegimeAtlasReport) -> pd.DataFrame:
    """Riepiloga la copertura per asset senza assegnare approvazioni."""

    if report.atlas.empty:
        return pd.DataFrame(
            columns=(
                "ticker",
                "source_status",
                "assignments",
                "context_cells",
                "adequate",
                "sparse",
                "insufficient",
                "transition_types",
                "transition_observations",
            )
        )
    rows: list[dict[str, object]] = []
    for ticker, group in report.atlas.groupby("ticker", sort=True):
        transitions = report.transitions.loc[
            report.transitions["ticker"].eq(ticker)
        ]
        coverage_counts = group["coverage"].value_counts()
        rows.append(
            {
                "ticker": ticker,
                "source_status": str(group["source_status"].iloc[0]),
                "assignments": int(
                    report.assignments["ticker"].eq(ticker).sum()
                ),
                "context_cells": int(len(group)),
                "adequate": int(
                    coverage_counts.get(
                        DescriptiveContextCoverage.ADEQUATE_DESCRIPTIVE_COVERAGE.value,
                        0,
                    )
                ),
                "sparse": int(
                    coverage_counts.get(
                        DescriptiveContextCoverage.SPARSE_CONTEXT.value,
                        0,
                    )
                ),
                "insufficient": int(
                    coverage_counts.get(
                        DescriptiveContextCoverage.INSUFFICIENT.value,
                        0,
                    )
                ),
                "transition_types": int(len(transitions)),
                "transition_observations": int(
                    transitions["observations"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_csv(frame: pd.DataFrame, path: str | Path, name: str) -> Path:
    destination = Path(path).expanduser()
    if destination.suffix.lower() != ".csv":
        raise ValueError(f"{name} deve terminare con .csv.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination


def write_regime_atlas(atlas: pd.DataFrame, path: str | Path) -> Path:
    """Esporta le frequenze dei contesti in formato tidy."""

    return _write_csv(atlas, path, "regime_atlas_output")


def write_regime_transitions(
    transitions: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Esporta le sole transizioni intraseduta fra barre consecutive."""

    return _write_csv(transitions, path, "regime_transition_output")
