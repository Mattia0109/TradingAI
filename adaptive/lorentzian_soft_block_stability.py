"""Stabilita' descrittiva per blocchi di un profilo Lorentziano soft.

Il modulo congela un profilo dichiarato e ne confronta soltanto la geometria
tra blocchi cronologici fissi. Non usa outcome futuri, non seleziona profili e
non produce direzioni o altri elementi operativi.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


BLOCK_COLUMNS = (
    "ticker",
    "profile",
    "block_id",
    "block_start_session",
    "block_end_session",
    "block_sessions",
    "block_state",
    "total_queries",
    "complete_queries",
    "coverage",
    "median_raw_distance",
    "median_adjusted_distance",
    "median_neighbor_age_bars",
    "p90_neighbor_age_bars",
    "median_joint_context_match_fraction",
    "median_context_mismatches",
    "median_neighbor_session_count",
    "median_neighbor_session_effective_count",
    "median_neighbor_max_session_share",
    "median_overlap_raw_reference",
)

SUMMARY_COLUMNS = (
    "ticker",
    "profile",
    "observed_sessions",
    "blocks_with_queries",
    "eligible_blocks",
    "minimum_required_blocks",
    "minimum_block_coverage",
    "median_block_coverage",
    "max_raw_distance_relative_change",
    "max_adjusted_distance_relative_change",
    "max_neighbor_age_relative_change",
    "max_context_match_absolute_change",
    "max_overlap_absolute_change",
    "minimum_neighbor_session_count",
    "minimum_effective_session_count",
    "maximum_neighbor_session_share",
    "stability_state",
)

CROSS_ASSET_COLUMNS = (
    "profile",
    "assets",
    "low_variation_assets",
    "moderate_variation_assets",
    "high_variation_assets",
    "insufficient_assets",
    "median_eligible_blocks",
    "median_minimum_block_coverage",
    "median_max_raw_distance_relative_change",
    "median_max_neighbor_age_relative_change",
    "median_max_context_match_absolute_change",
    "median_max_overlap_absolute_change",
    "median_minimum_effective_session_count",
    "median_maximum_neighbor_session_share",
)


@dataclass(frozen=True)
class LorentzianSoftBlockStabilityConfig:
    """Soglie pre-dichiarate per la sola variazione geometrica."""

    profile: str = "HL_2080_CTX_0.5"
    sessions_per_block: int = 30
    minimum_queries_per_block: int = 60
    minimum_complete_blocks: int = 3
    neighbors: int = 8
    moderate_raw_distance_change: float = 0.15
    high_raw_distance_change: float = 0.30
    moderate_age_change: float = 0.35
    high_age_change: float = 0.60
    moderate_context_change: float = 0.20
    high_context_change: float = 0.35
    moderate_overlap_change: float = 0.20
    high_overlap_change: float = 0.35
    moderate_minimum_coverage: float = 0.98
    high_minimum_coverage: float = 0.90
    moderate_minimum_effective_sessions: float = 6.0
    high_minimum_effective_sessions: float = 4.0

    def __post_init__(self) -> None:
        if not str(self.profile).strip():
            raise ValueError("profile non puo' essere vuoto.")
        if int(self.sessions_per_block) <= 0:
            raise ValueError("sessions_per_block deve essere positivo.")
        if int(self.minimum_queries_per_block) <= 0:
            raise ValueError("minimum_queries_per_block deve essere positivo.")
        if int(self.minimum_complete_blocks) < 2:
            raise ValueError("Servono almeno due blocchi completi.")
        if int(self.neighbors) <= 0:
            raise ValueError("neighbors deve essere positivo.")
        pairs = (
            (
                self.moderate_raw_distance_change,
                self.high_raw_distance_change,
                "raw_distance_change",
            ),
            (self.moderate_age_change, self.high_age_change, "age_change"),
            (
                self.moderate_context_change,
                self.high_context_change,
                "context_change",
            ),
            (
                self.moderate_overlap_change,
                self.high_overlap_change,
                "overlap_change",
            ),
        )
        for moderate, high, name in pairs:
            if not 0.0 <= float(moderate) < float(high):
                raise ValueError(f"Soglie {name} non valide.")
        if not (
            0.0
            < float(self.high_minimum_coverage)
            < float(self.moderate_minimum_coverage)
            <= 1.0
        ):
            raise ValueError("Soglie minimum_coverage non valide.")
        if not (
            0.0
            < float(self.high_minimum_effective_sessions)
            < float(self.moderate_minimum_effective_sessions)
            <= float(self.neighbors)
        ):
            raise ValueError("Soglie effective_sessions non valide.")


@dataclass(frozen=True)
class LorentzianSoftBlockStabilityReport:
    """Dettaglio per blocco e riepilogo della variazione geometrica."""

    blocks: pd.DataFrame
    summary: pd.DataFrame
    caveats: tuple[str, ...]
    research_only: bool = True


def _validate_inputs(
    ticker: str,
    details: pd.DataFrame,
    source_index: pd.DatetimeIndex,
    config: LorentzianSoftBlockStabilityConfig,
) -> tuple[str, pd.DataFrame, pd.DatetimeIndex]:
    symbol = str(ticker).strip().upper()
    if not symbol:
        raise ValueError("ticker non puo' essere vuoto.")
    if not isinstance(details, pd.DataFrame) or details.empty:
        raise ValueError("details deve essere un DataFrame non vuoto.")
    required = {
        "ticker",
        "timestamp",
        "profile",
        "neighbor_count",
        "raw_distance_median",
        "adjusted_distance_median",
        "neighbor_age_median_bars",
        "neighbor_age_p90_bars",
        "joint_context_match_fraction",
        "mean_context_mismatches",
        "neighbor_session_count",
        "neighbor_session_effective_count",
        "neighbor_max_session_share",
        "overlap_raw_reference",
    }
    missing = sorted(required.difference(details.columns))
    if missing:
        raise ValueError(f"Colonne dettaglio Lorentziano mancanti: {missing}.")
    if not isinstance(source_index, pd.DatetimeIndex) or source_index.empty:
        raise ValueError("source_index deve essere un DatetimeIndex non vuoto.")
    if source_index.tz is None:
        raise ValueError("source_index deve avere una timezone dichiarata.")
    if (
        source_index.hasnans
        or source_index.duplicated().any()
        or not source_index.is_monotonic_increasing
    ):
        raise ValueError("source_index contiene timestamp invalidi o disordinati.")

    selected = details.loc[
        details["ticker"].astype(str).str.upper().eq(symbol)
        & details["profile"].astype(str).eq(config.profile)
    ].copy()
    if selected.empty:
        raise ValueError(
            f"Profilo {config.profile} non disponibile per {symbol}."
        )
    timestamps = pd.to_datetime(selected["timestamp"], errors="raise")
    if not isinstance(timestamps.dtype, pd.DatetimeTZDtype):
        raise ValueError("I timestamp del dettaglio devono avere una timezone.")
    selected["timestamp"] = timestamps
    selected = selected.sort_values("timestamp").reset_index(drop=True)
    if selected["timestamp"].duplicated().any():
        raise ValueError("Timestamp query duplicati per ticker e profilo.")
    return symbol, selected, source_index


def _safe_median(frame: pd.DataFrame, column: str) -> float:
    if frame.empty:
        return math.nan
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.median()) if values.notna().any() else math.nan


def _safe_quantile(frame: pd.DataFrame, column: str, q: float) -> float:
    if frame.empty:
        return math.nan
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.quantile(q)) if values.notna().any() else math.nan


def _max_relative_change(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if numeric.size < 2:
        return math.nan
    previous = numeric[:-1]
    current = numeric[1:]
    denominator = np.maximum(np.abs(previous), 1e-12)
    return float(np.max(np.abs(current - previous) / denominator))


def _max_absolute_change(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if numeric.size < 2:
        return math.nan
    return float(np.max(np.abs(np.diff(numeric))))


def _finite_or_nan(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else math.nan


def _variation_state(
    summary: dict[str, object],
    config: LorentzianSoftBlockStabilityConfig,
) -> str:
    if int(summary["eligible_blocks"]) < int(config.minimum_complete_blocks):
        return "INSUFFICIENT_BLOCKS"

    min_coverage = float(summary["minimum_block_coverage"])
    raw_change = float(summary["max_raw_distance_relative_change"])
    age_change = float(summary["max_neighbor_age_relative_change"])
    context_change = float(summary["max_context_match_absolute_change"])
    overlap_change = float(summary["max_overlap_absolute_change"])
    effective_sessions = float(summary["minimum_effective_session_count"])

    high = (
        min_coverage < float(config.high_minimum_coverage)
        or raw_change > float(config.high_raw_distance_change)
        or age_change > float(config.high_age_change)
        or context_change > float(config.high_context_change)
        or overlap_change > float(config.high_overlap_change)
        or effective_sessions < float(config.high_minimum_effective_sessions)
    )
    if high:
        return "HIGH_BLOCK_VARIATION"

    moderate = (
        min_coverage < float(config.moderate_minimum_coverage)
        or raw_change > float(config.moderate_raw_distance_change)
        or age_change > float(config.moderate_age_change)
        or context_change > float(config.moderate_context_change)
        or overlap_change > float(config.moderate_overlap_change)
        or effective_sessions
        < float(config.moderate_minimum_effective_sessions)
    )
    return "MODERATE_BLOCK_VARIATION" if moderate else "LOW_BLOCK_VARIATION"


def analyze_lorentzian_soft_block_stability(
    ticker: str,
    details: pd.DataFrame,
    source_index: pd.DatetimeIndex,
    config: LorentzianSoftBlockStabilityConfig | None = None,
) -> LorentzianSoftBlockStabilityReport:
    """Confronta un profilo fisso tra blocchi senza usare outcome futuri."""

    settings = config or LorentzianSoftBlockStabilityConfig()
    symbol, selected, index = _validate_inputs(
        ticker, details, source_index, settings
    )
    local_index = index.tz_convert("America/New_York")
    session_dates = list(dict.fromkeys(timestamp.date() for timestamp in local_index))
    date_to_block = {
        session: ordinal // int(settings.sessions_per_block) + 1
        for ordinal, session in enumerate(session_dates)
    }
    selected["session_date"] = [
        timestamp.tz_convert("America/New_York").date()
        for timestamp in selected["timestamp"]
    ]
    unknown = sorted(set(selected["session_date"]).difference(date_to_block))
    if unknown:
        raise ValueError(f"Query fuori dall'indice sorgente: {unknown[:3]}.")
    selected["block_id"] = selected["session_date"].map(date_to_block).astype(int)

    block_rows: list[dict[str, object]] = []
    block_count = math.ceil(len(session_dates) / int(settings.sessions_per_block))
    for block_id in range(1, block_count + 1):
        first = (block_id - 1) * int(settings.sessions_per_block)
        last = min(first + int(settings.sessions_per_block), len(session_dates))
        block_sessions = session_dates[first:last]
        group = selected.loc[selected["block_id"].eq(block_id)]
        complete = group.loc[
            pd.to_numeric(group["neighbor_count"], errors="coerce").eq(
                int(settings.neighbors)
            )
        ]
        session_count = len(block_sessions)
        total_queries = len(group)
        complete_queries = len(complete)
        if total_queries == 0:
            block_state = "WARMUP_NO_QUERIES"
        elif session_count < int(settings.sessions_per_block):
            block_state = "PARTIAL_BLOCK"
        elif total_queries < int(settings.minimum_queries_per_block):
            block_state = "LOW_QUERY_COUNT"
        else:
            block_state = "ELIGIBLE"
        block_rows.append(
            {
                "ticker": symbol,
                "profile": settings.profile,
                "block_id": block_id,
                "block_start_session": block_sessions[0].isoformat(),
                "block_end_session": block_sessions[-1].isoformat(),
                "block_sessions": session_count,
                "block_state": block_state,
                "total_queries": total_queries,
                "complete_queries": complete_queries,
                "coverage": complete_queries / total_queries
                if total_queries
                else math.nan,
                "median_raw_distance": _safe_median(
                    complete, "raw_distance_median"
                ),
                "median_adjusted_distance": _safe_median(
                    complete, "adjusted_distance_median"
                ),
                "median_neighbor_age_bars": _safe_median(
                    complete, "neighbor_age_median_bars"
                ),
                "p90_neighbor_age_bars": _safe_quantile(
                    complete, "neighbor_age_p90_bars", 0.90
                ),
                "median_joint_context_match_fraction": _safe_median(
                    complete, "joint_context_match_fraction"
                ),
                "median_context_mismatches": _safe_median(
                    complete, "mean_context_mismatches"
                ),
                "median_neighbor_session_count": _safe_median(
                    complete, "neighbor_session_count"
                ),
                "median_neighbor_session_effective_count": _safe_median(
                    complete, "neighbor_session_effective_count"
                ),
                "median_neighbor_max_session_share": _safe_median(
                    complete, "neighbor_max_session_share"
                ),
                "median_overlap_raw_reference": _safe_median(
                    complete, "overlap_raw_reference"
                ),
            }
        )

    blocks = pd.DataFrame(block_rows, columns=BLOCK_COLUMNS)
    eligible = blocks.loc[blocks["block_state"].eq("ELIGIBLE")].copy()
    observed = blocks.loc[blocks["total_queries"].gt(0)]
    summary_row: dict[str, object] = {
        "ticker": symbol,
        "profile": settings.profile,
        "observed_sessions": len(session_dates),
        "blocks_with_queries": len(observed),
        "eligible_blocks": len(eligible),
        "minimum_required_blocks": int(settings.minimum_complete_blocks),
        "minimum_block_coverage": _finite_or_nan(eligible["coverage"].min()),
        "median_block_coverage": _finite_or_nan(eligible["coverage"].median()),
        "max_raw_distance_relative_change": _max_relative_change(
            eligible["median_raw_distance"]
        ),
        "max_adjusted_distance_relative_change": _max_relative_change(
            eligible["median_adjusted_distance"]
        ),
        "max_neighbor_age_relative_change": _max_relative_change(
            eligible["median_neighbor_age_bars"]
        ),
        "max_context_match_absolute_change": _max_absolute_change(
            eligible["median_joint_context_match_fraction"]
        ),
        "max_overlap_absolute_change": _max_absolute_change(
            eligible["median_overlap_raw_reference"]
        ),
        "minimum_neighbor_session_count": _finite_or_nan(
            eligible["median_neighbor_session_count"].min()
        ),
        "minimum_effective_session_count": _finite_or_nan(
            eligible["median_neighbor_session_effective_count"].min()
        ),
        "maximum_neighbor_session_share": _finite_or_nan(
            eligible["median_neighbor_max_session_share"].max()
        ),
    }
    summary_row["stability_state"] = _variation_state(summary_row, settings)
    summary = pd.DataFrame([summary_row], columns=SUMMARY_COLUMNS)
    return LorentzianSoftBlockStabilityReport(
        blocks=blocks,
        summary=summary,
        caveats=(
            "I blocchi sono fissati dalla prima sessione e non cambiano con dati futuri.",
            "Solo blocchi completi con il minimo di query dichiarato entrano nel riepilogo.",
            "La diversita' usa sedute distinte ed effective count del vicinato.",
            "Gli stati descrivono variazione geometrica, non accuratezza o rendimento.",
            "Il profilo e' congelato prima del confronto e non viene selezionato dai risultati.",
        ),
    )


def summarize_soft_block_stability_across_assets(
    summaries: pd.DataFrame,
) -> pd.DataFrame:
    """Riepiloga gli asset senza considerarli osservazioni indipendenti."""

    missing = sorted(set(SUMMARY_COLUMNS).difference(summaries.columns))
    if missing:
        raise ValueError(f"Colonne stabilita' soft mancanti: {missing}.")
    rows: list[dict[str, object]] = []
    for profile, group in summaries.groupby("profile", sort=False):
        states = group["stability_state"]
        rows.append(
            {
                "profile": profile,
                "assets": int(group["ticker"].nunique()),
                "low_variation_assets": int(
                    states.eq("LOW_BLOCK_VARIATION").sum()
                ),
                "moderate_variation_assets": int(
                    states.eq("MODERATE_BLOCK_VARIATION").sum()
                ),
                "high_variation_assets": int(
                    states.eq("HIGH_BLOCK_VARIATION").sum()
                ),
                "insufficient_assets": int(
                    states.eq("INSUFFICIENT_BLOCKS").sum()
                ),
                "median_eligible_blocks": float(
                    group["eligible_blocks"].median()
                ),
                "median_minimum_block_coverage": float(
                    group["minimum_block_coverage"].median()
                ),
                "median_max_raw_distance_relative_change": float(
                    group["max_raw_distance_relative_change"].median()
                ),
                "median_max_neighbor_age_relative_change": float(
                    group["max_neighbor_age_relative_change"].median()
                ),
                "median_max_context_match_absolute_change": float(
                    group["max_context_match_absolute_change"].median()
                ),
                "median_max_overlap_absolute_change": float(
                    group["max_overlap_absolute_change"].median()
                ),
                "median_minimum_effective_session_count": float(
                    group["minimum_effective_session_count"].median()
                ),
                "median_maximum_neighbor_session_share": float(
                    group["maximum_neighbor_session_share"].median()
                ),
            }
        )
    return pd.DataFrame(rows, columns=CROSS_ASSET_COLUMNS)


def _write_table(table: pd.DataFrame, columns: tuple[str, ...], path: str | Path) -> Path:
    destination = Path(path).expanduser()
    if destination.suffix.lower() != ".csv":
        raise ValueError("Il percorso di output deve terminare con .csv.")
    missing = sorted(set(columns).difference(table.columns))
    if missing:
        raise ValueError(f"Colonne output mancanti: {missing}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.loc[:, columns].to_csv(destination, index=False)
    return destination


def write_lorentzian_soft_block_details(
    blocks: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Esporta una riga per asset e blocco cronologico."""

    return _write_table(blocks, BLOCK_COLUMNS, path)


def write_lorentzian_soft_block_stability(
    summary: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Esporta una riga di stabilita' per asset e profilo congelato."""

    return _write_table(summary, SUMMARY_COLUMNS, path)
