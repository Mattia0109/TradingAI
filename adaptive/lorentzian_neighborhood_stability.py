"""Stress test descrittivo del vicinato Lorentziano intraday.

Il modulo confronta gli stessi stati correnti usando profondita' storiche
diverse. Misura copertura, distanza, eta' e sovrapposizione dei vicini senza
assegnare direzioni, usare outcome futuri o produrre elementi operativi.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DETAIL_COLUMNS = (
    "ticker",
    "timestamp",
    "filter_mode",
    "query_context",
    "history_limit",
    "reference_history_limit",
    "candidate_count",
    "neighbor_count",
    "distance_median",
    "neighbor_age_min_bars",
    "neighbor_age_median_bars",
    "neighbor_age_p90_bars",
    "neighbor_age_max_bars",
    "overlap_reference",
    "distance_ratio_reference",
)

SUMMARY_COLUMNS = (
    "ticker",
    "filter_mode",
    "history_limit",
    "reference_history_limit",
    "total_queries",
    "complete_queries",
    "coverage",
    "median_distance",
    "median_neighbor_age_bars",
    "p90_neighbor_age_bars",
    "median_overlap_reference",
    "p10_overlap_reference",
    "median_distance_ratio_reference",
    "reference_coverage",
    "reference_median_neighbor_age_bars",
    "overlap_state",
)


@dataclass(frozen=True)
class LorentzianNeighborhoodStabilityConfig:
    """Contratto dichiarato per il confronto fra memorie storiche."""

    history_limits: tuple[int, ...] = (520, 1_040, 2_080, 4_000)
    neighbors: int = 8
    minimum_candidates: int = 16
    embargo_bars: int = 4
    sample_stride: int = 4
    query_stride: int = 7
    minimum_overlap_queries: int = 30
    minimum_comparison_coverage: float = 0.50
    high_overlap_median: float = 0.75
    high_overlap_p10: float = 0.50
    moderate_overlap_median: float = 0.50
    moderate_overlap_p10: float = 0.25

    def __post_init__(self) -> None:
        limits = tuple(int(value) for value in self.history_limits)
        if not limits:
            raise ValueError("history_limits non puo' essere vuoto.")
        if len(set(limits)) != len(limits) or tuple(sorted(limits)) != limits:
            raise ValueError("history_limits deve essere crescente e senza duplicati.")
        if limits[0] < int(self.minimum_candidates):
            raise ValueError("La finestra minima e' troppo corta.")
        if int(self.neighbors) <= 0:
            raise ValueError("neighbors deve essere positivo.")
        if int(self.minimum_candidates) < int(self.neighbors):
            raise ValueError("minimum_candidates deve essere almeno neighbors.")
        if int(self.embargo_bars) < 0:
            raise ValueError("embargo_bars non puo' essere negativo.")
        if int(self.sample_stride) <= 0 or int(self.query_stride) <= 0:
            raise ValueError("Gli stride devono essere positivi.")
        if int(self.minimum_overlap_queries) <= 0:
            raise ValueError("minimum_overlap_queries deve essere positivo.")
        probabilities = (
            self.minimum_comparison_coverage,
            self.high_overlap_median,
            self.high_overlap_p10,
            self.moderate_overlap_median,
            self.moderate_overlap_p10,
        )
        if any(not 0.0 <= float(value) <= 1.0 for value in probabilities):
            raise ValueError("Le soglie di copertura/overlap devono essere in [0, 1].")
        if float(self.high_overlap_median) < float(self.moderate_overlap_median):
            raise ValueError("La soglia HIGH mediana deve superare MODERATE.")
        if float(self.high_overlap_p10) < float(self.moderate_overlap_p10):
            raise ValueError("La soglia HIGH p10 deve superare MODERATE.")


@dataclass(frozen=True)
class LorentzianNeighborhoodStabilityReport:
    """Dettagli e riepiloghi puramente geometrici."""

    details: pd.DataFrame
    summary: pd.DataFrame
    caveats: tuple[str, ...]
    research_only: bool = True


def _validate_inputs(
    ticker: str,
    normalized_features: pd.DataFrame,
    feature_states: pd.DataFrame,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    symbol = str(ticker).strip().upper()
    if not symbol:
        raise ValueError("ticker non puo' essere vuoto.")
    if not isinstance(normalized_features, pd.DataFrame) or normalized_features.empty:
        raise ValueError("normalized_features deve essere un DataFrame non vuoto.")
    if not isinstance(normalized_features.index, pd.DatetimeIndex):
        raise ValueError("normalized_features deve avere un DatetimeIndex.")
    if normalized_features.index.tz is None:
        raise ValueError("I timestamp devono avere una timezone dichiarata.")
    if normalized_features.index.hasnans or normalized_features.index.duplicated().any():
        raise ValueError("Timestamp Lorentziani mancanti o duplicati.")
    if not normalized_features.index.is_monotonic_increasing:
        raise ValueError("I timestamp Lorentziani devono essere ordinati.")
    if not isinstance(feature_states, pd.DataFrame):
        raise TypeError("feature_states deve essere un pandas DataFrame.")
    if not feature_states.index.equals(normalized_features.index):
        raise ValueError("feature_states deve avere lo stesso indice delle feature.")
    required = {"chop_segment", "squeeze_state"}
    missing = sorted(required.difference(feature_states.columns))
    if missing:
        raise ValueError(f"Stati descrittivi mancanti: {missing}.")
    numeric = normalized_features.apply(pd.to_numeric, errors="coerce").astype(float)
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("Le feature normalizzate non possono essere infinite.")
    return symbol, numeric, feature_states.loc[:, sorted(required)].copy()


def _session_phase(index: pd.DatetimeIndex) -> np.ndarray:
    local = index.tz_convert("America/New_York")
    minutes = local.hour * 60 + local.minute
    return np.select(
        [minutes < 10 * 60 + 30, minutes >= 15 * 60],
        ["OPEN", "CLOSE"],
        default="MID_SESSION",
    ).astype(object)


def _joint_contexts(phases: np.ndarray, states: pd.DataFrame) -> np.ndarray:
    chop = states["chop_segment"].astype("object").to_numpy()
    squeeze = states["squeeze_state"].astype("object").to_numpy()
    valid = (
        pd.notna(chop)
        & pd.notna(squeeze)
        & (chop != "INSUFFICIENT")
        & (squeeze != "INSUFFICIENT")
    )
    contexts = np.full(len(states), None, dtype=object)
    contexts[valid] = np.array(
        [
            f"{phase}|{chop_state}|{squeeze_state}"
            for phase, chop_state, squeeze_state in zip(
                phases[valid], chop[valid], squeeze[valid]
            )
        ],
        dtype=object,
    )
    return contexts


def _empty_report() -> LorentzianNeighborhoodStabilityReport:
    return LorentzianNeighborhoodStabilityReport(
        details=pd.DataFrame(columns=DETAIL_COLUMNS),
        summary=pd.DataFrame(columns=SUMMARY_COLUMNS),
        caveats=("Nessuna query dispone della profondita' storica dichiarata.",),
    )


def _overlap_state(
    history_limit: int,
    reference_limit: int,
    coverage: float,
    overlaps: pd.Series,
    config: LorentzianNeighborhoodStabilityConfig,
) -> str:
    if int(history_limit) == int(reference_limit):
        return "REFERENCE_WINDOW"
    clean = overlaps.dropna()
    if (
        not math.isfinite(float(coverage))
        or float(coverage) < float(config.minimum_comparison_coverage)
        or len(clean) < int(config.minimum_overlap_queries)
    ):
        return "INSUFFICIENT_COMPARISON"
    median = float(clean.median())
    p10 = float(clean.quantile(0.10))
    if (
        median >= float(config.high_overlap_median)
        and p10 >= float(config.high_overlap_p10)
    ):
        return "HIGH_NEIGHBOR_OVERLAP"
    if (
        median >= float(config.moderate_overlap_median)
        and p10 >= float(config.moderate_overlap_p10)
    ):
        return "MODERATE_NEIGHBOR_OVERLAP"
    return "LOW_NEIGHBOR_OVERLAP"


def _build_summary(
    details: pd.DataFrame,
    config: LorentzianNeighborhoodStabilityConfig,
) -> pd.DataFrame:
    if details.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    reference_limit = max(config.history_limits)
    rows: list[dict[str, object]] = []
    for (ticker, mode, history_limit), group in details.groupby(
        ["ticker", "filter_mode", "history_limit"], sort=True
    ):
        complete = group.loc[group["neighbor_count"].eq(config.neighbors)]
        total_queries = len(group)
        coverage = len(complete) / max(1, total_queries)
        rows.append(
            {
                "ticker": ticker,
                "filter_mode": mode,
                "history_limit": int(history_limit),
                "reference_history_limit": int(reference_limit),
                "total_queries": int(total_queries),
                "complete_queries": int(len(complete)),
                "coverage": float(coverage),
                "median_distance": float(complete["distance_median"].median())
                if not complete.empty
                else math.nan,
                "median_neighbor_age_bars": float(
                    complete["neighbor_age_median_bars"].median()
                )
                if not complete.empty
                else math.nan,
                "p90_neighbor_age_bars": float(
                    complete["neighbor_age_p90_bars"].quantile(0.90)
                )
                if not complete.empty
                else math.nan,
                "median_overlap_reference": float(
                    complete["overlap_reference"].median()
                )
                if complete["overlap_reference"].notna().any()
                else math.nan,
                "p10_overlap_reference": float(
                    complete["overlap_reference"].quantile(0.10)
                )
                if complete["overlap_reference"].notna().any()
                else math.nan,
                "median_distance_ratio_reference": float(
                    complete["distance_ratio_reference"].median()
                )
                if complete["distance_ratio_reference"].notna().any()
                else math.nan,
                "overlap_state": _overlap_state(
                    int(history_limit),
                    reference_limit,
                    coverage,
                    complete["overlap_reference"],
                    config,
                ),
            }
        )
    summary = pd.DataFrame(rows)
    reference = summary.loc[
        summary["history_limit"].eq(reference_limit),
        [
            "ticker",
            "filter_mode",
            "coverage",
            "median_neighbor_age_bars",
        ],
    ].rename(
        columns={
            "coverage": "reference_coverage",
            "median_neighbor_age_bars": "reference_median_neighbor_age_bars",
        }
    )
    summary = summary.merge(
        reference,
        on=["ticker", "filter_mode"],
        how="left",
        validate="many_to_one",
    )
    return summary.loc[:, SUMMARY_COLUMNS].sort_values(
        ["ticker", "filter_mode", "history_limit"], kind="mergesort"
    ).reset_index(drop=True)


def analyze_lorentzian_neighborhood_stability(
    ticker: str,
    normalized_features: pd.DataFrame,
    feature_states: pd.DataFrame,
    config: LorentzianNeighborhoodStabilityConfig | None = None,
) -> LorentzianNeighborhoodStabilityReport:
    """Confronta vicini storici annidati usando soltanto il passato."""

    settings = config or LorentzianNeighborhoodStabilityConfig()
    symbol, normalized, states = _validate_inputs(
        ticker, normalized_features, feature_states
    )
    matrix = normalized.to_numpy(dtype=float)
    finite_rows = np.isfinite(matrix).all(axis=1)
    phases = _session_phase(normalized.index)
    joint = _joint_contexts(phases, states)
    reference_limit = max(settings.history_limits)
    first_query = reference_limit + int(settings.embargo_bars) + 1
    query_positions = np.arange(
        first_query,
        len(normalized),
        int(settings.query_stride),
        dtype=int,
    )
    if query_positions.size == 0:
        return _empty_report()

    rows: list[dict[str, object]] = []
    for position in query_positions:
        for mode, context_values in (
            ("PHASE_ONLY", phases),
            ("JOINT_CONTEXT", joint),
        ):
            current_context = context_values[position]
            selections: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
            if current_context is not None and finite_rows[position]:
                last = position - int(settings.embargo_bars) - 1
                first = max(0, position - reference_limit)
                candidates = np.arange(first, last + 1, dtype=int)
                candidates = candidates[
                    candidates % int(settings.sample_stride) == 0
                ]
                candidates = candidates[
                    context_values[candidates] == current_context
                ]
                candidates = candidates[finite_rows[candidates]]
                if candidates.size:
                    distances = np.log1p(
                        np.abs(matrix[candidates] - matrix[position])
                    ).sum(axis=1)
                    for history_limit in settings.history_limits:
                        allowed = candidates >= position - int(history_limit)
                        positions = candidates[allowed]
                        local_distances = distances[allowed]
                        candidate_count = int(positions.size)
                        if candidate_count < int(settings.minimum_candidates):
                            selections[int(history_limit)] = (
                                np.array([], dtype=int),
                                np.array([], dtype=float),
                                candidate_count,
                            )
                            continue
                        order = np.lexsort((positions, local_distances))
                        selected_order = order[: int(settings.neighbors)]
                        selections[int(history_limit)] = (
                            positions[selected_order],
                            local_distances[selected_order],
                            candidate_count,
                        )

            reference_positions, reference_distances, _ = selections.get(
                reference_limit,
                (np.array([], dtype=int), np.array([], dtype=float), 0),
            )
            reference_complete = (
                reference_positions.size == int(settings.neighbors)
            )
            reference_median = (
                float(np.median(reference_distances))
                if reference_complete
                else math.nan
            )
            reference_set = set(reference_positions.tolist())

            for history_limit in settings.history_limits:
                selected, distances, candidate_count = selections.get(
                    int(history_limit),
                    (np.array([], dtype=int), np.array([], dtype=float), 0),
                )
                complete = selected.size == int(settings.neighbors)
                ages = position - selected if complete else np.array([], dtype=int)
                median_distance = (
                    float(np.median(distances)) if complete else math.nan
                )
                if complete and reference_complete:
                    overlap = len(set(selected.tolist()) & reference_set) / float(
                        settings.neighbors
                    )
                    if reference_median > 0.0:
                        distance_ratio = median_distance / reference_median
                    elif median_distance == 0.0:
                        distance_ratio = 1.0
                    else:
                        distance_ratio = math.nan
                else:
                    overlap = math.nan
                    distance_ratio = math.nan
                rows.append(
                    {
                        "ticker": symbol,
                        "timestamp": normalized.index[position],
                        "filter_mode": mode,
                        "query_context": str(current_context)
                        if current_context is not None
                        else "INSUFFICIENT",
                        "history_limit": int(history_limit),
                        "reference_history_limit": int(reference_limit),
                        "candidate_count": int(candidate_count),
                        "neighbor_count": int(selected.size),
                        "distance_median": median_distance,
                        "neighbor_age_min_bars": float(ages.min())
                        if complete
                        else math.nan,
                        "neighbor_age_median_bars": float(np.median(ages))
                        if complete
                        else math.nan,
                        "neighbor_age_p90_bars": float(np.quantile(ages, 0.90))
                        if complete
                        else math.nan,
                        "neighbor_age_max_bars": float(ages.max())
                        if complete
                        else math.nan,
                        "overlap_reference": float(overlap),
                        "distance_ratio_reference": float(distance_ratio),
                    }
                )

    details = pd.DataFrame(rows, columns=DETAIL_COLUMNS)
    summary = _build_summary(details, settings)
    return LorentzianNeighborhoodStabilityReport(
        details=details,
        summary=summary,
        caveats=(
            "Le query iniziano soltanto dopo la finestra storica massima dichiarata.",
            "Tutti i vicini precedono la query e rispettano embargo e campionamento.",
            "PHASE_ONLY e JOINT_CONTEXT sono confronti geometrici separati.",
            "Overlap e rapporti di distanza non misurano accuratezza o rendimento.",
            "Le etichette di overlap descrivono stabilita' dei vicini, non approvazioni.",
        ),
    )


def write_lorentzian_neighborhood_stability(
    summary: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Esporta il riepilogo geometrico in CSV tidy."""

    destination = Path(path).expanduser()
    if destination.suffix.lower() != ".csv":
        raise ValueError("lorentzian_stability_output deve terminare con .csv.")
    missing = sorted(set(SUMMARY_COLUMNS).difference(summary.columns))
    if missing:
        raise ValueError(f"Colonne riepilogo Lorentziano mancanti: {missing}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    summary.loc[:, SUMMARY_COLUMNS].to_csv(destination, index=False)
    return destination
