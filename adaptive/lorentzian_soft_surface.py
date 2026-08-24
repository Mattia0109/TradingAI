"""Superficie descrittiva di recenza e contesto per vicini Lorentziani.

La distanza grezza viene affiancata da penalita' dichiarate per eta' e
disaccordo contemporaneo degli stati CHOP/Squeeze. Il modulo studia soltanto
la geometria storica: non usa outcome futuri e non genera elementi operativi.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive.lorentzian_research import session_phase_context


DETAIL_COLUMNS = (
    "ticker",
    "timestamp",
    "profile",
    "recency_half_life_bars",
    "context_penalty",
    "candidate_count",
    "neighbor_count",
    "raw_distance_median",
    "adjusted_distance_median",
    "neighbor_age_min_bars",
    "neighbor_age_median_bars",
    "neighbor_age_p90_bars",
    "neighbor_age_max_bars",
    "joint_context_match_fraction",
    "mean_context_mismatches",
    "neighbor_session_count",
    "neighbor_session_effective_count",
    "neighbor_max_session_share",
    "overlap_raw_reference",
)

SUMMARY_COLUMNS = (
    "ticker",
    "profile",
    "recency_half_life_bars",
    "context_penalty",
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

CROSS_ASSET_COLUMNS = (
    "profile",
    "recency_half_life_bars",
    "context_penalty",
    "assets",
    "median_coverage",
    "median_raw_distance",
    "median_adjusted_distance",
    "median_neighbor_age_bars",
    "median_joint_context_match_fraction",
    "median_context_mismatches",
    "median_neighbor_session_count",
    "median_neighbor_session_effective_count",
    "median_neighbor_max_session_share",
    "median_overlap_raw_reference",
)


@dataclass(frozen=True)
class LorentzianSoftSurfaceConfig:
    """Griglia pre-dichiarata di penalita' geometriche."""

    history_limit: int = 4_000
    recency_half_lives: tuple[int, ...] = (520, 1_040, 2_080)
    context_penalties: tuple[float, ...] = (0.0, 0.25, 0.50, 1.0)
    neighbors: int = 8
    minimum_candidates: int = 16
    embargo_bars: int = 4
    sample_stride: int = 4
    query_stride: int = 7

    def __post_init__(self) -> None:
        if int(self.history_limit) < int(self.minimum_candidates):
            raise ValueError("history_limit troppo corto.")
        half_lives = tuple(int(value) for value in self.recency_half_lives)
        if (
            not half_lives
            or any(value <= 0 for value in half_lives)
            or tuple(sorted(set(half_lives))) != half_lives
        ):
            raise ValueError(
                "recency_half_lives deve essere crescente, positivo e unico."
            )
        if max(half_lives) > int(self.history_limit):
            raise ValueError("La half-life non puo' superare history_limit.")
        penalties = tuple(float(value) for value in self.context_penalties)
        if (
            not penalties
            or any(value < 0.0 for value in penalties)
            or tuple(sorted(set(penalties))) != penalties
        ):
            raise ValueError(
                "context_penalties deve essere crescente, non negativo e unico."
            )
        if int(self.neighbors) <= 0:
            raise ValueError("neighbors deve essere positivo.")
        if int(self.minimum_candidates) < int(self.neighbors):
            raise ValueError("minimum_candidates deve essere almeno neighbors.")
        if int(self.embargo_bars) < 0:
            raise ValueError("embargo_bars non puo' essere negativo.")
        if int(self.sample_stride) <= 0 or int(self.query_stride) <= 0:
            raise ValueError("Gli stride devono essere positivi.")


@dataclass(frozen=True)
class LorentzianSoftSurfaceReport:
    """Dettaglio per query e riepilogo per profilo."""

    details: pd.DataFrame
    summary: pd.DataFrame
    caveats: tuple[str, ...]
    research_only: bool = True


def _profile_name(half_life: int, context_penalty: float) -> str:
    if int(half_life) == 0:
        return "REFERENCE_RAW"
    penalty = f"{float(context_penalty):.2f}".rstrip("0").rstrip(".")
    return f"HL_{int(half_life)}_CTX_{penalty}"


def _profiles(
    config: LorentzianSoftSurfaceConfig,
) -> tuple[tuple[int, float, str], ...]:
    output = [(0, 0.0, _profile_name(0, 0.0))]
    output.extend(
        (half_life, penalty, _profile_name(half_life, penalty))
        for half_life in config.recency_half_lives
        for penalty in config.context_penalties
    )
    return tuple(output)


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
    if (
        normalized_features.index.hasnans
        or normalized_features.index.duplicated().any()
        or not normalized_features.index.is_monotonic_increasing
    ):
        raise ValueError("Timestamp Lorentziani mancanti, duplicati o disordinati.")
    if not isinstance(feature_states, pd.DataFrame):
        raise TypeError("feature_states deve essere un pandas DataFrame.")
    if not feature_states.index.equals(normalized_features.index):
        raise ValueError("feature_states deve avere lo stesso indice delle feature.")
    required = ("chop_segment", "squeeze_state")
    missing = sorted(set(required).difference(feature_states.columns))
    if missing:
        raise ValueError(f"Stati descrittivi mancanti: {missing}.")
    numeric = normalized_features.apply(pd.to_numeric, errors="coerce").astype(float)
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("Le feature normalizzate non possono essere infinite.")
    return symbol, numeric, feature_states.loc[:, required].copy()


def _valid_state(value: object) -> bool:
    return bool(pd.notna(value) and str(value) != "INSUFFICIENT")


def _summarize(
    details: pd.DataFrame,
    config: LorentzianSoftSurfaceConfig,
) -> pd.DataFrame:
    if details.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    rows: list[dict[str, object]] = []
    for (ticker, profile, half_life, penalty), group in details.groupby(
        ["ticker", "profile", "recency_half_life_bars", "context_penalty"],
        sort=False,
    ):
        complete = group.loc[group["neighbor_count"].eq(config.neighbors)]
        rows.append(
            {
                "ticker": ticker,
                "profile": profile,
                "recency_half_life_bars": int(half_life),
                "context_penalty": float(penalty),
                "total_queries": int(len(group)),
                "complete_queries": int(len(complete)),
                "coverage": len(complete) / max(1, len(group)),
                "median_raw_distance": float(complete["raw_distance_median"].median())
                if not complete.empty
                else math.nan,
                "median_adjusted_distance": float(
                    complete["adjusted_distance_median"].median()
                )
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
                "median_joint_context_match_fraction": float(
                    complete["joint_context_match_fraction"].median()
                )
                if not complete.empty
                else math.nan,
                "median_context_mismatches": float(
                    complete["mean_context_mismatches"].median()
                )
                if not complete.empty
                else math.nan,
                "median_neighbor_session_count": float(
                    complete["neighbor_session_count"].median()
                )
                if not complete.empty
                else math.nan,
                "median_neighbor_session_effective_count": float(
                    complete["neighbor_session_effective_count"].median()
                )
                if not complete.empty
                else math.nan,
                "median_neighbor_max_session_share": float(
                    complete["neighbor_max_session_share"].median()
                )
                if not complete.empty
                else math.nan,
                "median_overlap_raw_reference": float(
                    complete["overlap_raw_reference"].median()
                )
                if complete["overlap_raw_reference"].notna().any()
                else math.nan,
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def analyze_lorentzian_soft_surface(
    ticker: str,
    normalized_features: pd.DataFrame,
    feature_states: pd.DataFrame,
    config: LorentzianSoftSurfaceConfig | None = None,
) -> LorentzianSoftSurfaceReport:
    """Valuta una griglia recenza x contesto con candidati solo storici."""

    settings = config or LorentzianSoftSurfaceConfig()
    symbol, normalized, states = _validate_inputs(
        ticker, normalized_features, feature_states
    )
    profiles = _profiles(settings)
    matrix = normalized.to_numpy(dtype=float)
    finite_rows = np.isfinite(matrix).all(axis=1)
    phases = session_phase_context(normalized.index).to_numpy(dtype=object)
    chop = states["chop_segment"].astype(object).to_numpy()
    squeeze = states["squeeze_state"].astype(object).to_numpy()
    local_index = normalized.index.tz_convert("America/New_York")
    session_dates = np.array([value.date() for value in local_index], dtype=object)
    first_query = int(settings.history_limit) + int(settings.embargo_bars) + 1
    query_positions = np.arange(
        first_query,
        len(normalized),
        int(settings.query_stride),
        dtype=int,
    )
    rows: list[dict[str, object]] = []

    for position in query_positions:
        current_states_valid = _valid_state(chop[position]) and _valid_state(
            squeeze[position]
        )
        last = position - int(settings.embargo_bars) - 1
        first = max(0, position - int(settings.history_limit))
        candidates = np.arange(first, last + 1, dtype=int)
        candidates = candidates[candidates % int(settings.sample_stride) == 0]
        candidates = candidates[phases[candidates] == phases[position]]
        candidates = candidates[finite_rows[candidates]]
        if current_states_valid and candidates.size:
            valid_context = np.array(
                [
                    _valid_state(chop[candidate])
                    and _valid_state(squeeze[candidate])
                    for candidate in candidates
                ],
                dtype=bool,
            )
            candidates = candidates[valid_context]

        candidate_count = int(candidates.size)
        if not finite_rows[position] or not current_states_valid or candidate_count == 0:
            raw_distances = np.array([], dtype=float)
            ages = np.array([], dtype=float)
            mismatches = np.array([], dtype=float)
        else:
            raw_distances = np.log1p(
                np.abs(matrix[candidates] - matrix[position])
            ).sum(axis=1)
            ages = (position - candidates).astype(float)
            mismatches = (
                (chop[candidates] != chop[position]).astype(float)
                + (squeeze[candidates] != squeeze[position]).astype(float)
            )

        selections: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for half_life, context_penalty, profile in profiles:
            if candidate_count < int(settings.minimum_candidates):
                selections[profile] = (
                    np.array([], dtype=int),
                    np.array([], dtype=float),
                    np.array([], dtype=float),
                )
                continue
            if int(half_life) == 0:
                adjusted = raw_distances.copy()
            else:
                recency = math.log(2.0) * ages / float(half_life)
                adjusted = (
                    raw_distances
                    + recency
                    + float(context_penalty) * mismatches
                )
            order = np.lexsort((candidates, adjusted))
            selected_order = order[: int(settings.neighbors)]
            selections[profile] = (
                candidates[selected_order],
                raw_distances[selected_order],
                adjusted[selected_order],
            )

        reference_positions = selections["REFERENCE_RAW"][0]
        reference_complete = reference_positions.size == int(settings.neighbors)
        reference_set = set(reference_positions.tolist())
        for half_life, context_penalty, profile in profiles:
            selected, selected_raw, selected_adjusted = selections[profile]
            complete = selected.size == int(settings.neighbors)
            selected_ages = position - selected if complete else np.array([], dtype=int)
            selected_mismatches = (
                (
                    (chop[selected] != chop[position]).astype(float)
                    + (squeeze[selected] != squeeze[position]).astype(float)
                )
                if complete
                else np.array([], dtype=float)
            )
            if complete:
                _, session_counts = np.unique(
                    session_dates[selected], return_counts=True
                )
                session_shares = session_counts.astype(float) / float(
                    session_counts.sum()
                )
                session_effective_count = 1.0 / float(
                    np.square(session_shares).sum()
                )
                max_session_share = float(session_shares.max())
            else:
                session_counts = np.array([], dtype=int)
                session_effective_count = math.nan
                max_session_share = math.nan
            overlap = (
                len(set(selected.tolist()) & reference_set) / float(settings.neighbors)
                if complete and reference_complete
                else math.nan
            )
            rows.append(
                {
                    "ticker": symbol,
                    "timestamp": normalized.index[position],
                    "profile": profile,
                    "recency_half_life_bars": int(half_life),
                    "context_penalty": float(context_penalty),
                    "candidate_count": candidate_count,
                    "neighbor_count": int(selected.size),
                    "raw_distance_median": float(np.median(selected_raw))
                    if complete
                    else math.nan,
                    "adjusted_distance_median": float(np.median(selected_adjusted))
                    if complete
                    else math.nan,
                    "neighbor_age_min_bars": float(selected_ages.min())
                    if complete
                    else math.nan,
                    "neighbor_age_median_bars": float(np.median(selected_ages))
                    if complete
                    else math.nan,
                    "neighbor_age_p90_bars": float(
                        np.quantile(selected_ages, 0.90)
                    )
                    if complete
                    else math.nan,
                    "neighbor_age_max_bars": float(selected_ages.max())
                    if complete
                    else math.nan,
                    "joint_context_match_fraction": float(
                        np.mean(selected_mismatches == 0.0)
                    )
                    if complete
                    else math.nan,
                    "mean_context_mismatches": float(
                        np.mean(selected_mismatches)
                    )
                    if complete
                    else math.nan,
                    "neighbor_session_count": int(
                        session_counts.size
                    )
                    if complete
                    else 0,
                    "neighbor_session_effective_count": float(
                        session_effective_count
                    ),
                    "neighbor_max_session_share": float(max_session_share),
                    "overlap_raw_reference": float(overlap),
                }
            )

    details = pd.DataFrame(rows, columns=DETAIL_COLUMNS)
    return LorentzianSoftSurfaceReport(
        details=details,
        summary=_summarize(details, settings),
        caveats=(
            "La penalita' di recenza vale ln(2) all'eta' pari alla half-life.",
            "Il disaccordo CHOP e Squeeze aggiunge da zero a due penalita' dichiarate.",
            "Tutti i candidati sono storici, della stessa fase e oltre l'embargo.",
            "La griglia non viene scelta con outcome futuri e non ottimizza rendimento.",
            "Le metriche descrivono soltanto eta', distanza, contesto e diversita' dei vicini.",
        ),
    )


def summarize_soft_surface_across_assets(
    summaries: pd.DataFrame,
) -> pd.DataFrame:
    """Calcola mediane cross-asset senza trattarle come prove indipendenti."""

    missing = sorted(set(SUMMARY_COLUMNS).difference(summaries.columns))
    if missing:
        raise ValueError(f"Colonne superficie Lorentziana mancanti: {missing}.")
    rows: list[dict[str, object]] = []
    for (profile, half_life, penalty), group in summaries.groupby(
        ["profile", "recency_half_life_bars", "context_penalty"], sort=False
    ):
        rows.append(
            {
                "profile": profile,
                "recency_half_life_bars": int(half_life),
                "context_penalty": float(penalty),
                "assets": int(group["ticker"].nunique()),
                "median_coverage": float(group["coverage"].median()),
                "median_raw_distance": float(group["median_raw_distance"].median()),
                "median_adjusted_distance": float(
                    group["median_adjusted_distance"].median()
                ),
                "median_neighbor_age_bars": float(
                    group["median_neighbor_age_bars"].median()
                ),
                "median_joint_context_match_fraction": float(
                    group["median_joint_context_match_fraction"].median()
                ),
                "median_context_mismatches": float(
                    group["median_context_mismatches"].median()
                ),
                "median_neighbor_session_count": float(
                    group["median_neighbor_session_count"].median()
                ),
                "median_neighbor_session_effective_count": float(
                    group[
                        "median_neighbor_session_effective_count"
                    ].median()
                ),
                "median_neighbor_max_session_share": float(
                    group["median_neighbor_max_session_share"].median()
                ),
                "median_overlap_raw_reference": float(
                    group["median_overlap_raw_reference"].median()
                ),
            }
        )
    return pd.DataFrame(rows, columns=CROSS_ASSET_COLUMNS)


def write_lorentzian_soft_surface(
    summary: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Esporta i riepiloghi per asset e profilo in CSV tidy."""

    destination = Path(path).expanduser()
    if destination.suffix.lower() != ".csv":
        raise ValueError("lorentzian_soft_surface_output deve terminare con .csv.")
    missing = sorted(set(SUMMARY_COLUMNS).difference(summary.columns))
    if missing:
        raise ValueError(f"Colonne superficie Lorentziana mancanti: {missing}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    summary.loc[:, SUMMARY_COLUMNS].to_csv(destination, index=False)
    return destination
