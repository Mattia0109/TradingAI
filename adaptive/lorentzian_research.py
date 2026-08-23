"""Descrittori Lorentziani causali per la ricerca intraday.

Il modulo confronta lo stato corrente delle feature con stati storici gia'
osservati. Non assegna etichette direzionali, non usa outcome futuri e non
genera segnali, ordini, size, stop, leva o P&L.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_LORENTZIAN_FEATURES = (
    "squeeze_momentum_pct_close",
    "squeeze_momentum_change_pct_close",
    "choppiness",
    "cmf",
)

LORENTZIAN_DESCRIPTOR_COLUMNS = (
    "lorentzian_neighbor_count",
    "lorentzian_distance_min",
    "lorentzian_distance_median",
    "lorentzian_distance_iqr",
    "lorentzian_local_density",
    "lorentzian_neighbor_age_min_bars",
    "lorentzian_neighbor_age_median_bars",
)


class LorentzianParity(str, Enum):
    """Relazione verificabile con i riferimenti forniti."""

    EXACT_REFERENCE_DISTANCE = "EXACT_REFERENCE_DISTANCE"
    CLEAN_ROOM_CAUSAL_DESCRIPTOR = "CLEAN_ROOM_CAUSAL_DESCRIPTOR"
    CLASSIFIER_NOT_REPRODUCED = "CLASSIFIER_NOT_REPRODUCED"


@dataclass(frozen=True)
class LorentzianResearchConfig:
    """Configurazione prudenziale del vicinato storico a 15 minuti."""

    feature_columns: tuple[str, ...] = DEFAULT_LORENTZIAN_FEATURES
    normalization_window: int = 260
    normalization_min_periods: int = 130
    neighbors: int = 8
    minimum_candidates: int = 16
    embargo_bars: int = 4
    sample_stride: int = 4
    history_limit: int = 2_000
    clip_normalized_value: float = 8.0
    minimum_scale: float = 1e-9

    def __post_init__(self) -> None:
        if not 2 <= len(self.feature_columns) <= 5:
            raise ValueError("Servono da 2 a 5 feature Lorentziane.")
        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("Le feature Lorentziane devono essere uniche.")
        if any(not str(name).strip() for name in self.feature_columns):
            raise ValueError("I nomi delle feature non possono essere vuoti.")
        if int(self.normalization_window) < 4:
            raise ValueError("normalization_window deve essere almeno 4.")
        if not 2 <= int(self.normalization_min_periods) <= int(
            self.normalization_window
        ):
            raise ValueError("normalization_min_periods non valido.")
        if int(self.neighbors) <= 0:
            raise ValueError("neighbors deve essere positivo.")
        if int(self.minimum_candidates) < int(self.neighbors):
            raise ValueError("minimum_candidates deve essere almeno neighbors.")
        if int(self.embargo_bars) < 0:
            raise ValueError("embargo_bars non puo' essere negativo.")
        if int(self.sample_stride) <= 0:
            raise ValueError("sample_stride deve essere positivo.")
        if int(self.history_limit) < int(self.minimum_candidates):
            raise ValueError("history_limit troppo corto.")
        if float(self.clip_normalized_value) <= 0.0:
            raise ValueError("clip_normalized_value deve essere positivo.")
        if float(self.minimum_scale) <= 0.0:
            raise ValueError("minimum_scale deve essere positivo.")


@dataclass(frozen=True)
class LorentzianResearchReport:
    """Serie storiche descrittive, senza decisioni operative."""

    descriptors: pd.DataFrame
    normalized_features: pd.DataFrame
    feature_columns: tuple[str, ...]
    parity: dict[str, LorentzianParity]
    caveats: tuple[str, ...]
    research_only: bool = True


@dataclass(frozen=True)
class LorentzianDistributionSummary:
    """Riepilogo dell'intero campione, mai dell'ultima barra."""

    total_rows: int
    complete_rows: int
    coverage: float
    median_distance: float
    distance_p90: float
    median_density: float
    median_neighbor_age_bars: float


def lorentzian_distance(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    """Somma ``log(1 + abs(delta))`` del riferimento senza radice."""

    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    if left_values.ndim != 1 or right_values.ndim != 1:
        raise ValueError("I vettori devono essere monodimensionali.")
    if left_values.shape != right_values.shape:
        raise ValueError("I vettori devono avere la stessa dimensione.")
    if not 2 <= left_values.size <= 5:
        raise ValueError("La formula dichiarata accetta da 2 a 5 feature.")
    if not np.isfinite(left_values).all() or not np.isfinite(
        right_values
    ).all():
        raise ValueError("Le feature devono essere finite.")
    return float(np.log1p(np.abs(left_values - right_values)).sum())


def session_phase_context(index: pd.DatetimeIndex) -> pd.Series:
    """Etichetta OPEN/MID/CLOSE per timestamp gia' in ora di New York."""

    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index deve essere un DatetimeIndex.")
    if index.tz is None:
        raise ValueError("I timestamp devono avere una timezone dichiarata.")
    local = index.tz_convert("America/New_York")
    minutes = local.hour * 60 + local.minute
    labels = np.select(
        [minutes < 10 * 60 + 30, minutes >= 15 * 60],
        ["OPEN", "CLOSE"],
        default="MID_SESSION",
    )
    return pd.Series(labels, index=index, dtype="object", name="session_phase")


class CausalLorentzianResearchEngine:
    """Costruisce un vicinato storico robusto e strettamente causale."""

    def __init__(self, config: LorentzianResearchConfig | None = None) -> None:
        self.config = config or LorentzianResearchConfig()

    def _prepare_features(self, values: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(values, pd.DataFrame):
            raise TypeError("values deve essere un pandas DataFrame.")
        if values.empty:
            raise ValueError("values non puo' essere vuoto.")
        if not isinstance(values.index, pd.DatetimeIndex):
            raise ValueError("values deve avere un DatetimeIndex.")
        if values.index.hasnans:
            raise ValueError("Sono presenti timestamp non validi.")
        if values.index.duplicated().any():
            raise ValueError("Sono presenti timestamp duplicati.")
        if not values.index.is_monotonic_increasing:
            raise ValueError("I timestamp devono essere ordinati.")
        missing = sorted(set(self.config.feature_columns).difference(values.columns))
        if missing:
            raise ValueError(f"Feature Lorentziane mancanti: {missing}.")

        numeric = values.loc[:, self.config.feature_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        if np.isinf(numeric.to_numpy(dtype=float)).any():
            raise ValueError("Le feature non possono contenere valori infiniti.")
        return numeric.astype(float)

    def _normalize(self, numeric: pd.DataFrame) -> pd.DataFrame:
        """Normalizzazione mediana/IQR trailing, senza stime globali."""

        config = self.config
        rolling = numeric.rolling(
            int(config.normalization_window),
            min_periods=int(config.normalization_min_periods),
        )
        median = rolling.median()
        lower = rolling.quantile(0.25)
        upper = rolling.quantile(0.75)
        scale = (upper - lower) / 1.349
        scale = scale.where(scale > float(config.minimum_scale))
        fallback = pd.DataFrame(
            float(config.minimum_scale),
            index=numeric.index,
            columns=numeric.columns,
        )
        safe_scale = scale.combine_first(fallback)
        normalized = (numeric - median) / safe_scale
        return normalized.clip(
            lower=-float(config.clip_normalized_value),
            upper=float(config.clip_normalized_value),
        )

    @staticmethod
    def _prepare_contexts(
        index: pd.DatetimeIndex,
        contexts: pd.Series | None,
    ) -> np.ndarray | None:
        if contexts is None:
            return None
        if not isinstance(contexts, pd.Series):
            raise TypeError("contexts deve essere una pandas Series.")
        if not contexts.index.equals(index):
            raise ValueError("contexts deve avere lo stesso indice delle feature.")
        if contexts.isna().any():
            raise ValueError("contexts non puo' contenere valori mancanti.")
        return contexts.astype(str).to_numpy(dtype=object)

    def compute(
        self,
        values: pd.DataFrame,
        contexts: pd.Series | None = None,
    ) -> LorentzianResearchReport:
        """Calcola descrittori usando soltanto righe storiche ammissibili."""

        config = self.config
        numeric = self._prepare_features(values)
        normalized = self._normalize(numeric)
        context_values = self._prepare_contexts(normalized.index, contexts)
        matrix = normalized.to_numpy(dtype=float)
        rows = len(normalized)

        output = {
            name: np.full(rows, np.nan, dtype=float)
            for name in LORENTZIAN_DESCRIPTOR_COLUMNS
        }
        output["lorentzian_neighbor_count"] = np.zeros(rows, dtype=float)

        for position in range(rows):
            current = matrix[position]
            if not np.isfinite(current).all():
                continue

            last_candidate = position - int(config.embargo_bars) - 1
            if last_candidate < 0:
                continue
            first_candidate = max(
                0,
                position - int(config.history_limit),
            )
            candidate_positions = np.arange(
                first_candidate,
                last_candidate + 1,
                dtype=int,
            )
            candidate_positions = candidate_positions[
                candidate_positions % int(config.sample_stride) == 0
            ]
            if context_values is not None:
                candidate_positions = candidate_positions[
                    context_values[candidate_positions]
                    == context_values[position]
                ]
            if candidate_positions.size < int(config.minimum_candidates):
                continue

            candidates = matrix[candidate_positions]
            valid = np.isfinite(candidates).all(axis=1)
            candidate_positions = candidate_positions[valid]
            candidates = candidates[valid]
            if candidate_positions.size < int(config.minimum_candidates):
                continue

            distances = np.log1p(np.abs(candidates - current)).sum(axis=1)
            order = np.lexsort((candidate_positions, distances))
            selected = order[: int(config.neighbors)]
            selected_distances = distances[selected]
            selected_positions = candidate_positions[selected]
            ages = position - selected_positions

            output["lorentzian_neighbor_count"][position] = float(
                selected_distances.size
            )
            minimum = float(selected_distances.min())
            median_distance = float(np.median(selected_distances))
            distance_iqr = float(
                np.quantile(selected_distances, 0.75)
                - np.quantile(selected_distances, 0.25)
            )
            output["lorentzian_distance_min"][position] = minimum
            output["lorentzian_distance_median"][position] = median_distance
            output["lorentzian_distance_iqr"][position] = distance_iqr
            output["lorentzian_local_density"][position] = 1.0 / (
                1.0 + median_distance
            )
            output["lorentzian_neighbor_age_min_bars"][position] = float(
                ages.min()
            )
            output["lorentzian_neighbor_age_median_bars"][position] = float(
                np.median(ages)
            )

        descriptors = pd.DataFrame(output, index=normalized.index)
        parity = {
            "distance_formula": LorentzianParity.EXACT_REFERENCE_DISTANCE,
            "historical_neighborhood": (
                LorentzianParity.CLEAN_ROOM_CAUSAL_DESCRIPTOR
            ),
            "directional_classifier": LorentzianParity.CLASSIFIER_NOT_REPRODUCED,
        }
        caveats = (
            "La formula scelta e' sum(log1p(abs(delta))); la variante con "
            "radice quadrata dell'altra sorgente non e' equivalente.",
            "Il classificatore direzionale originale non e' riprodotto: "
            "dipende da librerie esterne non incluse nelle sorgenti.",
            "La normalizzazione mediana/IQR, la ricerca esatta, l'embargo e "
            "il filtro di contesto sono componenti clean-room descrittivi.",
        )
        return LorentzianResearchReport(
            descriptors=descriptors,
            normalized_features=normalized,
            feature_columns=tuple(config.feature_columns),
            parity=parity,
            caveats=caveats,
        )


def summarize_lorentzian_distribution(
    report: LorentzianResearchReport,
) -> LorentzianDistributionSummary:
    """Riassume tutte le righe complete senza selezionare il valore corrente."""

    if not isinstance(report, LorentzianResearchReport):
        raise TypeError("report deve essere LorentzianResearchReport.")
    frame = report.descriptors
    complete = frame.loc[
        frame["lorentzian_neighbor_count"] > 0,
        list(LORENTZIAN_DESCRIPTOR_COLUMNS),
    ].dropna()
    total_rows = len(frame)
    complete_rows = len(complete)
    if complete.empty:
        return LorentzianDistributionSummary(
            total_rows=total_rows,
            complete_rows=0,
            coverage=0.0,
            median_distance=math.nan,
            distance_p90=math.nan,
            median_density=math.nan,
            median_neighbor_age_bars=math.nan,
        )
    return LorentzianDistributionSummary(
        total_rows=total_rows,
        complete_rows=complete_rows,
        coverage=complete_rows / max(1, total_rows),
        median_distance=float(
            complete["lorentzian_distance_median"].median()
        ),
        distance_p90=float(
            complete["lorentzian_distance_median"].quantile(0.90)
        ),
        median_density=float(complete["lorentzian_local_density"].median()),
        median_neighbor_age_bars=float(
            complete["lorentzian_neighbor_age_median_bars"].median()
        ),
    )
