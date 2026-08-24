"""Stabilita' storica dei contesti intraday descrittivi.

Il modulo confronta blocchi fissi gia' osservati usando distribuzioni degli
stati fase x CHOP x Squeeze e delle transizioni intraseduta. Non usa prezzi
futuri, rendimenti, etichette direzionali o componenti operative.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive.intraday_regime_atlas import IntradayRegimeAtlasReport


class ContextDistributionShift(str, Enum):
    LOW_SHIFT = "LOW_SHIFT"
    MODERATE_SHIFT = "MODERATE_SHIFT"
    HIGH_SHIFT = "HIGH_SHIFT"
    INSUFFICIENT = "INSUFFICIENT"


@dataclass(frozen=True)
class IntradayContextStabilityConfig:
    """Soglie esplicite per confronti fra blocchi completi consecutivi."""

    interval_minutes: int = 15
    minimum_complete_blocks: int = 4
    minimum_occupancy_rows_per_block: int = 40
    minimum_transition_rows_per_block: int = 80
    moderate_total_variation: float = 0.10
    high_total_variation: float = 0.25
    moderate_js_divergence: float = 0.02
    high_js_divergence: float = 0.08
    calibration_permutations: int = 96
    calibration_quantile: float = 0.95
    calibration_seed: int = 20260824

    def __post_init__(self) -> None:
        for name in (
            "interval_minutes",
            "minimum_complete_blocks",
            "minimum_occupancy_rows_per_block",
            "minimum_transition_rows_per_block",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} deve essere positivo.")
        if not (
            0.0
            < self.moderate_total_variation
            < self.high_total_variation
            <= 1.0
        ):
            raise ValueError("Le soglie di variazione totale non sono ordinate.")
        if not (
            0.0
            < self.moderate_js_divergence
            < self.high_js_divergence
            <= 1.0
        ):
            raise ValueError("Le soglie Jensen-Shannon non sono ordinate.")
        if int(self.calibration_permutations) < 20:
            raise ValueError("calibration_permutations deve essere almeno 20.")
        if not 0.5 < float(self.calibration_quantile) < 1.0:
            raise ValueError("calibration_quantile deve essere in (0.5, 1).")
        if int(self.calibration_seed) < 0:
            raise ValueError("calibration_seed non puo' essere negativo.")


BLOCK_OCCUPANCY_COLUMNS = (
    "ticker",
    "source_status",
    "block",
    "session_phase",
    "observations",
    "sessions",
    "context_count",
    "entropy_bits",
    "effective_contexts",
)

DRIFT_COLUMNS = (
    "ticker",
    "source_status",
    "component",
    "scope",
    "reference_block",
    "current_block",
    "reference_observations",
    "current_observations",
    "categories",
    "total_variation",
    "jensen_shannon_bits",
    "null_total_variation_quantile",
    "null_jensen_shannon_quantile",
    "total_variation_excess",
    "jensen_shannon_excess",
    "shift",
)

PERSISTENCE_COLUMNS = (
    "ticker",
    "source_status",
    "component",
    "scope",
    "block_transitions",
    "elevated_transitions",
    "high_transitions",
    "longest_elevated_run",
    "latest_shift",
    "max_total_variation",
    "max_jensen_shannon_bits",
    "max_total_variation_excess",
    "max_jensen_shannon_excess",
    "pattern",
)


@dataclass(frozen=True)
class IntradayContextStabilityReport:
    block_occupancy: pd.DataFrame
    occupancy_drift: pd.DataFrame
    transition_drift: pd.DataFrame
    persistence: pd.DataFrame
    caveats: tuple[str, ...]
    research_only: bool = True


def _empty(columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _distribution(values: pd.Series, categories: list[str]) -> np.ndarray:
    if values.empty:
        return np.zeros(len(categories), dtype=float)
    counts = values.astype(str).value_counts().reindex(categories, fill_value=0)
    return counts.to_numpy(dtype=float) / float(counts.sum())


def _total_variation(left: np.ndarray, right: np.ndarray) -> float:
    return float(0.5 * np.abs(left - right).sum())


def _jensen_shannon_bits(left: np.ndarray, right: np.ndarray) -> float:
    midpoint = 0.5 * (left + right)

    def divergence(values: np.ndarray) -> float:
        positive = values > 0.0
        return float(
            np.sum(values[positive] * np.log2(values[positive] / midpoint[positive]))
        )

    return float(0.5 * divergence(left) + 0.5 * divergence(right))


def _stable_seed(base_seed: int, key: str) -> int:
    digest = hashlib.blake2b(
        key.encode("utf-8"),
        digest_size=8,
    ).digest()
    return (int.from_bytes(digest, "little") + int(base_seed)) % (2**63 - 1)


def _calibration_null(
    left_values: pd.Series,
    right_values: pd.Series,
    categories: list[str],
    config: IntradayContextStabilityConfig,
    comparison_key: str,
) -> tuple[float, float]:
    """Stima il drift casuale preservando numerosita' e frequenze aggregate."""

    left_strings = left_values.astype(str).to_numpy(dtype=object)
    right_strings = right_values.astype(str).to_numpy(dtype=object)
    pooled_strings = np.concatenate([left_strings, right_strings])
    if left_strings.size == 0 or right_strings.size == 0:
        return math.nan, math.nan
    category_positions = {value: position for position, value in enumerate(categories)}
    codes = np.fromiter(
        (category_positions[str(value)] for value in pooled_strings),
        dtype=np.int64,
        count=pooled_strings.size,
    )
    rng = np.random.default_rng(
        _stable_seed(config.calibration_seed, comparison_key)
    )
    null_tvd = np.empty(config.calibration_permutations, dtype=float)
    null_js = np.empty(config.calibration_permutations, dtype=float)
    left_size = int(left_strings.size)
    category_count = len(categories)
    for position in range(config.calibration_permutations):
        shuffled = rng.permutation(codes)
        left = np.bincount(
            shuffled[:left_size], minlength=category_count
        ).astype(float)
        right = np.bincount(
            shuffled[left_size:], minlength=category_count
        ).astype(float)
        left /= float(left.sum())
        right /= float(right.sum())
        null_tvd[position] = _total_variation(left, right)
        null_js[position] = _jensen_shannon_bits(left, right)
    return (
        float(np.quantile(null_tvd, config.calibration_quantile)),
        float(np.quantile(null_js, config.calibration_quantile)),
    )


def _shift_level(
    observations_left: int,
    observations_right: int,
    complete_blocks: int,
    minimum_rows: int,
    total_variation: float,
    jensen_shannon: float,
    null_total_variation: float,
    null_jensen_shannon: float,
    config: IntradayContextStabilityConfig,
) -> str:
    if (
        complete_blocks < config.minimum_complete_blocks
        or observations_left < minimum_rows
        or observations_right < minimum_rows
        or not math.isfinite(null_total_variation)
        or not math.isfinite(null_jensen_shannon)
    ):
        return ContextDistributionShift.INSUFFICIENT.value
    tvd_above_null = total_variation > null_total_variation
    js_above_null = jensen_shannon > null_jensen_shannon
    if (
        (tvd_above_null and total_variation >= config.high_total_variation)
        or (js_above_null and jensen_shannon >= config.high_js_divergence)
    ):
        return ContextDistributionShift.HIGH_SHIFT.value
    if (
        (tvd_above_null and total_variation >= config.moderate_total_variation)
        or (js_above_null and jensen_shannon >= config.moderate_js_divergence)
    ):
        return ContextDistributionShift.MODERATE_SHIFT.value
    return ContextDistributionShift.LOW_SHIFT.value


def _entropy(values: pd.Series) -> tuple[float, float]:
    if values.empty:
        return math.nan, math.nan
    shares = values.value_counts(normalize=True).to_numpy(dtype=float)
    entropy = float(-(shares * np.log2(shares)).sum())
    return entropy, float(2.0**entropy)


def _transition_events(
    assignments: pd.DataFrame,
    interval_minutes: int,
) -> pd.DataFrame:
    if assignments.empty:
        return pd.DataFrame()
    ordered = assignments.sort_values(
        ["ticker", "timestamp"], kind="mergesort"
    ).copy()
    grouped = ordered.groupby(["ticker", "session_date"], sort=False)
    ordered["previous_timestamp"] = grouped["timestamp"].shift(1)
    ordered["previous_context"] = grouped["context_key"].shift(1)
    consecutive = (ordered["timestamp"] - ordered["previous_timestamp"]).eq(
        pd.Timedelta(minutes=interval_minutes)
    )
    events = ordered.loc[consecutive & ordered["block_complete"]].copy()
    if events.empty:
        return events
    events["transition_key"] = (
        events["previous_context"].astype(str)
        + "->"
        + events["context_key"].astype(str)
    )
    return events


def _build_block_occupancy(assignments: pd.DataFrame) -> pd.DataFrame:
    complete = assignments.loc[assignments["block_complete"]].copy()
    if complete.empty:
        return _empty(BLOCK_OCCUPANCY_COLUMNS)
    rows: list[dict[str, object]] = []
    for keys, group in complete.groupby(
        ["ticker", "source_status", "block", "session_phase"],
        sort=True,
    ):
        ticker, source_status, block, phase = keys
        entropy, effective = _entropy(group["context_key"])
        rows.append(
            {
                "ticker": ticker,
                "source_status": source_status,
                "block": int(block),
                "session_phase": phase,
                "observations": int(len(group)),
                "sessions": int(group["session_date"].nunique()),
                "context_count": int(group["context_key"].nunique()),
                "entropy_bits": entropy,
                "effective_contexts": effective,
            }
        )
    return pd.DataFrame(rows, columns=BLOCK_OCCUPANCY_COLUMNS).sort_values(
        ["ticker", "block", "session_phase"], kind="mergesort"
    ).reset_index(drop=True)


def _compare_blocks(
    values: pd.DataFrame,
    value_column: str,
    component: str,
    scope_column: str | None,
    minimum_rows: int,
    config: IntradayContextStabilityConfig,
) -> pd.DataFrame:
    if values.empty:
        return _empty(DRIFT_COLUMNS)
    rows: list[dict[str, object]] = []
    for ticker, ticker_group in values.groupby("ticker", sort=True):
        complete_blocks = sorted(
            int(value) for value in ticker_group["block"].unique()
        )
        source_status = str(ticker_group["source_status"].iloc[0])
        scopes = (
            sorted(ticker_group[scope_column].astype(str).unique())
            if scope_column is not None
            else ["ALL_SESSION"]
        )
        for scope in scopes:
            scoped = (
                ticker_group.loc[ticker_group[scope_column].eq(scope)]
                if scope_column is not None
                else ticker_group
            )
            categories = sorted(scoped[value_column].astype(str).unique())
            for reference_block, current_block in zip(
                complete_blocks[:-1], complete_blocks[1:]
            ):
                left_values = scoped.loc[
                    scoped["block"].eq(reference_block), value_column
                ]
                right_values = scoped.loc[
                    scoped["block"].eq(current_block), value_column
                ]
                left = _distribution(left_values, categories)
                right = _distribution(right_values, categories)
                tvd = _total_variation(left, right)
                js = _jensen_shannon_bits(left, right)
                enough = (
                    len(complete_blocks) >= config.minimum_complete_blocks
                    and len(left_values) >= minimum_rows
                    and len(right_values) >= minimum_rows
                )
                if enough:
                    null_tvd, null_js = _calibration_null(
                        left_values,
                        right_values,
                        categories,
                        config,
                        (
                            f"{ticker}|{component}|{scope}|"
                            f"{reference_block}|{current_block}"
                        ),
                    )
                else:
                    null_tvd, null_js = math.nan, math.nan
                tvd_excess = (
                    max(0.0, tvd - null_tvd)
                    if math.isfinite(null_tvd)
                    else math.nan
                )
                js_excess = (
                    max(0.0, js - null_js)
                    if math.isfinite(null_js)
                    else math.nan
                )
                rows.append(
                    {
                        "ticker": ticker,
                        "source_status": source_status,
                        "component": component,
                        "scope": scope,
                        "reference_block": reference_block,
                        "current_block": current_block,
                        "reference_observations": int(len(left_values)),
                        "current_observations": int(len(right_values)),
                        "categories": int(len(categories)),
                        "total_variation": tvd,
                        "jensen_shannon_bits": js,
                        "null_total_variation_quantile": null_tvd,
                        "null_jensen_shannon_quantile": null_js,
                        "total_variation_excess": tvd_excess,
                        "jensen_shannon_excess": js_excess,
                        "shift": _shift_level(
                            len(left_values),
                            len(right_values),
                            len(complete_blocks),
                            minimum_rows,
                            tvd,
                            js,
                            null_tvd,
                            null_js,
                            config,
                        ),
                    }
                )
    return pd.DataFrame(rows, columns=DRIFT_COLUMNS).sort_values(
        ["ticker", "component", "scope", "reference_block"],
        kind="mergesort",
    ).reset_index(drop=True)


def _longest_elevated_run(levels: list[str]) -> int:
    longest = 0
    current = 0
    for level in levels:
        if level in {
            ContextDistributionShift.MODERATE_SHIFT.value,
            ContextDistributionShift.HIGH_SHIFT.value,
        }:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _pattern(levels: list[str]) -> str:
    sufficient = [
        level
        for level in levels
        if level != ContextDistributionShift.INSUFFICIENT.value
    ]
    if not sufficient:
        return "INSUFFICIENT"
    elevated = sum(
        level
        in {
            ContextDistributionShift.MODERATE_SHIFT.value,
            ContextDistributionShift.HIGH_SHIFT.value,
        }
        for level in sufficient
    )
    high = sum(
        level == ContextDistributionShift.HIGH_SHIFT.value
        for level in sufficient
    )
    longest = _longest_elevated_run(sufficient)
    if elevated == 0:
        return "LOW_OR_NONE"
    if high >= 2 and longest >= 2:
        return "PERSISTENT_HIGH"
    if elevated >= 2 and longest >= 2:
        return "PERSISTENT_ELEVATED"
    return "ISOLATED_SHIFT"


def _build_persistence(drift: pd.DataFrame) -> pd.DataFrame:
    if drift.empty:
        return _empty(PERSISTENCE_COLUMNS)
    rows: list[dict[str, object]] = []
    for keys, group in drift.groupby(
        ["ticker", "source_status", "component", "scope"], sort=True
    ):
        ticker, source_status, component, scope = keys
        ordered = group.sort_values("current_block", kind="mergesort")
        levels = ordered["shift"].astype(str).tolist()
        sufficient = ordered.loc[
            ~ordered["shift"].eq(ContextDistributionShift.INSUFFICIENT.value)
        ]
        rows.append(
            {
                "ticker": ticker,
                "source_status": source_status,
                "component": component,
                "scope": scope,
                "block_transitions": int(len(group)),
                "elevated_transitions": int(
                    ordered["shift"].isin(
                        {
                            ContextDistributionShift.MODERATE_SHIFT.value,
                            ContextDistributionShift.HIGH_SHIFT.value,
                        }
                    ).sum()
                ),
                "high_transitions": int(
                    ordered["shift"].eq(
                        ContextDistributionShift.HIGH_SHIFT.value
                    ).sum()
                ),
                "longest_elevated_run": _longest_elevated_run(levels),
                "latest_shift": str(ordered["shift"].iloc[-1]),
                "max_total_variation": (
                    float(sufficient["total_variation"].max())
                    if not sufficient.empty
                    else math.nan
                ),
                "max_jensen_shannon_bits": (
                    float(sufficient["jensen_shannon_bits"].max())
                    if not sufficient.empty
                    else math.nan
                ),
                "max_total_variation_excess": (
                    float(sufficient["total_variation_excess"].max())
                    if not sufficient.empty
                    else math.nan
                ),
                "max_jensen_shannon_excess": (
                    float(sufficient["jensen_shannon_excess"].max())
                    if not sufficient.empty
                    else math.nan
                ),
                "pattern": _pattern(levels),
            }
        )
    return pd.DataFrame(rows, columns=PERSISTENCE_COLUMNS).sort_values(
        ["ticker", "component", "scope"], kind="mergesort"
    ).reset_index(drop=True)


def analyze_intraday_context_stability(
    atlas_report: IntradayRegimeAtlasReport,
    config: IntradayContextStabilityConfig | None = None,
) -> IntradayContextStabilityReport:
    """Confronta soltanto blocchi completi e consecutivi gia' osservati."""

    settings = config or IntradayContextStabilityConfig()
    assignments = atlas_report.assignments.copy()
    if assignments.empty:
        empty_drift = _empty(DRIFT_COLUMNS)
        return IntradayContextStabilityReport(
            block_occupancy=_empty(BLOCK_OCCUPANCY_COLUMNS),
            occupancy_drift=empty_drift.copy(),
            transition_drift=empty_drift.copy(),
            persistence=_empty(PERSISTENCE_COLUMNS),
            caveats=("Nessuna assegnazione valida disponibile.",),
        )
    required = {
        "ticker",
        "source_status",
        "timestamp",
        "session_date",
        "block",
        "block_complete",
        "session_phase",
        "context_key",
    }
    missing = sorted(required.difference(assignments.columns))
    if missing:
        raise ValueError(f"Colonne atlante mancanti: {missing}.")
    assignments["timestamp"] = pd.to_datetime(
        assignments["timestamp"], errors="coerce"
    )
    if assignments["timestamp"].isna().any():
        raise ValueError("Timestamp atlante non validi.")

    complete = assignments.loc[assignments["block_complete"]].copy()
    block_occupancy = _build_block_occupancy(assignments)
    occupancy_drift = _compare_blocks(
        complete,
        "context_key",
        "OCCUPANCY",
        "session_phase",
        settings.minimum_occupancy_rows_per_block,
        settings,
    )
    transition_events = _transition_events(
        assignments,
        settings.interval_minutes,
    )
    transition_drift = _compare_blocks(
        transition_events,
        "transition_key",
        "TRANSITIONS",
        None,
        settings.minimum_transition_rows_per_block,
        settings,
    )
    persistence = _build_persistence(
        pd.concat(
            [occupancy_drift, transition_drift],
            ignore_index=True,
        )
    )
    return IntradayContextStabilityReport(
        block_occupancy=block_occupancy,
        occupancy_drift=occupancy_drift,
        transition_drift=transition_drift,
        persistence=persistence,
        caveats=(
            "TVD e Jensen-Shannon confrontano distribuzioni osservate, non risultati futuri.",
            "Ogni confronto deve superare anche il quantile nullo da "
            "permutazioni deterministiche.",
            "Le transizioni sono ricostruite solo tra barre consecutive della stessa seduta.",
            "I blocchi incompleti sono esclusi e quelli storici non vengono riassegnati.",
            "Le soglie sono descrittive e dichiarate; non approvano alcun modello.",
        ),
    )


def summarize_context_stability(
    report: IntradayContextStabilityReport,
) -> pd.DataFrame:
    """Conta i pattern per asset, mantenendo separata la sorgente limitata."""

    if report.persistence.empty:
        return pd.DataFrame(
            columns=(
                "ticker",
                "source_status",
                "low_or_none",
                "isolated",
                "persistent_elevated",
                "persistent_high",
                "insufficient",
            )
        )
    rows: list[dict[str, object]] = []
    for ticker, group in report.persistence.groupby("ticker", sort=True):
        counts = group["pattern"].value_counts()
        rows.append(
            {
                "ticker": ticker,
                "source_status": str(group["source_status"].iloc[0]),
                "low_or_none": int(counts.get("LOW_OR_NONE", 0)),
                "isolated": int(counts.get("ISOLATED_SHIFT", 0)),
                "persistent_elevated": int(
                    counts.get("PERSISTENT_ELEVATED", 0)
                ),
                "persistent_high": int(counts.get("PERSISTENT_HIGH", 0)),
                "insufficient": int(counts.get("INSUFFICIENT", 0)),
            }
        )
    return pd.DataFrame(rows)


def write_context_stability(
    persistence: pd.DataFrame,
    path: str | Path,
) -> Path:
    destination = Path(path).expanduser()
    if destination.suffix.lower() != ".csv":
        raise ValueError("context_stability_output deve terminare con .csv.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    persistence.to_csv(destination, index=False)
    return destination
