"""Audit descrittivo della stabilità delle feature intraday.

Il modulo misura cambiamenti di distribuzione, profili per fase di sessione e
ridondanza fra feature. Non usa outcome futuri e non genera segnali, ordini,
size, stop, leva o P&L.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import time
from enum import Enum
from itertools import combinations
from typing import Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


class DistributionShift(str, Enum):
    """Intensità descrittiva del cambiamento fra due blocchi passati."""

    LOW_SHIFT = "LOW_SHIFT"
    MODERATE_SHIFT = "MODERATE_SHIFT"
    HIGH_SHIFT = "HIGH_SHIFT"


@dataclass(frozen=True)
class IntradayStabilityConfig:
    """Configurazione session-aware per ETF/azioni USA a 15 minuti."""

    market_timezone: str = "America/New_York"
    regular_session_start: time = time(9, 30)
    open_phase_end: time = time(10, 30)
    close_phase_start: time = time(15, 0)
    regular_session_end: time = time(16, 0)
    sessions_per_block: int = 15
    minimum_complete_blocks: int = 3
    minimum_valid_rows_per_block: int = 40
    psi_bins: int = 8
    moderate_psi: float = 0.10
    high_psi: float = 0.25
    moderate_median_shift_iqr: float = 0.75
    high_median_shift_iqr: float = 1.50
    redundancy_threshold: float = 0.85
    numeric_features: tuple[str, ...] = (
        "squeeze_momentum_pct_close",
        "squeeze_momentum_change_pct_close",
        "choppiness",
        "cmf",
    )
    state_features: tuple[str, ...] = (
        "squeeze_state",
        "chop_segment",
    )

    def __post_init__(self) -> None:
        ZoneInfo(self.market_timezone)
        ordered_times = (
            self.regular_session_start,
            self.open_phase_end,
            self.close_phase_start,
            self.regular_session_end,
        )
        minutes = tuple(self._minutes(value) for value in ordered_times)
        if tuple(sorted(minutes)) != minutes or len(set(minutes)) != len(minutes):
            raise ValueError("Le fasi della sessione non sono ordinate.")
        if int(self.sessions_per_block) <= 0:
            raise ValueError("sessions_per_block deve essere positivo.")
        if int(self.minimum_complete_blocks) < 2:
            raise ValueError("Servono almeno due blocchi completi.")
        if int(self.minimum_valid_rows_per_block) <= 0:
            raise ValueError("minimum_valid_rows_per_block deve essere positivo.")
        if int(self.psi_bins) < 3:
            raise ValueError("psi_bins deve essere almeno 3.")
        if not 0.0 < self.moderate_psi < self.high_psi:
            raise ValueError("Le soglie PSI non sono ordinate.")
        if not (
            0.0
            < self.moderate_median_shift_iqr
            < self.high_median_shift_iqr
        ):
            raise ValueError("Le soglie di spostamento mediano non sono ordinate.")
        if not 0.0 < self.redundancy_threshold <= 1.0:
            raise ValueError("redundancy_threshold deve essere in (0, 1].")
        if not self.numeric_features:
            raise ValueError("Serve almeno una feature numerica.")
        if len(set(self.numeric_features)) != len(self.numeric_features):
            raise ValueError("Le feature numeriche devono essere uniche.")
        if len(set(self.state_features)) != len(self.state_features):
            raise ValueError("Le feature di stato devono essere uniche.")

    @staticmethod
    def _minutes(value: time) -> int:
        return value.hour * 60 + value.minute


@dataclass(frozen=True)
class IntradayStabilityReport:
    """Tabelle descrittive; nessun campo è una decisione operativa."""

    ticker: str
    observed_sessions: int
    complete_blocks: int
    excluded_rows: int
    block_summary: pd.DataFrame
    numeric_drift: pd.DataFrame
    numeric_persistence: pd.DataFrame
    state_drift: pd.DataFrame
    state_persistence: pd.DataFrame
    phase_summary: pd.DataFrame
    redundancy: pd.DataFrame
    research_only: bool = True


def _localized_timestamps(
    values: pd.Series | pd.DatetimeIndex,
    market_timezone: str,
) -> pd.DatetimeIndex:
    parsed = pd.DatetimeIndex(pd.to_datetime(values, errors="coerce"))
    if parsed.isna().any():
        raise ValueError("Sono presenti timestamp non validi.")
    if parsed.tz is None:
        parsed = parsed.tz_localize(
            market_timezone,
            ambiguous="NaT",
            nonexistent="NaT",
        )
        if parsed.isna().any():
            raise ValueError("Timestamp locale ambiguo o inesistente.")
    return parsed.tz_convert(market_timezone)


def regular_session_frame(
    data: pd.DataFrame,
    config: IntradayStabilityConfig | None = None,
) -> tuple[pd.DataFrame, int]:
    """Restituisce OHLCV RTH, impedendo contaminazione da extended-hours."""

    settings = config or IntradayStabilityConfig()
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data deve essere un pandas DataFrame.")
    if data.empty:
        raise ValueError("data non può essere vuoto.")
    frame = data.copy()
    frame.columns = [str(column).lower().strip() for column in frame.columns]
    if "date" in frame.columns:
        timestamps = _localized_timestamps(frame["date"], settings.market_timezone)
    elif isinstance(frame.index, pd.DatetimeIndex):
        timestamps = _localized_timestamps(frame.index, settings.market_timezone)
    else:
        raise ValueError("Serve una colonna date o un DatetimeIndex.")
    if timestamps.duplicated().any():
        raise ValueError("Sono presenti timestamp duplicati.")

    minutes = timestamps.hour * 60 + timestamps.minute
    start = settings._minutes(settings.regular_session_start)
    end = settings._minutes(settings.regular_session_end)
    mask = (timestamps.dayofweek < 5) & (minutes >= start) & (minutes < end)
    filtered = frame.loc[np.asarray(mask)].copy()
    if filtered.empty:
        raise ValueError("Nessuna barra nella sessione regolare USA.")
    filtered["date"] = timestamps[mask]
    filtered = filtered.sort_values("date").reset_index(drop=True)
    return filtered, int(len(frame) - len(filtered))


def add_dimensionless_squeeze_features(
    values: pd.DataFrame,
    market: pd.DataFrame,
    config: IntradayStabilityConfig | None = None,
) -> pd.DataFrame:
    """Aggiunge versioni relative al close senza alterare le formule originali."""

    settings = config or IntradayStabilityConfig()
    if not isinstance(values, pd.DataFrame):
        raise TypeError("values deve essere un pandas DataFrame.")
    if not isinstance(values.index, pd.DatetimeIndex):
        raise ValueError("values deve avere un DatetimeIndex.")
    required = {"squeeze_momentum", "squeeze_momentum_change"}
    missing = sorted(required.difference(values.columns))
    if missing:
        raise ValueError(f"Feature Squeeze mancanti: {missing}.")

    regular, _ = regular_session_frame(market, settings)
    close_index = _localized_timestamps(
        regular["date"],
        settings.market_timezone,
    )
    close = pd.Series(
        pd.to_numeric(regular["close"], errors="coerce").to_numpy(dtype=float),
        index=close_index,
        dtype=float,
    )
    if close.isna().any() or (close <= 0.0).any():
        raise ValueError("Il close deve essere numerico e positivo.")
    if close.index.duplicated().any():
        raise ValueError("Timestamp close duplicati.")

    result = values.copy()
    result.index = _localized_timestamps(result.index, settings.market_timezone)
    aligned_close = close.reindex(result.index)
    if aligned_close.isna().any():
        raise ValueError("Impossibile allineare feature e close RTH.")
    result["squeeze_momentum_pct_close"] = (
        pd.to_numeric(result["squeeze_momentum"], errors="coerce")
        / aligned_close
    )
    result["squeeze_momentum_change_pct_close"] = (
        pd.to_numeric(result["squeeze_momentum_change"], errors="coerce")
        / aligned_close
    )
    return result


class IntradayFeatureStabilityAnalyzer:
    """Confronta blocchi storici fissi senza riassegnarli quando arrivano dati."""

    def __init__(self, config: IntradayStabilityConfig | None = None) -> None:
        self.config = config or IntradayStabilityConfig()

    @staticmethod
    def _population_stability_index(
        reference: pd.Series,
        current: pd.Series,
        bins: int,
    ) -> float:
        reference_values = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
        current_values = pd.to_numeric(current, errors="coerce").dropna().to_numpy()
        if reference_values.size == 0 or current_values.size == 0:
            return math.nan
        quantiles = np.linspace(0.0, 1.0, bins + 1)[1:-1]
        inner_edges = np.unique(np.quantile(reference_values, quantiles))
        if inner_edges.size == 0:
            return 0.0 if np.allclose(
                np.median(reference_values),
                np.median(current_values),
            ) else 1.0
        edges = np.concatenate(([-np.inf], inner_edges, [np.inf]))
        reference_counts, _ = np.histogram(reference_values, bins=edges)
        current_counts, _ = np.histogram(current_values, bins=edges)
        epsilon = 1e-6
        reference_share = np.clip(
            reference_counts / reference_counts.sum(), epsilon, None
        )
        current_share = np.clip(
            current_counts / current_counts.sum(), epsilon, None
        )
        return float(
            np.sum(
                (current_share - reference_share)
                * np.log(current_share / reference_share)
            )
        )

    def _shift_level(self, psi: float, median_shift_iqr: float) -> DistributionShift:
        config = self.config
        if (
            psi >= config.high_psi
            or median_shift_iqr >= config.high_median_shift_iqr
        ):
            return DistributionShift.HIGH_SHIFT
        if (
            psi >= config.moderate_psi
            or median_shift_iqr >= config.moderate_median_shift_iqr
        ):
            return DistributionShift.MODERATE_SHIFT
        return DistributionShift.LOW_SHIFT

    @staticmethod
    def _longest_true_run(values: Sequence[bool]) -> int:
        longest = 0
        current = 0
        for value in values:
            current = current + 1 if bool(value) else 0
            longest = max(longest, current)
        return longest

    def _persistence_summary(
        self,
        drift: pd.DataFrame,
        metrics: dict[str, str],
    ) -> pd.DataFrame:
        columns = [
            "ticker",
            "feature",
            "transitions",
            "elevated_transitions",
            "high_transitions",
            "longest_elevated_run",
            "latest_shift",
            "pattern",
            *metrics,
        ]
        if drift.empty:
            return pd.DataFrame(columns=columns)
        rows: list[dict[str, object]] = []
        for (ticker, feature), group in drift.groupby(
            ["ticker", "feature"],
            sort=True,
        ):
            ordered = group.sort_values("current_block")
            valid = ordered.loc[ordered["shift"] != "INSUFFICIENT"]
            levels = valid["shift"].astype(str).tolist()
            elevated = [
                level
                in {
                    DistributionShift.MODERATE_SHIFT.value,
                    DistributionShift.HIGH_SHIFT.value,
                }
                for level in levels
            ]
            high = [
                level == DistributionShift.HIGH_SHIFT.value for level in levels
            ]
            longest_elevated = self._longest_true_run(elevated)
            longest_high = self._longest_true_run(high)
            if not levels:
                pattern = "INSUFFICIENT"
                latest = "INSUFFICIENT"
            elif longest_high >= 2:
                pattern = "PERSISTENT_HIGH"
                latest = levels[-1]
            elif longest_elevated >= 2:
                pattern = "PERSISTENT_ELEVATED"
                latest = levels[-1]
            elif any(elevated):
                pattern = "ISOLATED_SHIFT"
                latest = levels[-1]
            else:
                pattern = "LOW_OR_NONE"
                latest = levels[-1]
            row: dict[str, object] = {
                "ticker": ticker,
                "feature": feature,
                "transitions": len(levels),
                "elevated_transitions": int(sum(elevated)),
                "high_transitions": int(sum(high)),
                "longest_elevated_run": longest_elevated,
                "latest_shift": latest,
                "pattern": pattern,
            }
            for output_name, source_name in metrics.items():
                numeric = pd.to_numeric(valid[source_name], errors="coerce").dropna()
                row[output_name] = (
                    float(numeric.max()) if not numeric.empty else math.nan
                )
            rows.append(row)
        return pd.DataFrame(rows, columns=columns)

    def _prepare_values(self, values: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        if not isinstance(values, pd.DataFrame):
            raise TypeError("values deve essere un pandas DataFrame.")
        if values.empty:
            raise ValueError("values non può essere vuoto.")
        if not isinstance(values.index, pd.DatetimeIndex):
            raise ValueError("values deve avere un DatetimeIndex.")
        required = set(self.config.numeric_features) | set(self.config.state_features)
        missing = sorted(required.difference(values.columns))
        if missing:
            raise ValueError(f"Feature mancanti: {missing}.")

        frame = values.copy()
        timestamps = _localized_timestamps(frame.index, self.config.market_timezone)
        if timestamps.duplicated().any():
            raise ValueError("Sono presenti timestamp feature duplicati.")
        frame.index = timestamps
        frame = frame.sort_index()
        minutes = frame.index.hour * 60 + frame.index.minute
        start = self.config._minutes(self.config.regular_session_start)
        end = self.config._minutes(self.config.regular_session_end)
        regular_mask = (
            (frame.index.dayofweek < 5)
            & (minutes >= start)
            & (minutes < end)
        )
        excluded_rows = int((~regular_mask).sum())
        frame = frame.loc[regular_mask].copy()
        if frame.empty:
            raise ValueError("Nessuna feature nella sessione regolare USA.")

        frame["session_date"] = frame.index.date
        sessions = sorted(frame["session_date"].unique())
        session_ordinals = {
            session: position for position, session in enumerate(sessions)
        }
        frame["block"] = (
            frame["session_date"].map(session_ordinals).astype(int)
            // self.config.sessions_per_block
            + 1
        )
        minutes = frame.index.hour * 60 + frame.index.minute
        open_end = self.config._minutes(self.config.open_phase_end)
        close_start = self.config._minutes(self.config.close_phase_start)
        frame["session_phase"] = np.select(
            [minutes < open_end, minutes >= close_start],
            ["OPEN", "CLOSE"],
            default="MID_SESSION",
        )
        return frame, excluded_rows

    def analyze(self, ticker: str, values: pd.DataFrame) -> IntradayStabilityReport:
        normalized_ticker = str(ticker).upper().strip()
        if not normalized_ticker:
            raise ValueError("ticker non può essere vuoto.")
        frame, excluded_rows = self._prepare_values(values)
        observed_sessions = int(frame["session_date"].nunique())
        block_sessions = frame.groupby("block")["session_date"].nunique()
        complete_block_ids = tuple(
            int(block)
            for block, count in block_sessions.items()
            if int(count) == self.config.sessions_per_block
        )
        complete = frame.loc[frame["block"].isin(complete_block_ids)].copy()

        block_rows: list[dict[str, object]] = []
        for block, group in complete.groupby("block", sort=True):
            for feature in self.config.numeric_features:
                numeric = pd.to_numeric(group[feature], errors="coerce")
                clean = numeric.dropna()
                q25 = float(clean.quantile(0.25)) if not clean.empty else math.nan
                q75 = float(clean.quantile(0.75)) if not clean.empty else math.nan
                block_rows.append(
                    {
                        "ticker": normalized_ticker,
                        "block": int(block),
                        "first_session": min(group["session_date"]),
                        "last_session": max(group["session_date"]),
                        "feature": feature,
                        "rows": int(len(group)),
                        "valid_rows": int(clean.size),
                        "availability": float(clean.size / len(group)),
                        "median": (
                            float(clean.median()) if not clean.empty else math.nan
                        ),
                        "iqr": q75 - q25 if not clean.empty else math.nan,
                    }
                )
        block_summary = pd.DataFrame(block_rows)

        drift_rows: list[dict[str, object]] = []
        for previous_block, current_block in zip(
            complete_block_ids[:-1], complete_block_ids[1:]
        ):
            previous = complete.loc[complete["block"] == previous_block]
            current = complete.loc[complete["block"] == current_block]
            for feature in self.config.numeric_features:
                reference = pd.to_numeric(previous[feature], errors="coerce").dropna()
                observed = pd.to_numeric(current[feature], errors="coerce").dropna()
                enough = (
                    len(reference) >= self.config.minimum_valid_rows_per_block
                    and len(observed) >= self.config.minimum_valid_rows_per_block
                )
                if not enough:
                    psi = math.nan
                    shift = math.nan
                    level: str = "INSUFFICIENT"
                else:
                    psi = self._population_stability_index(
                        reference,
                        observed,
                        self.config.psi_bins,
                    )
                    q25 = float(reference.quantile(0.25))
                    q75 = float(reference.quantile(0.75))
                    scale = max(
                        q75 - q25,
                        abs(float(reference.median())) * 1e-6,
                        1e-12,
                    )
                    shift = (
                        abs(
                            float(observed.median())
                            - float(reference.median())
                        )
                        / scale
                    )
                    level = self._shift_level(psi, shift).value
                drift_rows.append(
                    {
                        "ticker": normalized_ticker,
                        "feature": feature,
                        "reference_block": previous_block,
                        "current_block": current_block,
                        "reference_rows": int(len(reference)),
                        "current_rows": int(len(observed)),
                        "psi": psi,
                        "median_shift_iqr": shift,
                        "shift": level,
                    }
                )
        numeric_drift = pd.DataFrame(drift_rows)
        numeric_persistence = self._persistence_summary(
            numeric_drift,
            {
                "max_psi": "psi",
                "max_median_shift_iqr": "median_shift_iqr",
            },
        )

        state_rows: list[dict[str, object]] = []
        for previous_block, current_block in zip(
            complete_block_ids[:-1], complete_block_ids[1:]
        ):
            previous = complete.loc[complete["block"] == previous_block]
            current = complete.loc[complete["block"] == current_block]
            for feature in self.config.state_features:
                left = previous[feature].dropna().astype(str)
                right = current[feature].dropna().astype(str)
                left = left.loc[left != "INSUFFICIENT"]
                right = right.loc[right != "INSUFFICIENT"]
                categories = sorted(set(left) | set(right))
                if not categories or left.empty or right.empty:
                    tvd = math.nan
                    level = "INSUFFICIENT"
                else:
                    left_share = left.value_counts(normalize=True).reindex(
                        categories,
                        fill_value=0.0,
                    )
                    right_share = right.value_counts(normalize=True).reindex(
                        categories,
                        fill_value=0.0,
                    )
                    tvd = float(0.5 * (left_share - right_share).abs().sum())
                    level = (
                        DistributionShift.HIGH_SHIFT.value
                        if tvd >= self.config.high_psi
                        else DistributionShift.MODERATE_SHIFT.value
                        if tvd >= self.config.moderate_psi
                        else DistributionShift.LOW_SHIFT.value
                    )
                state_rows.append(
                    {
                        "ticker": normalized_ticker,
                        "feature": feature,
                        "reference_block": previous_block,
                        "current_block": current_block,
                        "total_variation": tvd,
                        "shift": level,
                    }
                )
        state_drift = pd.DataFrame(state_rows)
        state_persistence = self._persistence_summary(
            state_drift,
            {"max_total_variation": "total_variation"},
        )

        phase_rows: list[dict[str, object]] = []
        for phase, group in complete.groupby("session_phase", sort=True):
            for feature in self.config.numeric_features:
                clean = pd.to_numeric(group[feature], errors="coerce").dropna()
                q25 = float(clean.quantile(0.25)) if not clean.empty else math.nan
                q75 = float(clean.quantile(0.75)) if not clean.empty else math.nan
                phase_rows.append(
                    {
                        "ticker": normalized_ticker,
                        "phase": phase,
                        "feature": feature,
                        "rows": int(len(group)),
                        "valid_rows": int(len(clean)),
                        "availability": float(len(clean) / len(group)),
                        "median": (
                            float(clean.median()) if not clean.empty else math.nan
                        ),
                        "iqr": q75 - q25 if not clean.empty else math.nan,
                    }
                )
        phase_summary = pd.DataFrame(phase_rows)

        redundancy_rows: list[dict[str, object]] = []
        numeric_frame = complete.loc[:, self.config.numeric_features].apply(
            pd.to_numeric,
            errors="coerce",
        )
        correlations = numeric_frame.corr(method="spearman", min_periods=20)
        for left, right in combinations(self.config.numeric_features, 2):
            correlation = (
                correlations.loc[left, right]
                if not correlations.empty
                else math.nan
            )
            absolute = abs(float(correlation)) if pd.notna(correlation) else math.nan
            redundancy_rows.append(
                {
                    "ticker": normalized_ticker,
                    "left_feature": left,
                    "right_feature": right,
                    "spearman": (
                        float(correlation) if pd.notna(correlation) else math.nan
                    ),
                    "absolute_correlation": absolute,
                    "relation": (
                        "HIGH_REDUNDANCY"
                        if pd.notna(absolute)
                        and absolute >= self.config.redundancy_threshold
                        else "DISTINCT_OR_UNRESOLVED"
                    ),
                }
            )
        redundancy = pd.DataFrame(redundancy_rows)

        return IntradayStabilityReport(
            ticker=normalized_ticker,
            observed_sessions=observed_sessions,
            complete_blocks=len(complete_block_ids),
            excluded_rows=excluded_rows,
            block_summary=block_summary,
            numeric_drift=numeric_drift,
            numeric_persistence=numeric_persistence,
            state_drift=state_drift,
            state_persistence=state_persistence,
            phase_summary=phase_summary,
            redundancy=redundancy,
        )
