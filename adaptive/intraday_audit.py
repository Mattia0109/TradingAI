"""Audit descrittivo dei dati intraday, senza decisioni operative.

Il modulo verifica se barre da 15 minuti sono abbastanza coerenti per
studiare feature e regimi. Non genera segnali, ordini, size, stop o leva e non
calcola performance di trading.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import time
from enum import Enum
from typing import Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


class IntradayReadiness(str, Enum):
    """Esito del gate dati; non è un'approvazione della strategia."""

    READY = "READY"
    LIMITED = "LIMITED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class IntradayAuditConfig:
    """Profilo iniziale per ETF/azioni USA durante la sessione regolare."""

    interval_minutes: int = 15
    market_timezone: str = "America/New_York"
    regular_session_start: time = time(9, 30)
    regular_session_end: time = time(16, 0)
    minimum_feature_sessions: int = 15
    minimum_validation_sessions: int = 120
    minimum_session_coverage: float = 0.95
    rejection_session_coverage: float = 0.80
    maximum_gap_fraction: float = 0.05
    maximum_zero_volume_fraction: float = 0.10
    maximum_stale_close_fraction: float = 0.50
    maximum_outlier_fraction: float = 0.01
    outlier_modified_zscore: float = 10.0

    def __post_init__(self) -> None:
        if int(self.interval_minutes) <= 0:
            raise ValueError("interval_minutes deve essere positivo.")
        ZoneInfo(self.market_timezone)
        start = self._minutes(self.regular_session_start)
        end = self._minutes(self.regular_session_end)
        if end <= start:
            raise ValueError("La fine sessione deve seguire l'inizio.")
        if (end - start) % int(self.interval_minutes) != 0:
            raise ValueError("La sessione deve essere divisibile per l'intervallo.")
        if int(self.minimum_feature_sessions) <= 0:
            raise ValueError("minimum_feature_sessions deve essere positivo.")
        if self.minimum_validation_sessions < self.minimum_feature_sessions:
            raise ValueError(
                "minimum_validation_sessions deve essere almeno pari al minimo feature."
            )
        for name in (
            "minimum_session_coverage",
            "rejection_session_coverage",
            "maximum_gap_fraction",
            "maximum_zero_volume_fraction",
            "maximum_stale_close_fraction",
            "maximum_outlier_fraction",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} deve essere compreso tra 0 e 1.")
        if self.rejection_session_coverage > self.minimum_session_coverage:
            raise ValueError("Le soglie di coverage non sono ordinate.")
        if self.outlier_modified_zscore <= 0.0:
            raise ValueError("outlier_modified_zscore deve essere positivo.")

    @staticmethod
    def _minutes(value: time) -> int:
        return value.hour * 60 + value.minute

    @property
    def expected_bars_per_session(self) -> int:
        duration = self._minutes(self.regular_session_end) - self._minutes(
            self.regular_session_start
        )
        return duration // int(self.interval_minutes)


@dataclass(frozen=True)
class IntradayAssetAudit:
    ticker: str
    status: IntradayReadiness
    reasons: tuple[str, ...]
    raw_rows: int
    valid_rows: int
    regular_session_rows: int
    extended_hours_rows: int
    weekend_rows: int
    observed_sessions: int
    expected_bars_per_session: int
    median_session_coverage: float
    full_session_fraction: float
    duplicate_rows: int
    invalid_timestamp_rows: int
    invalid_bar_rows: int
    off_grid_rows: int
    estimated_missing_bars: int
    gap_fraction: float
    zero_volume_fraction: float
    stale_close_fraction: float
    outlier_fraction: float
    median_cadence_minutes: float
    history_calendar_days: int
    timezone_assumed: bool
    data_quality_score: float
    first_timestamp: pd.Timestamp | None
    last_timestamp: pd.Timestamp | None
    research_only: bool = True


@dataclass(frozen=True)
class IntradayAuditReport:
    status: IntradayReadiness
    assets: Mapping[str, IntradayAssetAudit]
    errors: Mapping[str, str]
    research_only: bool = True


class IntradayResearchAuditor:
    """Controlla barre intraday rispetto a un profilo di sessione dichiarato."""

    required_columns = ("open", "high", "low", "close", "volume")

    def __init__(self, config: IntradayAuditConfig | None = None) -> None:
        self.config = config or IntradayAuditConfig()

    def _timestamps(
        self,
        values: pd.Series,
    ) -> tuple[pd.Series, bool]:
        parsed = pd.to_datetime(values, errors="coerce")
        try:
            timezone = parsed.dt.tz
        except AttributeError:
            parsed = pd.to_datetime(values, errors="coerce", utc=True)
            timezone = parsed.dt.tz
        if timezone is not None:
            return parsed.dt.tz_convert("UTC"), False
        localized = parsed.dt.tz_localize(
            self.config.market_timezone,
            ambiguous="NaT",
            nonexistent="NaT",
        )
        return localized.dt.tz_convert("UTC"), True

    @staticmethod
    def _robust_outlier_fraction(returns: pd.Series, threshold: float) -> float:
        clean = pd.to_numeric(returns, errors="coerce").dropna()
        if clean.empty:
            return 0.0
        median = float(clean.median())
        absolute_deviation = (clean - median).abs()
        mad = float(absolute_deviation.median())
        if not math.isfinite(mad) or mad <= 0.0:
            return 0.0
        modified_zscore = 0.6745 * (clean - median) / mad
        return float((modified_zscore.abs() > threshold).mean())

    def audit(self, ticker: str, data: pd.DataFrame) -> IntradayAssetAudit:
        normalized_ticker = str(ticker).upper().strip()
        if not normalized_ticker:
            raise ValueError("ticker non può essere vuoto.")
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data deve essere un pandas DataFrame.")
        if data.empty:
            raise ValueError("data non può essere vuoto.")

        frame = data.copy()
        frame.columns = [str(column).lower().strip() for column in frame.columns]
        if "date" not in frame.columns:
            if not isinstance(frame.index, pd.DatetimeIndex):
                raise ValueError("Serve una colonna date o un DatetimeIndex.")
            frame["date"] = frame.index
        missing = [column for column in self.required_columns if column not in frame]
        if missing:
            raise ValueError(f"Colonne OHLCV mancanti: {missing}.")

        raw_rows = len(frame)
        timestamps, timezone_assumed = self._timestamps(frame["date"])
        invalid_timestamp_rows = int(timestamps.isna().sum())
        duplicate_rows = int(timestamps.dropna().duplicated(keep=False).sum())
        frame["date"] = timestamps

        for column in self.required_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        price_columns = ["open", "high", "low", "close"]
        invalid_prices = (
            frame[price_columns].isna().any(axis=1)
            | (frame[price_columns] <= 0.0).any(axis=1)
        )
        incoherent_ohlc = (
            (frame["high"] < frame["low"])
            | (frame["high"] < frame[["open", "close"]].max(axis=1))
            | (frame["low"] > frame[["open", "close"]].min(axis=1))
        )
        invalid_volume = frame["volume"].isna() | (frame["volume"] < 0.0)
        invalid_bar_mask = invalid_prices | incoherent_ohlc | invalid_volume
        invalid_bar_rows = int(invalid_bar_mask.sum())

        valid = frame.loc[frame["date"].notna() & ~invalid_bar_mask].copy()
        valid = valid.sort_values("date").drop_duplicates("date", keep="last")
        if valid.empty:
            raise ValueError("Nessuna barra intraday valida disponibile.")

        market_timezone = ZoneInfo(self.config.market_timezone)
        local_timestamp = valid["date"].dt.tz_convert(market_timezone)
        local_minutes = local_timestamp.dt.hour * 60 + local_timestamp.dt.minute
        session_start = self.config._minutes(self.config.regular_session_start)
        session_end = self.config._minutes(self.config.regular_session_end)
        weekday = local_timestamp.dt.dayofweek < 5
        in_regular_hours = (local_minutes >= session_start) & (
            local_minutes < session_end
        )
        regular_mask = weekday & in_regular_hours
        weekend_rows = int((~weekday).sum())
        extended_hours_rows = int((weekday & ~in_regular_hours).sum())

        regular = valid.loc[regular_mask].copy()
        regular["local_timestamp"] = local_timestamp.loc[regular_mask]
        regular["session_date"] = regular["local_timestamp"].dt.date
        regular["local_minutes"] = local_minutes.loc[regular_mask]
        expected = self.config.expected_bars_per_session

        if regular.empty:
            session_counts = pd.Series(dtype=int)
            median_coverage = 0.0
            full_session_fraction = 0.0
            off_grid_rows = 0
            estimated_missing = 0
            median_cadence = 0.0
            stale_fraction = 0.0
            outlier_fraction = 0.0
            zero_volume_fraction = 0.0
        else:
            session_counts = regular.groupby("session_date").size()
            coverage = (session_counts / expected).clip(upper=1.0)
            median_coverage = float(coverage.median())
            full_session_fraction = float((session_counts >= expected).mean())
            off_grid_rows = int(
                (
                    (regular["local_minutes"] - session_start)
                    % self.config.interval_minutes
                    != 0
                ).sum()
            )

            missing_total = 0
            cadence: list[float] = []
            stale_count = 0
            comparable_count = 0
            intraday_returns: list[pd.Series] = []
            for _, group in regular.groupby("session_date", sort=True):
                ordered = group.sort_values("date")
                grid_count = int(
                    (
                        (ordered["local_minutes"] - session_start)
                        % self.config.interval_minutes
                        == 0
                    ).sum()
                )
                missing_total += max(0, expected - min(expected, grid_count))
                deltas = (
                    ordered["date"].diff().dt.total_seconds().div(60.0).dropna()
                )
                cadence.extend(float(value) for value in deltas if value > 0.0)
                close = ordered["close"].astype(float)
                stale_count += int(close.diff().eq(0.0).sum())
                comparable_count += max(0, len(close) - 1)
                intraday_returns.append(np.log(close / close.shift(1)))

            estimated_missing = missing_total
            median_cadence = float(np.median(cadence)) if cadence else 0.0
            stale_fraction = (
                stale_count / comparable_count if comparable_count else 0.0
            )
            returns = pd.concat(intraday_returns, ignore_index=True)
            outlier_fraction = self._robust_outlier_fraction(
                returns,
                self.config.outlier_modified_zscore,
            )
            zero_volume_fraction = float((regular["volume"] <= 0.0).mean())

        observed_sessions = len(session_counts)
        expected_observations = max(1, observed_sessions * expected)
        gap_fraction = estimated_missing / expected_observations
        history_calendar_days = int(
            (valid["date"].iloc[-1] - valid["date"].iloc[0]).days + 1
        )
        valid_fraction = max(
            0.0,
            1.0 - (invalid_timestamp_rows + invalid_bar_rows) / raw_rows,
        )
        cadence_score = max(
            0.0,
            1.0 - off_grid_rows / max(1, len(regular)) - gap_fraction,
        )
        volume_score = max(0.0, 1.0 - zero_volume_fraction)
        quality_score = float(
            np.clip(
                0.35 * median_coverage
                + 0.25 * valid_fraction
                + 0.20 * cadence_score
                + 0.10 * volume_score
                + 0.10 * max(0.0, 1.0 - stale_fraction),
                0.0,
                1.0,
            )
        )

        rejected: list[str] = []
        limited: list[str] = []
        if regular.empty:
            rejected.append("Nessuna barra nella sessione regolare USA.")
        if duplicate_rows:
            rejected.append(f"Timestamp duplicati: {duplicate_rows} righe.")
        if invalid_timestamp_rows:
            rejected.append(
                f"Timestamp non validi: {invalid_timestamp_rows} righe."
            )
        if invalid_bar_rows:
            rejected.append(f"Barre OHLCV non valide: {invalid_bar_rows} righe.")
        if weekend_rows:
            rejected.append(f"Barre equity nel weekend: {weekend_rows}.")
        if off_grid_rows:
            rejected.append(
                f"Barre fuori dalla griglia {self.config.interval_minutes}m: "
                f"{off_grid_rows}."
            )
        if observed_sessions < self.config.minimum_feature_sessions:
            rejected.append(
                "Sessioni insufficienti anche per l'audit feature: "
                f"{observed_sessions}/{self.config.minimum_feature_sessions}."
            )
        if median_coverage < self.config.rejection_session_coverage:
            rejected.append(
                f"Coverage mediana troppo bassa: {median_coverage:.1%}."
            )
        if gap_fraction > self.config.maximum_gap_fraction:
            rejected.append(f"Troppi gap intraday: {gap_fraction:.1%}.")

        if not rejected:
            if timezone_assumed:
                limited.append("Timezone assente: America/New_York assunta.")
            if observed_sessions < self.config.minimum_validation_sessions:
                limited.append(
                    "Storico insufficiente per una validazione multi-regime: "
                    f"{observed_sessions}/{self.config.minimum_validation_sessions} "
                    "sessioni."
                )
            if median_coverage < self.config.minimum_session_coverage:
                limited.append(
                    f"Coverage mediana sotto il target: {median_coverage:.1%}."
                )
            if estimated_missing:
                limited.append(
                    f"Barre mancanti stimate nelle sessioni osservate: "
                    f"{estimated_missing}."
                )
            if zero_volume_fraction > self.config.maximum_zero_volume_fraction:
                limited.append(
                    f"Volume nullo troppo frequente: {zero_volume_fraction:.1%}."
                )
            if stale_fraction > self.config.maximum_stale_close_fraction:
                limited.append(
                    f"Chiusure stale troppo frequenti: {stale_fraction:.1%}."
                )
            if outlier_fraction > self.config.maximum_outlier_fraction:
                limited.append(
                    f"Outlier robusti sopra soglia: {outlier_fraction:.1%}."
                )

        if rejected:
            status = IntradayReadiness.REJECTED
            reasons = tuple(rejected)
        elif limited:
            status = IntradayReadiness.LIMITED
            reasons = tuple(limited)
        else:
            status = IntradayReadiness.READY
            reasons = (
                "Dati idonei alla ricerca descrittiva delle feature 15m.",
            )

        return IntradayAssetAudit(
            ticker=normalized_ticker,
            status=status,
            reasons=reasons,
            raw_rows=raw_rows,
            valid_rows=len(valid),
            regular_session_rows=len(regular),
            extended_hours_rows=extended_hours_rows,
            weekend_rows=weekend_rows,
            observed_sessions=observed_sessions,
            expected_bars_per_session=expected,
            median_session_coverage=median_coverage,
            full_session_fraction=full_session_fraction,
            duplicate_rows=duplicate_rows,
            invalid_timestamp_rows=invalid_timestamp_rows,
            invalid_bar_rows=invalid_bar_rows,
            off_grid_rows=off_grid_rows,
            estimated_missing_bars=estimated_missing,
            gap_fraction=gap_fraction,
            zero_volume_fraction=zero_volume_fraction,
            stale_close_fraction=stale_fraction,
            outlier_fraction=outlier_fraction,
            median_cadence_minutes=median_cadence,
            history_calendar_days=history_calendar_days,
            timezone_assumed=timezone_assumed,
            data_quality_score=quality_score,
            first_timestamp=valid["date"].iloc[0],
            last_timestamp=valid["date"].iloc[-1],
        )

    def audit_markets(
        self,
        markets: Mapping[str, pd.DataFrame],
    ) -> IntradayAuditReport:
        if not markets:
            raise ValueError("Serve almeno un mercato.")
        assets: dict[str, IntradayAssetAudit] = {}
        errors: dict[str, str] = {}
        for raw_ticker, data in markets.items():
            ticker = str(raw_ticker).upper().strip()
            try:
                assets[ticker] = self.audit(ticker, data)
            except Exception as exc:
                errors[ticker] = f"{type(exc).__name__}: {exc}"

        if not assets:
            status = IntradayReadiness.REJECTED
        elif any(
            audit.status is IntradayReadiness.REJECTED
            for audit in assets.values()
        ):
            status = IntradayReadiness.REJECTED
        elif errors or any(
            audit.status is IntradayReadiness.LIMITED
            for audit in assets.values()
        ):
            status = IntradayReadiness.LIMITED
        else:
            status = IntradayReadiness.READY
        return IntradayAuditReport(status=status, assets=assets, errors=errors)
