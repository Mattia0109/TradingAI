"""Sorgente locale verificata per dati intraday FirstRate Data.

Il modulo legge archivi ZIP 1m, valida struttura e OHLC(V), limita i dati alla
sessione regolare USA e costruisce barre 15m soltanto da bucket completi. Non
calcola segnali, outcome futuri, ordini, size, leva o P&L.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from datetime import time
from enum import Enum
from io import TextIOWrapper
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


class LocalSourceReadiness(str, Enum):
    """Idoneita' della sorgente a report descrittivi, non a operativita'."""

    READY_DESCRIPTIVE = "READY_DESCRIPTIVE"
    LIMITED = "LIMITED"
    PRICE_ONLY = "PRICE_ONLY"


@dataclass(frozen=True)
class LocalIntradayDataConfig:
    """Contratto esplicito per campioni USA a frequenza un minuto."""

    market_timezone: str = "America/New_York"
    regular_session_start: time = time(9, 30)
    regular_session_end: time = time(16, 0)
    early_close_end: time = time(13, 0)
    source_interval_minutes: int = 1
    output_interval_minutes: int = 15
    # Date NYSE di chiusura alle 13:00 comprese nel periodo preconfigurato.
    # 2022-09-30 -> 2023-09-29. Per campioni differenti il chiamante deve
    # dichiarare il calendario applicabile invece di inferirlo dai prezzi.
    early_close_dates: tuple[str, ...] = ("2022-11-25", "2023-07-03")
    minimum_descriptive_sessions: int = 120
    minimum_median_minute_coverage: float = 0.95
    minimum_complete_bucket_fraction: float = 0.95
    maximum_member_bytes: int = 250_000_000

    def __post_init__(self) -> None:
        ZoneInfo(self.market_timezone)
        if int(self.source_interval_minutes) != 1:
            raise ValueError("La sorgente V1 richiede barre da un minuto.")
        if int(self.output_interval_minutes) <= 1:
            raise ValueError("L'intervallo di uscita deve superare un minuto.")
        if self.output_interval_minutes % self.source_interval_minutes:
            raise ValueError("Gli intervalli non sono divisibili.")
        start = self._minutes(self.regular_session_start)
        early = self._minutes(self.early_close_end)
        end = self._minutes(self.regular_session_end)
        if not start < early < end:
            raise ValueError("Gli orari di sessione non sono ordinati.")
        for name in (
            "minimum_median_minute_coverage",
            "minimum_complete_bucket_fraction",
        ):
            value = float(getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} deve essere in (0, 1].")
        if int(self.minimum_descriptive_sessions) <= 0:
            raise ValueError("minimum_descriptive_sessions deve essere positivo.")
        if int(self.maximum_member_bytes) <= 0:
            raise ValueError("maximum_member_bytes deve essere positivo.")
        parsed_early_closes = pd.to_datetime(
            list(self.early_close_dates),
            errors="coerce",
        )
        if parsed_early_closes.isna().any():
            raise ValueError("early_close_dates contiene date non valide.")

    @staticmethod
    def _minutes(value: time) -> int:
        return value.hour * 60 + value.minute

    @property
    def regular_minutes(self) -> int:
        return self._minutes(self.regular_session_end) - self._minutes(
            self.regular_session_start
        )

    @property
    def early_close_minutes(self) -> int:
        return self._minutes(self.early_close_end) - self._minutes(
            self.regular_session_start
        )

    @property
    def early_close_date_set(self) -> frozenset[str]:
        return frozenset(
            pd.Timestamp(value).date().isoformat()
            for value in self.early_close_dates
        )


@dataclass(frozen=True)
class LocalIntradaySourceAudit:
    ticker: str
    source_path: str
    raw_rows: int
    regular_session_rows: int
    observed_sessions: int
    has_volume: bool
    median_minute_coverage: float
    minimum_minute_coverage: float
    complete_bucket_fraction: float
    early_close_sessions: int
    estimated_missing_minutes: int
    excluded_rows: int
    output_bars: int
    status: LocalSourceReadiness
    reasons: tuple[str, ...]
    first_timestamp: pd.Timestamp
    last_timestamp: pd.Timestamp
    research_only: bool = True


@dataclass(frozen=True)
class LocalIntradayLoadResult:
    ticker: str
    bars: pd.DataFrame
    audit: LocalIntradaySourceAudit
    research_only: bool = True


def ticker_from_path(path: str | Path) -> str:
    """Estrae il simbolo senza affidarsi al contenuto dell'archivio."""

    name = Path(path).name
    match = re.match(r"^([A-Za-z0-9._-]+?)_1min(?:_|\.)", name)
    if not match:
        raise ValueError(
            "Nome archivio non riconosciuto; atteso TICKER_1min_*.zip."
        )
    return match.group(1).upper()


def discover_first_rate_archives(
    directory: str | Path,
    tickers: Iterable[str] | None = None,
) -> Mapping[str, Path]:
    """Trova in modo deterministico un solo archivio per ticker."""

    root = Path(directory).expanduser()
    if not root.is_dir():
        raise ValueError(f"Directory dati inesistente: {root}")
    requested = None
    if tickers is not None:
        requested = {
            str(ticker).upper().strip()
            for ticker in tickers
            if str(ticker).strip()
        }
    found: dict[str, Path] = {}
    for path in sorted(root.glob("*.zip")):
        try:
            ticker = ticker_from_path(path)
        except ValueError:
            continue
        if requested is not None and ticker not in requested:
            continue
        if ticker in found:
            raise ValueError(f"Archivio duplicato per {ticker}.")
        found[ticker] = path
    if requested is not None:
        missing = sorted(requested.difference(found))
        if missing:
            raise ValueError(f"Archivi mancanti per: {', '.join(missing)}.")
    if not found:
        raise ValueError("Nessun archivio FirstRate Data trovato.")
    return found


class FirstRateIntradayLoader:
    """Carica un archivio alla volta e conserva soltanto barre 15m validate."""

    price_columns = ("open", "high", "low", "close")

    def __init__(self, config: LocalIntradayDataConfig | None = None) -> None:
        self.config = config or LocalIntradayDataConfig()

    def _read_archive(self, path: Path) -> pd.DataFrame:
        try:
            archive = zipfile.ZipFile(path)
        except (OSError, zipfile.BadZipFile) as exc:
            raise ValueError(f"Archivio ZIP non valido: {path.name}.") from exc
        with archive:
            members = [item for item in archive.infolist() if not item.is_dir()]
            if len(members) != 1:
                raise ValueError("L'archivio deve contenere esattamente un file.")
            member = members[0]
            member_path = PurePosixPath(member.filename.replace("\\", "/"))
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError("Percorso interno ZIP non sicuro.")
            if member.flag_bits & 0x1:
                raise ValueError("Archivi ZIP cifrati non supportati.")
            if member.file_size > self.config.maximum_member_bytes:
                raise ValueError("File interno oltre il limite dichiarato.")
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f"CRC ZIP non valido: {bad_member}.")

            with archive.open(member) as raw:
                preview = raw.readline().decode("utf-8-sig", errors="strict")
            delimiter = "," if "," in preview else None
            preview_fields = (
                [part.strip() for part in preview.split(",")]
                if delimiter
                else preview.split()
            )
            first_field = preview_fields[0].lower() if preview_fields else ""
            has_header = first_field in {"timestamp", "date", "datetime"}
            with archive.open(member) as raw:
                text = TextIOWrapper(raw, encoding="utf-8-sig", errors="strict")
                if has_header:
                    frame = pd.read_csv(text, sep=delimiter)
                else:
                    frame = pd.read_csv(text, sep=delimiter, header=None)

        if not has_header:
            if frame.shape[1] == 6:
                frame.columns = [
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]
            elif frame.shape[1] == 5:
                frame.columns = ["timestamp", "open", "high", "low", "close"]
            else:
                raise ValueError(
                    "Schema senza header non riconosciuto: servono 5 o 6 colonne."
                )
        return frame

    def _normalize(self, raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        frame = raw.copy()
        frame.columns = [str(column).lower().strip() for column in frame.columns]
        aliases = {
            "timestamp": "date",
            "datetime": "date",
            "adj close": "adj_close",
        }
        frame = frame.rename(columns=aliases)
        missing = sorted(set(("date", *self.price_columns)).difference(frame))
        if missing:
            raise ValueError(f"Colonne prezzo mancanti: {missing}.")

        parsed = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="coerce"))
        if parsed.isna().any():
            raise ValueError(f"Timestamp non validi: {int(parsed.isna().sum())}.")
        if parsed.tz is None:
            parsed = parsed.tz_localize(
                self.config.market_timezone,
                ambiguous="raise",
                nonexistent="raise",
            )
        else:
            parsed = parsed.tz_convert(self.config.market_timezone)
        if parsed.duplicated().any():
            raise ValueError(
                f"Timestamp duplicati: {int(parsed.duplicated(keep=False).sum())}."
            )
        if ((parsed.second != 0) | (parsed.microsecond != 0)).any():
            raise ValueError("Timestamp fuori dalla griglia di un minuto.")
        frame["date"] = parsed

        numeric_columns = [*self.price_columns]
        if "volume" in frame:
            numeric_columns.append("volume")
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[numeric_columns].isna().any().any():
            raise ValueError("Valori OHLC(V) non numerici.")
        if (frame[list(self.price_columns)] <= 0.0).any().any():
            raise ValueError("I prezzi devono essere positivi.")
        if "volume" in frame and (frame["volume"] < 0.0).any():
            raise ValueError("Il volume non puo' essere negativo.")
        invalid_ohlc = (
            (frame["high"] < frame["low"])
            | (frame["high"] < frame[["open", "close"]].max(axis=1))
            | (frame["low"] > frame[["open", "close"]].min(axis=1))
        )
        if invalid_ohlc.any():
            raise ValueError(f"Relazioni OHLC non valide: {int(invalid_ohlc.sum())}.")

        minutes = parsed.hour * 60 + parsed.minute
        start = self.config._minutes(self.config.regular_session_start)
        end = self.config._minutes(self.config.regular_session_end)
        early_end = self.config._minutes(self.config.early_close_end)
        session_dates = np.asarray(
            [value.isoformat() for value in parsed.date],
            dtype=object,
        )
        declared_early_close = np.isin(
            session_dates,
            list(self.config.early_close_date_set),
        )
        row_session_end = np.where(declared_early_close, early_end, end)
        regular_mask = (
            (parsed.dayofweek < 5)
            & (minutes >= start)
            & (minutes < row_session_end)
        )
        excluded = int((~regular_mask).sum())
        regular = frame.loc[np.asarray(regular_mask)].copy()
        if regular.empty:
            raise ValueError("Nessuna riga nella sessione regolare USA.")
        regular = regular.sort_values("date").reset_index(drop=True)
        regular["session_date"] = regular["date"].dt.date
        regular["minute_of_day"] = (
            regular["date"].dt.hour * 60 + regular["date"].dt.minute
        )
        return regular, excluded

    def _session_expected_minutes(self, group: pd.DataFrame) -> tuple[int, bool]:
        """Usa il calendario dichiarato, senza inferirlo dall'andamento dei dati."""

        session_date = str(group["session_date"].iloc[0])
        is_declared_early_close = session_date in self.config.early_close_date_set
        if is_declared_early_close:
            return self.config.early_close_minutes, True
        return self.config.regular_minutes, False

    def _resample(self, regular: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        start = self.config._minutes(self.config.regular_session_start)
        interval = int(self.config.output_interval_minutes)
        output_rows: list[dict] = []
        coverages: list[float] = []
        expected_buckets = 0
        complete_buckets = 0
        missing_minutes = 0
        early_close_sessions = 0

        for session_date, group in regular.groupby("session_date", sort=True):
            ordered = group.sort_values("date").copy()
            expected_minutes, early_close = self._session_expected_minutes(ordered)
            early_close_sessions += int(early_close)
            expected_grid = set(range(start, start + expected_minutes))
            observed_grid = set(ordered["minute_of_day"].astype(int))
            valid_observed = observed_grid.intersection(expected_grid)
            missing = max(0, expected_minutes - len(valid_observed))
            missing_minutes += missing
            coverages.append(len(valid_observed) / expected_minutes)
            session_buckets = expected_minutes // interval
            expected_buckets += session_buckets

            ordered["bucket"] = (
                (ordered["minute_of_day"] - start) // interval
            ).astype(int)
            for bucket, bucket_frame in ordered.groupby("bucket", sort=True):
                bucket = int(bucket)
                if not 0 <= bucket < session_buckets:
                    continue
                bucket_start_minute = start + bucket * interval
                expected_bucket_grid = set(
                    range(bucket_start_minute, bucket_start_minute + interval)
                )
                observed_bucket_grid = set(
                    bucket_frame["minute_of_day"].astype(int)
                )
                if observed_bucket_grid != expected_bucket_grid:
                    continue
                complete_buckets += 1
                ordered_bucket = bucket_frame.sort_values("date")
                local_start = pd.Timestamp(session_date).tz_localize(
                    self.config.market_timezone
                ) + pd.Timedelta(minutes=bucket_start_minute)
                row = {
                    "date": local_start,
                    "open": float(ordered_bucket["open"].iloc[0]),
                    "high": float(ordered_bucket["high"].max()),
                    "low": float(ordered_bucket["low"].min()),
                    "close": float(ordered_bucket["close"].iloc[-1]),
                }
                if "volume" in ordered_bucket:
                    row["volume"] = float(ordered_bucket["volume"].sum())
                output_rows.append(row)

        if not output_rows:
            raise ValueError("Nessun bucket 15m completo disponibile.")
        bars = pd.DataFrame(output_rows).sort_values("date").reset_index(drop=True)
        if bars["date"].duplicated().any():
            raise RuntimeError("Il resampling ha prodotto timestamp duplicati.")
        metadata = {
            "observed_sessions": int(regular["session_date"].nunique()),
            "median_coverage": float(np.median(coverages)),
            "minimum_coverage": float(np.min(coverages)),
            "complete_bucket_fraction": (
                complete_buckets / expected_buckets if expected_buckets else 0.0
            ),
            "early_close_sessions": early_close_sessions,
            "missing_minutes": missing_minutes,
        }
        return bars, metadata

    def load(self, path: str | Path, ticker: str | None = None) -> LocalIntradayLoadResult:
        archive_path = Path(path).expanduser()
        if not archive_path.is_file():
            raise ValueError(f"Archivio inesistente: {archive_path}")
        normalized_ticker = (
            str(ticker).upper().strip() if ticker is not None else ticker_from_path(path)
        )
        if not normalized_ticker:
            raise ValueError("ticker non puo' essere vuoto.")

        raw = self._read_archive(archive_path)
        raw_rows = len(raw)
        regular, excluded = self._normalize(raw)
        bars, metadata = self._resample(regular)
        has_volume = "volume" in bars
        reasons: list[str] = []
        if not has_volume:
            status = LocalSourceReadiness.PRICE_ONLY
            reasons.append("Volume assente: CMF e report a quattro feature esclusi.")
        else:
            status = LocalSourceReadiness.READY_DESCRIPTIVE
        if metadata["observed_sessions"] < self.config.minimum_descriptive_sessions:
            status = LocalSourceReadiness.LIMITED
            reasons.append(
                "Sessioni sotto il minimo descrittivo: "
                f"{metadata['observed_sessions']}/"
                f"{self.config.minimum_descriptive_sessions}."
            )
        if (
            metadata["median_coverage"]
            < self.config.minimum_median_minute_coverage
        ):
            status = LocalSourceReadiness.LIMITED
            reasons.append(
                "Coverage mediana 1m sotto soglia: "
                f"{metadata['median_coverage']:.1%}."
            )
        if (
            metadata["complete_bucket_fraction"]
            < self.config.minimum_complete_bucket_fraction
        ):
            status = LocalSourceReadiness.LIMITED
            reasons.append(
                "Bucket 15m completi sotto soglia: "
                f"{metadata['complete_bucket_fraction']:.1%}."
            )
        if not reasons:
            reasons.append(
                "Dati idonei a report descrittivi 15m; nessuna approvazione operativa."
            )

        audit = LocalIntradaySourceAudit(
            ticker=normalized_ticker,
            source_path=str(archive_path),
            raw_rows=raw_rows,
            regular_session_rows=len(regular),
            observed_sessions=metadata["observed_sessions"],
            has_volume=has_volume,
            median_minute_coverage=metadata["median_coverage"],
            minimum_minute_coverage=metadata["minimum_coverage"],
            complete_bucket_fraction=metadata["complete_bucket_fraction"],
            early_close_sessions=metadata["early_close_sessions"],
            estimated_missing_minutes=metadata["missing_minutes"],
            excluded_rows=excluded,
            output_bars=len(bars),
            status=status,
            reasons=tuple(reasons),
            first_timestamp=bars["date"].iloc[0],
            last_timestamp=bars["date"].iloc[-1],
        )
        return LocalIntradayLoadResult(
            ticker=normalized_ticker,
            bars=bars,
            audit=audit,
        )


def load_local_intraday_markets(
    directory: str | Path,
    tickers: Iterable[str] | None = None,
    config: LocalIntradayDataConfig | None = None,
) -> tuple[
    Mapping[str, pd.DataFrame],
    Mapping[str, LocalIntradaySourceAudit],
    Mapping[str, str],
]:
    """Carica asset in isolamento; un archivio difettoso non ferma gli altri."""

    paths = discover_first_rate_archives(directory, tickers)
    loader = FirstRateIntradayLoader(config)
    markets: dict[str, pd.DataFrame] = {}
    audits: dict[str, LocalIntradaySourceAudit] = {}
    errors: dict[str, str] = {}
    for ticker, path in paths.items():
        try:
            result = loader.load(path, ticker)
            markets[ticker] = result.bars
            audits[ticker] = result.audit
        except Exception as exc:
            errors[ticker] = f"{type(exc).__name__}: {exc}"
    return markets, audits, errors
