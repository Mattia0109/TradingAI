"""Audit descrittivo per tick Kibot ``Date,Time,Price,Bid,Ask,Size``.

Ogni riga rappresenta un'esecuzione. Record identici nello stesso secondo non
vengono deduplicati, perche' possono essere transazioni distinte. Il modulo
costruisce barre OHLCV 15m e statistiche di qualita' del flusso; non produce
segnali, ordini, size operative, outcome futuri o P&L.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from enum import Enum
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


class KibotTickReadiness(str, Enum):
    READY_DESCRIPTIVE = "READY_DESCRIPTIVE"
    LIMITED = "LIMITED"


@dataclass(frozen=True)
class KibotTickConfig:
    market_timezone: str = "America/New_York"
    regular_session_start: time = time(9, 30)
    regular_session_end: time = time(16, 0)
    output_interval_minutes: int = 15
    minimum_descriptive_sessions: int = 120
    minimum_bucket_coverage: float = 0.95

    def __post_init__(self) -> None:
        ZoneInfo(self.market_timezone)
        if int(self.output_interval_minutes) <= 0:
            raise ValueError("output_interval_minutes deve essere positivo.")
        start = self._minutes(self.regular_session_start)
        end = self._minutes(self.regular_session_end)
        if end <= start or (end - start) % self.output_interval_minutes:
            raise ValueError("Sessione e intervallo non compatibili.")
        if int(self.minimum_descriptive_sessions) <= 0:
            raise ValueError("minimum_descriptive_sessions deve essere positivo.")
        if not 0.0 < float(self.minimum_bucket_coverage) <= 1.0:
            raise ValueError("minimum_bucket_coverage deve essere in (0, 1].")

    @staticmethod
    def _minutes(value: time) -> int:
        return value.hour * 60 + value.minute

    @property
    def expected_bars_per_session(self) -> int:
        duration = self._minutes(self.regular_session_end) - self._minutes(
            self.regular_session_start
        )
        return duration // int(self.output_interval_minutes)


@dataclass(frozen=True)
class KibotTickAudit:
    ticker: str
    source_path: str
    raw_rows: int
    regular_ticks: int
    excluded_ticks: int
    observed_sessions: int
    output_bars: int
    bucket_coverage: float
    duplicate_timestamp_rows: int
    exact_duplicate_rows: int
    crossed_quote_fraction: float
    locked_quote_fraction: float
    outside_nbbo_fraction: float
    median_spread_bps: float
    spread_p90_bps: float
    median_ticks_per_bar: float
    minimum_ticks_per_bar: int
    status: KibotTickReadiness
    reasons: tuple[str, ...]
    first_timestamp: pd.Timestamp
    last_timestamp: pd.Timestamp
    research_only: bool = True


@dataclass(frozen=True)
class KibotTickResult:
    ticker: str
    bars: pd.DataFrame
    audit: KibotTickAudit
    research_only: bool = True


class KibotTickLoader:
    """Valida tick con NBBO e li aggrega senza inventare osservazioni."""

    columns = ("date", "time", "price", "bid", "ask", "size")

    def __init__(self, config: KibotTickConfig | None = None) -> None:
        self.config = config or KibotTickConfig()

    def _read(self, path: Path) -> pd.DataFrame:
        with path.open("r", encoding="utf-8", errors="strict") as handle:
            preview = handle.readline().strip()
        if not preview:
            raise ValueError("File tick vuoto.")
        first = [part.strip().lower() for part in preview.split(",")]
        if len(first) != len(self.columns):
            raise ValueError(
                "Schema Kibot non valido; atteso Date,Time,Price,Bid,Ask,Size."
            )
        has_header = first == list(self.columns)
        frame = pd.read_csv(
            path,
            header=0 if has_header else None,
            names=None if has_header else list(self.columns),
        )
        frame.columns = [str(column).lower().strip() for column in frame.columns]
        if tuple(frame.columns) != self.columns:
            raise ValueError(
                "Schema Kibot non valido; atteso Date,Time,Price,Bid,Ask,Size."
            )
        return frame

    def load(
        self,
        path: str | Path,
        ticker: str = "IVE",
    ) -> KibotTickResult:
        source = Path(path).expanduser()
        if not source.is_file():
            raise ValueError(f"File tick inesistente: {source}")
        normalized_ticker = str(ticker).upper().strip()
        if not normalized_ticker:
            raise ValueError("ticker non puo' essere vuoto.")

        raw = self._read(source)
        raw_rows = len(raw)
        exact_duplicate_rows = int(raw.duplicated(keep=False).sum())
        timestamps = pd.DatetimeIndex(
            pd.to_datetime(
                raw["date"].astype(str) + " " + raw["time"].astype(str),
                format="%m/%d/%Y %H:%M:%S",
                errors="coerce",
            )
        )
        if timestamps.isna().any():
            raise ValueError(f"Timestamp non validi: {int(timestamps.isna().sum())}.")
        timestamps = timestamps.tz_localize(
            self.config.market_timezone,
            ambiguous="raise",
            nonexistent="raise",
        )
        duplicate_timestamp_rows = int(timestamps.duplicated(keep=False).sum())

        frame = raw.copy()
        frame["date"] = timestamps
        frame["sequence"] = np.arange(len(frame), dtype=int)
        for column in ("price", "bid", "ask", "size"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[["price", "bid", "ask", "size"]].isna().any().any():
            raise ValueError("Valori Price/Bid/Ask/Size non numerici.")
        if (frame[["price", "bid", "ask"]] <= 0.0).any().any():
            raise ValueError("Price/Bid/Ask devono essere positivi.")
        if (frame["size"] <= 0.0).any():
            raise ValueError("Size deve essere positivo.")

        minutes = timestamps.hour * 60 + timestamps.minute
        start = self.config._minutes(self.config.regular_session_start)
        end = self.config._minutes(self.config.regular_session_end)
        regular_mask = (
            (timestamps.dayofweek < 5)
            & (minutes >= start)
            & (minutes < end)
        )
        regular = frame.loc[np.asarray(regular_mask)].copy()
        excluded_ticks = int((~regular_mask).sum())
        if regular.empty:
            raise ValueError("Nessun tick nella sessione regolare USA.")
        regular = regular.sort_values(
            ["date", "sequence"],
            kind="mergesort",
        ).reset_index(drop=True)
        regular["session_date"] = regular["date"].dt.date
        regular["minute_of_day"] = (
            regular["date"].dt.hour * 60 + regular["date"].dt.minute
        )
        interval = int(self.config.output_interval_minutes)
        regular["bucket"] = ((regular["minute_of_day"] - start) // interval).astype(
            int
        )
        midpoint = (regular["bid"] + regular["ask"]) / 2.0
        regular["spread_bps"] = (
            (regular["ask"] - regular["bid"]) / midpoint * 10_000.0
        )
        regular["outside_nbbo"] = (
            (regular["price"] < regular["bid"])
            | (regular["price"] > regular["ask"])
        )

        rows: list[dict[str, float | int | pd.Timestamp]] = []
        for (session_date, bucket), group in regular.groupby(
            ["session_date", "bucket"],
            sort=True,
        ):
            ordered = group.sort_values(["date", "sequence"], kind="mergesort")
            bucket_start_minute = start + int(bucket) * interval
            local_start = pd.Timestamp(session_date).tz_localize(
                self.config.market_timezone
            ) + pd.Timedelta(minutes=bucket_start_minute)
            rows.append(
                {
                    "date": local_start,
                    "open": float(ordered["price"].iloc[0]),
                    "high": float(ordered["price"].max()),
                    "low": float(ordered["price"].min()),
                    "close": float(ordered["price"].iloc[-1]),
                    "volume": float(ordered["size"].sum()),
                    "tick_count": int(len(ordered)),
                    "median_spread_bps": float(ordered["spread_bps"].median()),
                    "spread_p90_bps": float(ordered["spread_bps"].quantile(0.90)),
                    "outside_nbbo_fraction": float(ordered["outside_nbbo"].mean()),
                }
            )
        bars = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        observed_sessions = int(regular["session_date"].nunique())
        expected_bars = observed_sessions * self.config.expected_bars_per_session
        bucket_coverage = len(bars) / max(1, expected_bars)

        crossed = regular["ask"] < regular["bid"]
        locked = regular["ask"] == regular["bid"]
        reasons: list[str] = []
        status = KibotTickReadiness.READY_DESCRIPTIVE
        if observed_sessions < self.config.minimum_descriptive_sessions:
            status = KibotTickReadiness.LIMITED
            reasons.append(
                "Sessioni sotto il minimo descrittivo: "
                f"{observed_sessions}/{self.config.minimum_descriptive_sessions}."
            )
        if bucket_coverage < self.config.minimum_bucket_coverage:
            status = KibotTickReadiness.LIMITED
            reasons.append(f"Coverage bucket 15m sotto soglia: {bucket_coverage:.1%}.")
        if not reasons:
            reasons.append(
                "Flusso idoneo a descrizioni tick/15m; nessuna approvazione operativa."
            )

        audit = KibotTickAudit(
            ticker=normalized_ticker,
            source_path=str(source),
            raw_rows=raw_rows,
            regular_ticks=len(regular),
            excluded_ticks=excluded_ticks,
            observed_sessions=observed_sessions,
            output_bars=len(bars),
            bucket_coverage=bucket_coverage,
            duplicate_timestamp_rows=duplicate_timestamp_rows,
            exact_duplicate_rows=exact_duplicate_rows,
            crossed_quote_fraction=float(crossed.mean()),
            locked_quote_fraction=float(locked.mean()),
            outside_nbbo_fraction=float(regular["outside_nbbo"].mean()),
            median_spread_bps=float(regular["spread_bps"].median()),
            spread_p90_bps=float(regular["spread_bps"].quantile(0.90)),
            median_ticks_per_bar=float(bars["tick_count"].median()),
            minimum_ticks_per_bar=int(bars["tick_count"].min()),
            status=status,
            reasons=tuple(reasons),
            first_timestamp=regular["date"].iloc[0],
            last_timestamp=regular["date"].iloc[-1],
        )
        return KibotTickResult(
            ticker=normalized_ticker,
            bars=bars,
            audit=audit,
        )
