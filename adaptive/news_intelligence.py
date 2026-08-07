"""Valutazione strutturata delle notizie e della reazione del mercato.

La classificazione NLP è deliberatamente un'interfaccia esterna: questo modulo
non inventa sentiment da testo senza un modello o provider verificato.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


class NewsEventType(str, Enum):
    EARNINGS = "EARNINGS"
    GUIDANCE = "GUIDANCE"
    MERGER_ACQUISITION = "MERGER_ACQUISITION"
    REGULATORY = "REGULATORY"
    MACRO = "MACRO"
    PRODUCT = "PRODUCT"
    MANAGEMENT = "MANAGEMENT"
    ANALYST = "ANALYST"
    OTHER = "OTHER"


def _finite(name: str, value: float) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} deve essere finito.")
    return normalized


def _unit(name: str, value: float) -> float:
    normalized = _finite(name, value)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} deve essere compreso tra 0 e 1.")
    return normalized


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("Il timestamp della notizia deve essere datetime.")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class ClassifiedNewsEvent:
    event_id: str
    published_at: datetime
    tickers: tuple[str, ...]
    event_type: NewsEventType = NewsEventType.OTHER
    sector: str = "UNKNOWN"
    sentiment: float = 0.0
    importance: float = 0.5
    novelty: float = 0.5
    expected_duration_bars: int = 1
    source_credibility: float = 0.5
    source_latency_seconds: float = 0.0
    price_reaction: float | None = None
    market_adjusted_reaction: float | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        event_id = str(self.event_id).strip()
        if not event_id:
            raise ValueError("event_id non può essere vuoto.")
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "published_at", _utc(self.published_at))
        if isinstance(self.tickers, (str, bytes)):
            raise TypeError("tickers deve essere una sequenza, non una stringa.")
        normalized_tickers = tuple(
            dict.fromkeys(str(ticker).upper().strip() for ticker in self.tickers)
        )
        if not normalized_tickers or any(not ticker for ticker in normalized_tickers):
            raise ValueError("La notizia deve avere almeno un ticker valido.")
        object.__setattr__(self, "tickers", normalized_tickers)
        object.__setattr__(self, "event_type", NewsEventType(self.event_type))
        object.__setattr__(self, "sector", str(self.sector).upper().strip() or "UNKNOWN")

        sentiment = _finite("sentiment", self.sentiment)
        if not -1.0 <= sentiment <= 1.0:
            raise ValueError("sentiment deve essere compreso tra -1 e 1.")
        object.__setattr__(self, "sentiment", sentiment)
        for name in ("importance", "novelty", "source_credibility"):
            object.__setattr__(self, name, _unit(name, getattr(self, name)))
        if int(self.expected_duration_bars) <= 0:
            raise ValueError("expected_duration_bars deve essere positivo.")
        object.__setattr__(
            self,
            "expected_duration_bars",
            int(self.expected_duration_bars),
        )
        latency = _finite("source_latency_seconds", self.source_latency_seconds)
        if latency < 0.0:
            raise ValueError("source_latency_seconds non può essere negativa.")
        object.__setattr__(self, "source_latency_seconds", latency)
        for name in ("price_reaction", "market_adjusted_reaction"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite(name, value))


@dataclass(frozen=True)
class NewsAssessment:
    ticker: str
    as_of: datetime
    event_count: int
    raw_sentiment: float
    adjusted_sentiment: float
    relevance: float
    contradiction_score: float
    event_risk: float
    expected_duration_bars: int
    event_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        ticker = str(self.ticker).upper().strip()
        if not ticker:
            raise ValueError("ticker non può essere vuoto.")
        object.__setattr__(self, "ticker", ticker)
        object.__setattr__(self, "as_of", _utc(self.as_of))
        event_count = int(self.event_count)
        if event_count < 0:
            raise ValueError("event_count non può essere negativo.")
        object.__setattr__(self, "event_count", event_count)
        for name in ("raw_sentiment", "adjusted_sentiment"):
            value = _finite(name, getattr(self, name))
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} deve essere compreso tra -1 e 1.")
        for name in ("relevance", "contradiction_score", "event_risk"):
            _unit(name, getattr(self, name))
        if int(self.expected_duration_bars) <= 0:
            raise ValueError("expected_duration_bars deve essere positivo.")
        if len(self.event_ids) != event_count:
            raise ValueError("event_count ed event_ids non sono coerenti.")


class NewsClassifier(Protocol):
    """Contratto per un futuro modello NLP o provider di news."""

    def classify(
        self,
        headline: str,
        body: str,
        metadata: Mapping[str, Any],
    ) -> ClassifiedNewsEvent: ...


@dataclass(frozen=True)
class NewsIntelligenceConfig:
    maximum_age_hours: float = 72.0
    time_half_life_hours: float = 12.0
    maximum_source_latency_seconds: float = 3_600.0
    reaction_scale: float = 0.02

    def __post_init__(self) -> None:
        for name in (
            "maximum_age_hours",
            "time_half_life_hours",
            "maximum_source_latency_seconds",
            "reaction_scale",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} deve essere positivo.")


class StructuredNewsIntelligence:
    """Aggrega eventi classificati senza usare informazioni future."""

    def __init__(self, config: NewsIntelligenceConfig | None = None) -> None:
        self.config = config or NewsIntelligenceConfig()

    def assess(
        self,
        ticker: str,
        events: Sequence[ClassifiedNewsEvent],
        *,
        as_of: datetime,
    ) -> NewsAssessment:
        normalized_ticker = str(ticker).upper().strip()
        assessment_time = _utc(as_of)
        weighted_events: list[tuple[ClassifiedNewsEvent, float, float, float]] = []

        for event in events:
            if normalized_ticker not in event.tickers:
                continue
            age_hours = (assessment_time - event.published_at).total_seconds() / 3_600.0
            if age_hours < 0.0 or age_hours > self.config.maximum_age_hours:
                continue

            time_weight = 0.5 ** (age_hours / self.config.time_half_life_hours)
            latency_weight = max(
                0.0,
                1.0
                - event.source_latency_seconds
                / self.config.maximum_source_latency_seconds,
            )
            weight = (
                event.importance
                * event.novelty
                * event.source_credibility
                * time_weight
                * latency_weight
            )
            if weight <= 0.0:
                continue

            reaction = (
                event.market_adjusted_reaction
                if event.market_adjusted_reaction is not None
                else event.price_reaction
            )
            reaction_signal = (
                float(np.tanh(reaction / self.config.reaction_scale))
                if reaction is not None
                else event.sentiment
            )
            contradiction = 0.0
            if reaction is not None and event.sentiment * reaction < 0.0:
                contradiction = min(
                    1.0,
                    abs(reaction) / self.config.reaction_scale,
                ) * abs(event.sentiment)
            adjusted = (
                0.30 * event.sentiment + 0.70 * reaction_signal
                if contradiction > 0.0
                else 0.65 * event.sentiment + 0.35 * reaction_signal
            )
            weighted_events.append((event, weight, adjusted, contradiction))

        if not weighted_events:
            return NewsAssessment(
                ticker=normalized_ticker,
                as_of=assessment_time,
                event_count=0,
                raw_sentiment=0.0,
                adjusted_sentiment=0.0,
                relevance=0.0,
                contradiction_score=0.0,
                event_risk=0.0,
                expected_duration_bars=1,
            )

        total_weight = sum(item[1] for item in weighted_events)
        raw_sentiment = sum(
            event.sentiment * weight
            for event, weight, _, _ in weighted_events
        ) / total_weight
        adjusted_sentiment = sum(
            adjusted * weight
            for _, weight, adjusted, _ in weighted_events
        ) / total_weight
        contradiction_score = sum(
            contradiction * weight
            for _, weight, _, contradiction in weighted_events
        ) / total_weight
        maximum_weight = max(item[1] for item in weighted_events)
        relevance = max(maximum_weight, 1.0 - math.exp(-total_weight))
        event_risk = max(
            weight * (0.50 + 0.50 * contradiction)
            for _, weight, _, contradiction in weighted_events
        )
        expected_duration = round(
            sum(
                event.expected_duration_bars * weight
                for event, weight, _, _ in weighted_events
            )
            / total_weight
        )

        return NewsAssessment(
            ticker=normalized_ticker,
            as_of=assessment_time,
            event_count=len(weighted_events),
            raw_sentiment=float(np.clip(raw_sentiment, -1.0, 1.0)),
            adjusted_sentiment=float(np.clip(adjusted_sentiment, -1.0, 1.0)),
            relevance=float(np.clip(relevance, 0.0, 1.0)),
            contradiction_score=float(np.clip(contradiction_score, 0.0, 1.0)),
            event_risk=float(np.clip(event_risk, 0.0, 1.0)),
            expected_duration_bars=max(1, expected_duration),
            event_ids=tuple(item[0].event_id for item in weighted_events),
        )
