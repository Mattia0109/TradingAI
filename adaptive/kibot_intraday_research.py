"""Ricerca descrittiva 15m su campioni tick Kibot.

Il modulo collega il loader tick alle feature intraday, alla stabilita' per
blocchi e ai descrittori Lorentziani causali. Non calcola direzioni, outcome
futuri, segnali, operazioni, size, stop, leva o P&L.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from adaptive.intraday_features import (
    IntradayFeatureReport,
    IntradayReferenceFeatureEngine,
)
from adaptive.intraday_stability import (
    IntradayFeatureStabilityAnalyzer,
    IntradayStabilityConfig,
    IntradayStabilityReport,
    add_dimensionless_squeeze_features,
    regular_session_frame,
)
from adaptive.kibot_tick_data import KibotTickLoader, KibotTickResult
from adaptive.lorentzian_research import (
    CausalLorentzianResearchEngine,
    LorentzianResearchConfig,
    LorentzianResearchReport,
    session_phase_context,
    session_slot_context,
)


@dataclass(frozen=True)
class KibotIntradayResearchConfig:
    """Parametri espliciti per campioni brevi, mai promossi a validazione."""

    sessions_per_block: int = 5
    minimum_complete_blocks: int = 3
    context_normalization_window: int = 20
    context_normalization_min_periods: int = 10
    neighbors: int = 8
    minimum_candidates: int = 16
    embargo_bars: int = 4
    sample_stride: int = 4
    history_limit: int = 2_000

    def __post_init__(self) -> None:
        if int(self.sessions_per_block) <= 0:
            raise ValueError("sessions_per_block deve essere positivo.")
        if int(self.minimum_complete_blocks) < 2:
            raise ValueError("minimum_complete_blocks deve essere almeno 2.")
        if int(self.context_normalization_window) < 4:
            raise ValueError("context_normalization_window deve essere almeno 4.")
        if not 2 <= int(self.context_normalization_min_periods) <= int(
            self.context_normalization_window
        ):
            raise ValueError("context_normalization_min_periods non valido.")
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


@dataclass(frozen=True)
class KibotIntradayResearchReport:
    """Output descrittivi integrati del campione tick e delle barre 15m."""

    tick_result: KibotTickResult
    feature_report: IntradayFeatureReport
    features: pd.DataFrame
    stability: IntradayStabilityReport
    lorentzian: LorentzianResearchReport
    phase_microstructure: pd.DataFrame
    caveats: tuple[str, ...]
    research_only: bool = True


def summarize_phase_microstructure(bars: pd.DataFrame) -> pd.DataFrame:
    """Riassume liquidita' osservata per fase senza trasformarla in decisioni."""

    if not isinstance(bars, pd.DataFrame) or bars.empty:
        raise ValueError("bars deve essere un DataFrame non vuoto.")
    required = {
        "date",
        "tick_count",
        "median_spread_bps",
        "spread_p90_bps",
        "outside_nbbo_fraction",
    }
    missing = sorted(required.difference(bars.columns))
    if missing:
        raise ValueError(f"Colonne microstruttura mancanti: {missing}.")
    indexed = bars.copy()
    indexed["date"] = pd.to_datetime(indexed["date"], errors="coerce")
    if indexed["date"].isna().any():
        raise ValueError("Timestamp microstruttura non validi.")
    indexed = indexed.set_index(pd.DatetimeIndex(indexed.pop("date")))
    indexed["session_phase"] = session_phase_context(indexed.index)
    rows: list[dict[str, float | int | str]] = []
    for phase in ("OPEN", "MID_SESSION", "CLOSE"):
        subset = indexed.loc[indexed["session_phase"].eq(phase)]
        if subset.empty:
            continue
        rows.append(
            {
                "session_phase": phase,
                "bars": int(len(subset)),
                "median_tick_count": float(subset["tick_count"].median()),
                "median_spread_bps": float(
                    subset["median_spread_bps"].median()
                ),
                "median_p90_spread_bps": float(
                    subset["spread_p90_bps"].median()
                ),
                "mean_outside_nbbo_fraction": float(
                    subset["outside_nbbo_fraction"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


class KibotIntradayResearchEngine:
    """Esegue la pipeline descrittiva completa su un singolo file Kibot."""

    def __init__(
        self,
        config: KibotIntradayResearchConfig | None = None,
    ) -> None:
        self.config = config or KibotIntradayResearchConfig()

    def analyze(
        self,
        path: str | Path,
        ticker: str = "IVE",
    ) -> KibotIntradayResearchReport:
        settings = self.config
        tick_result = KibotTickLoader().load(path, ticker)
        stability_config = IntradayStabilityConfig(
            sessions_per_block=settings.sessions_per_block,
            minimum_complete_blocks=settings.minimum_complete_blocks,
        )
        regular, _ = regular_session_frame(
            tick_result.bars,
            stability_config,
        )
        feature_report = IntradayReferenceFeatureEngine().compute(regular)
        features = add_dimensionless_squeeze_features(
            feature_report.values,
            regular,
            stability_config,
        )
        stability = IntradayFeatureStabilityAnalyzer(
            stability_config
        ).analyze(tick_result.ticker, features)
        lorentzian_config = LorentzianResearchConfig(
            context_normalization_window=(
                settings.context_normalization_window
            ),
            context_normalization_min_periods=(
                settings.context_normalization_min_periods
            ),
            neighbors=settings.neighbors,
            minimum_candidates=settings.minimum_candidates,
            embargo_bars=settings.embargo_bars,
            sample_stride=settings.sample_stride,
            history_limit=settings.history_limit,
        )
        lorentzian = CausalLorentzianResearchEngine(
            lorentzian_config
        ).compute(
            features,
            session_phase_context(features.index),
            session_slot_context(features.index),
        )
        caveats = tuple(tick_result.audit.reasons) + (
            "Normalizzazione Lorentziana per slot ridotta per il campione breve.",
            "Le fasi della stessa seduta e gli asset correlati non sono prove indipendenti.",
            "Il report verifica la pipeline descrittiva, non la redditivita'.",
        )
        return KibotIntradayResearchReport(
            tick_result=tick_result,
            feature_report=feature_report,
            features=features,
            stability=stability,
            lorentzian=lorentzian,
            phase_microstructure=summarize_phase_microstructure(
                tick_result.bars
            ),
            caveats=caveats,
        )
