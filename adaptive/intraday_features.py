"""Feature intraday descrittive derivate da riferimenti Pine dichiarati.

Il modulo replica soltanto formule numeriche osservabili. Non combina le
feature in un punteggio, non classifica la direzione e non produce segnali,
ordini, size, stop, leva o P&L.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


class FeatureParity(str, Enum):
    """Livello di equivalenza verificabile rispetto al riferimento."""

    EXACT_REFERENCE_FORMULA = "EXACT_REFERENCE_FORMULA"
    PARTIAL_MISSING_LIBRARY_SOURCE = "PARTIAL_MISSING_LIBRARY_SOURCE"
    UNAVAILABLE_NO_VOLUME = "UNAVAILABLE_NO_VOLUME"


@dataclass(frozen=True)
class IntradayFeatureConfig:
    """Parametri predefiniti copiati come valori, non come codice Pine."""

    squeeze_bb_length: int = 20
    squeeze_declared_bb_multiplier: float = 2.0
    squeeze_kc_length: int = 20
    squeeze_kc_multiplier: float = 1.5
    squeeze_use_true_range: bool = True
    chop_length: int = 10
    chop_choppy_threshold: float = 60.0
    chop_trending_threshold: float = 40.0
    cmf_length: int = 20
    cmf_gradient_length: int = 100

    def __post_init__(self) -> None:
        for name in (
            "squeeze_bb_length",
            "squeeze_kc_length",
            "chop_length",
            "cmf_length",
            "cmf_gradient_length",
        ):
            if int(getattr(self, name)) <= 1:
                raise ValueError(f"{name} deve essere maggiore di 1.")
        if self.squeeze_declared_bb_multiplier <= 0.0:
            raise ValueError("squeeze_declared_bb_multiplier deve essere positivo.")
        if self.squeeze_kc_multiplier <= 0.0:
            raise ValueError("squeeze_kc_multiplier deve essere positivo.")
        if not 0.0 <= self.chop_trending_threshold <= 100.0:
            raise ValueError("chop_trending_threshold fuori intervallo.")
        if not 0.0 <= self.chop_choppy_threshold <= 100.0:
            raise ValueError("chop_choppy_threshold fuori intervallo.")
        if self.chop_trending_threshold >= self.chop_choppy_threshold:
            raise ValueError("Le soglie CHOP non sono ordinate.")

    @property
    def minimum_history(self) -> int:
        squeeze_history = 2 * int(self.squeeze_kc_length) - 1
        return max(
            squeeze_history,
            int(self.squeeze_bb_length),
            int(self.chop_length),
            int(self.cmf_length),
        )


@dataclass(frozen=True)
class IntradayFeatureReport:
    """Serie descrittive e limiti di parità, senza output operativo."""

    values: pd.DataFrame
    parity: Mapping[str, FeatureParity]
    caveats: tuple[str, ...]
    research_only: bool = True


def lorentzian_distance(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    """Distanza del riferimento: somma di log(1 + differenza assoluta)."""

    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    if left_values.ndim != 1 or right_values.ndim != 1:
        raise ValueError("I vettori Lorentziani devono essere monodimensionali.")
    if left_values.shape != right_values.shape:
        raise ValueError("I vettori Lorentziani devono avere la stessa dimensione.")
    if not 2 <= left_values.size <= 5:
        raise ValueError("Il riferimento accetta da 2 a 5 feature.")
    if not np.isfinite(left_values).all() or not np.isfinite(right_values).all():
        raise ValueError("Le feature Lorentziane devono essere finite.")
    return float(np.log1p(np.abs(left_values - right_values)).sum())


class IntradayReferenceFeatureEngine:
    """Replica causale delle formule complete disponibili nei riferimenti."""

    price_columns = ("open", "high", "low", "close")
    required_columns = (*price_columns, "volume")

    def __init__(self, config: IntradayFeatureConfig | None = None) -> None:
        self.config = config or IntradayFeatureConfig()

    @staticmethod
    def _true_range(frame: pd.DataFrame) -> pd.Series:
        previous_close = frame["close"].shift(1)
        return pd.concat(
            [
                frame["high"] - frame["low"],
                (frame["high"] - previous_close).abs(),
                (frame["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1, skipna=True)

    @staticmethod
    def _pine_rma(values: pd.Series, length: int) -> pd.Series:
        """Wilder RMA con seed SMA, equivalente alla semantica di Pine."""

        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        output = np.full(numeric.size, np.nan, dtype=float)
        seed_position: int | None = None
        for position in range(length - 1, numeric.size):
            window = numeric[position - length + 1 : position + 1]
            if np.isfinite(window).all():
                output[position] = float(window.mean())
                seed_position = position
                break
        if seed_position is None:
            return pd.Series(output, index=values.index, dtype=float)
        for position in range(seed_position + 1, numeric.size):
            current = numeric[position]
            previous = output[position - 1]
            if math.isfinite(current) and math.isfinite(previous):
                output[position] = (
                    previous * (length - 1) + current
                ) / length
        return pd.Series(output, index=values.index, dtype=float)

    @staticmethod
    def _pine_linreg(values: pd.Series, length: int, offset: int = 0) -> pd.Series:
        """Valore finale della regressione lineare mobile usata da Pine."""

        x = np.arange(length, dtype=float)
        x_mean = float(x.mean())
        denominator = float(((x - x_mean) ** 2).sum())

        def endpoint(window: np.ndarray) -> float:
            if not np.isfinite(window).all():
                return np.nan
            y_mean = float(window.mean())
            slope = float(((x - x_mean) * (window - y_mean)).sum()) / denominator
            intercept = y_mean - slope * x_mean
            return intercept + slope * (length - 1 - offset)

        return values.rolling(length, min_periods=length).apply(
            endpoint,
            raw=True,
        )

    def _prepare(
        self,
        data: pd.DataFrame,
        allow_price_only: bool = False,
    ) -> pd.DataFrame:
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
        required = self.price_columns if allow_price_only else self.required_columns
        missing = [column for column in required if column not in frame]
        if missing:
            raise ValueError(f"Colonne OHLCV mancanti: {missing}.")

        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        if frame["date"].isna().any():
            raise ValueError("Sono presenti timestamp non validi.")
        if frame["date"].duplicated().any():
            raise ValueError("Sono presenti timestamp duplicati.")
        numeric_columns = list(self.price_columns)
        if "volume" in frame:
            numeric_columns.append("volume")
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[numeric_columns].isna().any().any():
            raise ValueError("Sono presenti valori OHLCV non numerici.")
        if (frame[["open", "high", "low", "close"]] <= 0.0).any().any():
            raise ValueError("I prezzi devono essere positivi.")
        if "volume" in frame:
            if (frame["volume"] < 0.0).any():
                raise ValueError("Il volume non può essere negativo.")
            if float(frame["volume"].sum()) <= 0.0:
                raise ValueError("Il riferimento CMF richiede dati di volume.")
        invalid_ohlc = (
            (frame["high"] < frame["low"])
            | (frame["high"] < frame[["open", "close"]].max(axis=1))
            | (frame["low"] > frame[["open", "close"]].min(axis=1))
        )
        if invalid_ohlc.any():
            raise ValueError("Relazione OHLC non valida.")
        return frame.sort_values("date").reset_index(drop=True)

    def compute(
        self,
        data: pd.DataFrame,
        allow_price_only: bool = False,
    ) -> IntradayFeatureReport:
        """Calcola serie trailing; ogni riga dipende soltanto da dati fino a t."""

        config = self.config
        frame = self._prepare(data, allow_price_only=allow_price_only)
        has_volume = "volume" in frame
        close = frame["close"]
        high = frame["high"]
        low = frame["low"]
        true_range = self._true_range(frame)

        bb_basis = close.rolling(config.squeeze_bb_length).mean()
        # Il riferimento fornito dichiara un moltiplicatore BB separato, ma
        # applica intenzionalmente multKC anche alla deviazione BB.
        bb_deviation = (
            close.rolling(config.squeeze_bb_length).std(ddof=0)
            * config.squeeze_kc_multiplier
        )
        upper_bb = bb_basis + bb_deviation
        lower_bb = bb_basis - bb_deviation

        kc_basis = close.rolling(config.squeeze_kc_length).mean()
        kc_range = (
            true_range
            if config.squeeze_use_true_range
            else high - low
        )
        kc_range_mean = kc_range.rolling(config.squeeze_kc_length).mean()
        upper_kc = kc_basis + kc_range_mean * config.squeeze_kc_multiplier
        lower_kc = kc_basis - kc_range_mean * config.squeeze_kc_multiplier
        squeeze_valid = pd.concat(
            [upper_bb, lower_bb, upper_kc, lower_kc],
            axis=1,
        ).notna().all(axis=1)
        squeeze_on = squeeze_valid & (lower_bb > lower_kc) & (upper_bb < upper_kc)
        squeeze_off = squeeze_valid & (lower_bb < lower_kc) & (upper_bb > upper_kc)
        squeeze_state = np.full(len(frame), "INSUFFICIENT", dtype=object)
        squeeze_state[squeeze_valid.to_numpy()] = "NEUTRAL"
        squeeze_state[squeeze_off.to_numpy()] = "SQUEEZE_OFF"
        squeeze_state[squeeze_on.to_numpy()] = "SQUEEZE_ON"

        high_low_midpoint = (
            high.rolling(config.squeeze_kc_length).max()
            + low.rolling(config.squeeze_kc_length).min()
        ) / 2.0
        detrended_midpoint = (high_low_midpoint + kc_basis) / 2.0
        detrended = close - detrended_midpoint
        squeeze_momentum = self._pine_linreg(
            detrended,
            config.squeeze_kc_length,
            offset=0,
        )

        atr = self._pine_rma(true_range, config.chop_length)
        chop_range = (
            high.rolling(config.chop_length).max()
            - low.rolling(config.chop_length).min()
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            choppiness = 100.0 + (
                100.0
                * np.log10(atr / chop_range)
                / math.log10(config.chop_length)
            )
        choppiness = pd.Series(choppiness, index=frame.index, dtype=float).replace(
            [np.inf, -np.inf],
            np.nan,
        )
        chop_segment = np.full(len(frame), "INSUFFICIENT", dtype=object)
        chop_valid = choppiness.notna()
        chop_segment[chop_valid.to_numpy()] = "NEUTRAL"
        chop_segment[
            (chop_valid & (choppiness > config.chop_choppy_threshold)).to_numpy()
        ] = "CHOPPY"
        chop_segment[
            (chop_valid & (choppiness < config.chop_trending_threshold)).to_numpy()
        ] = "TRENDING"

        if has_volume:
            volume = frame["volume"]
            candle_range = high - low
            accumulation_distribution = pd.Series(
                np.where(
                    candle_range == 0.0,
                    0.0,
                    ((2.0 * close - low - high) / candle_range) * volume,
                ),
                index=frame.index,
                dtype=float,
            )
            rolling_volume = volume.rolling(config.cmf_length).sum()
            cmf = (
                accumulation_distribution.rolling(config.cmf_length).sum()
                / rolling_volume.replace(0.0, np.nan)
            )
            cmf_highest = cmf.rolling(config.cmf_gradient_length).max()
            cmf_lowest = cmf.rolling(config.cmf_gradient_length).min()
            cmf_gradient_extreme = pd.concat(
                [cmf_highest, cmf_lowest.abs()],
                axis=1,
            ).max(axis=1)
        else:
            cmf = pd.Series(np.nan, index=frame.index, dtype=float)
            cmf_gradient_extreme = pd.Series(
                np.nan,
                index=frame.index,
                dtype=float,
            )

        values = pd.DataFrame(
            {
                "squeeze_momentum": squeeze_momentum,
                "squeeze_momentum_change": squeeze_momentum.diff(),
                "squeeze_on": squeeze_on,
                "squeeze_off": squeeze_off,
                "squeeze_state": squeeze_state,
                "choppiness": choppiness,
                "chop_segment": chop_segment,
                "cmf": cmf,
                "cmf_gradient_extreme": cmf_gradient_extreme,
            },
        )
        values.index = pd.DatetimeIndex(frame["date"], name="date")
        parity = {
            "squeeze_momentum": FeatureParity.EXACT_REFERENCE_FORMULA,
            "choppiness": FeatureParity.EXACT_REFERENCE_FORMULA,
            "cmf": (
                FeatureParity.EXACT_REFERENCE_FORMULA
                if has_volume
                else FeatureParity.UNAVAILABLE_NO_VOLUME
            ),
            "lorentzian_distance": FeatureParity.EXACT_REFERENCE_FORMULA,
            "lorentzian_classifier": (
                FeatureParity.PARTIAL_MISSING_LIBRARY_SOURCE
            ),
        }
        caveats = [
            "Il moltiplicatore BB dichiarato 2.0 non è usato dal riferimento; "
            "la deviazione BB usa il moltiplicatore KC 1.5.",
            "Il classificatore Lorentzian completo dipende da MLExtensions/2 "
            "e KernelFunctions/2: qui è disponibile soltanto la distanza esatta.",
            "Colori, alert, filtri operativi e statistiche del riferimento sono "
            "deliberatamente esclusi.",
        ]
        if not has_volume:
            caveats.append(
                "Volume assente: CMF non viene calcolato e rimane NaN."
            )
        return IntradayFeatureReport(
            values=values,
            parity=parity,
            caveats=tuple(caveats),
        )
