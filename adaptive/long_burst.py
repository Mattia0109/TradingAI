"""Motore causale LONG/NO_TRADE per rialzi brevi, solo ricerca.

La V1 combina quattro informazioni complementari:

* Choppiness Index per il regime;
* classificazione k-NN con distanza Lorentziana per la direzione;
* Squeeze Momentum per timing e accelerazione;
* Chaikin Money Flow per la conferma del volume.

Il modulo non contiene broker, ordini, position sizing o leva.  Produce
soltanto segnali descrittivi e forecast paper-only con orizzonte da una a
cinque barre.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd

from adaptive.models import Direction, StrategyForecast


class LongBurstRegime(str, Enum):
    """Regimi essenziali per il Challenger rialzista di breve durata."""

    BREAKOUT = "BREAKOUT"
    TREND = "TREND"
    MIXED = "MIXED"
    RANGE = "RANGE"
    SHOCK = "SHOCK"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True)
class LongBurstConfig:
    """Parametri statici della V1; nessun valore viene ottimizzato online."""

    annualization_factor: int = 252
    forecast_horizon_bars: int = 3
    chop_window: int = 14
    breakout_window: int = 20
    bollinger_window: int = 20
    bollinger_deviations: float = 2.0
    keltner_window: int = 20
    keltner_atr_multiplier: float = 1.5
    cmf_window: int = 20
    volatility_window: int = 20
    long_volatility_window: int = 60
    correlation_window: int = 60
    lorentzian_lookback: int = 252
    lorentzian_neighbors: int = 8
    lorentzian_minimum_samples: int = 24
    lorentzian_sample_stride: int = 5
    lorentzian_weight: float = 0.45
    squeeze_weight: float = 0.35
    cmf_weight: float = 0.20
    minimum_component_support: int = 2
    minimum_lorentzian_score: float = 0.02
    minimum_raw_score: float = 0.18
    minimum_confidence: float = 0.10
    minimum_net_edge_bps: float = 5.0
    minimum_opportunity_score: float = 0.02
    maximum_choppiness: float = 70.0
    shock_threshold: float = 4.0
    minimum_data_quality: float = 0.90
    round_trip_cost_bps: float = 10.0

    def __post_init__(self) -> None:
        integer_fields = (
            "annualization_factor",
            "chop_window",
            "breakout_window",
            "bollinger_window",
            "keltner_window",
            "cmf_window",
            "volatility_window",
            "long_volatility_window",
            "correlation_window",
            "lorentzian_lookback",
            "lorentzian_neighbors",
            "lorentzian_minimum_samples",
            "lorentzian_sample_stride",
        )
        for name in integer_fields:
            if int(getattr(self, name)) <= 1:
                raise ValueError(f"{name} deve essere maggiore di 1.")
        if not 1 <= int(self.forecast_horizon_bars) <= 5:
            raise ValueError("forecast_horizon_bars deve essere compreso tra 1 e 5.")
        if self.lorentzian_sample_stride < self.forecast_horizon_bars:
            raise ValueError(
                "lorentzian_sample_stride deve essere almeno pari all'orizzonte."
            )
        if not 1 <= int(self.minimum_component_support) <= 3:
            raise ValueError("minimum_component_support deve essere compreso tra 1 e 3.")
        if self.bollinger_deviations <= 0.0:
            raise ValueError("bollinger_deviations deve essere positivo.")
        if self.keltner_atr_multiplier <= 0.0:
            raise ValueError("keltner_atr_multiplier deve essere positivo.")
        weights = (
            float(self.lorentzian_weight),
            float(self.squeeze_weight),
            float(self.cmf_weight),
        )
        if any(weight < 0.0 for weight in weights):
            raise ValueError("I pesi dei segnali non possono essere negativi.")
        if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("I pesi Lorentzian, Squeeze e CMF devono sommare a 1.")
        for name in (
            "minimum_lorentzian_score",
            "minimum_raw_score",
            "minimum_confidence",
            "minimum_opportunity_score",
            "minimum_data_quality",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} deve essere compreso tra 0 e 1.")
        if not 0.0 <= self.maximum_choppiness <= 100.0:
            raise ValueError("maximum_choppiness deve essere compreso tra 0 e 100.")
        if self.shock_threshold <= 0.0:
            raise ValueError("shock_threshold deve essere positivo.")
        if self.minimum_net_edge_bps < 0.0 or self.round_trip_cost_bps < 0.0:
            raise ValueError("Costi ed edge minimo non possono essere negativi.")

    @property
    def minimum_history(self) -> int:
        feature_warmup = max(
            self.breakout_window,
            self.bollinger_window,
            self.keltner_window,
            self.cmf_window,
            self.volatility_window,
            self.long_volatility_window,
            self.correlation_window,
        ) + 6
        labelled_history = (
            self.lorentzian_minimum_samples * self.lorentzian_sample_stride
            + self.forecast_horizon_bars
            + max(self.volatility_window, 21)
        )
        return max(feature_warmup, labelled_history)


@dataclass(frozen=True)
class LongBurstSignal:
    """Fotografia immutabile dell'ultima decisione del signal engine."""

    ticker: str
    timestamp: datetime
    direction: Direction
    horizon_bars: int
    regime: LongBurstRegime
    choppiness: float
    lorentzian_score: float
    squeeze_score: float
    cmf_score: float
    breakout_strength: float
    price_acceleration: float
    raw_score: float
    agreement: float
    correlation_penalty: float
    regime_quality: float
    data_quality: float
    confidence: float
    expected_return: float
    expected_risk: float
    estimated_cost_return: float
    net_edge: float
    opportunity_score: float
    lorentzian_samples: int
    reason_code: str
    paper_only: bool = True

    def __post_init__(self) -> None:
        ticker = str(self.ticker).upper().strip()
        if not ticker:
            raise ValueError("ticker non può essere vuoto.")
        object.__setattr__(self, "ticker", ticker)
        if self.direction not in (Direction.LONG, Direction.FLAT):
            raise ValueError("LongBurstSignal accetta soltanto LONG o FLAT.")
        if not 1 <= int(self.horizon_bars) <= 5:
            raise ValueError("horizon_bars deve essere compreso tra 1 e 5.")
        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp deve essere datetime.")
        bounded = (
            "agreement",
            "correlation_penalty",
            "regime_quality",
            "data_quality",
            "confidence",
        )
        for name in bounded:
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} deve essere finito e compreso tra 0 e 1.")
        signed = (
            "lorentzian_score",
            "squeeze_score",
            "cmf_score",
            "breakout_strength",
            "price_acceleration",
            "raw_score",
        )
        for name in signed:
            value = float(getattr(self, name))
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} deve essere finito e compreso tra -1 e 1.")
        for name in (
            "expected_return",
            "expected_risk",
            "estimated_cost_return",
            "net_edge",
            "opportunity_score",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} deve essere finito.")
        if self.expected_return < 0.0 or self.expected_risk < 0.0:
            raise ValueError("Rendimento atteso e rischio atteso non possono essere negativi.")
        if self.estimated_cost_return < 0.0:
            raise ValueError("estimated_cost_return non può essere negativo.")
        if int(self.lorentzian_samples) < 0:
            raise ValueError("lorentzian_samples non può essere negativo.")
        if self.paper_only is not True:
            raise ValueError("LongBurstSignal deve restare paper-only.")

    def to_forecast(self) -> StrategyForecast:
        """Adatta il segnale al contratto esistente senza renderlo operativo."""

        annualized_volatility = (
            self.expected_risk
            / math.sqrt(self.horizon_bars)
            * math.sqrt(252.0)
        )
        regime_fit = self.regime_quality if self.direction is Direction.LONG else 0.0
        return StrategyForecast(
            strategy_id="long_burst_momentum_v1",
            ticker=self.ticker,
            timestamp=self.timestamp,
            direction=self.direction,
            confidence=self.confidence if self.direction is Direction.LONG else 0.0,
            expected_return=(
                self.expected_return if self.direction is Direction.LONG else 0.0
            ),
            expected_volatility=annualized_volatility,
            horizon_bars=self.horizon_bars,
            regime_fit=regime_fit,
            reasons=(self.reason_code,),
            metadata={
                "choppiness": self.choppiness,
                "lorentzian_score": self.lorentzian_score,
                "squeeze_score": self.squeeze_score,
                "cmf_score": self.cmf_score,
                "raw_score": self.raw_score,
                "net_edge_bps": self.net_edge * 10_000.0,
            },
        )


class LongBurstSignalEngine:
    """Calcola in batch una serie di segnali causali LONG/NO_TRADE."""

    required_columns = ("open", "high", "low", "close", "volume")

    def __init__(self, config: LongBurstConfig | None = None) -> None:
        self.config = config or LongBurstConfig()

    @staticmethod
    def _prepare(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data deve essere un pandas DataFrame.")
        if data.empty:
            raise ValueError("data non può essere vuoto.")
        frame = data.copy()
        frame.columns = [str(column).lower().strip() for column in frame.columns]
        missing = [
            column
            for column in LongBurstSignalEngine.required_columns
            if column not in frame.columns
        ]
        if missing:
            raise ValueError(f"Colonne OHLCV mancanti: {missing}.")

        if "date" in frame.columns:
            timestamp = pd.to_datetime(frame["date"], errors="coerce", utc=True)
            if timestamp.isna().any():
                raise ValueError("Sono presenti timestamp non validi.")
            frame.index = pd.DatetimeIndex(timestamp)
        elif isinstance(frame.index, pd.DatetimeIndex):
            frame.index = pd.to_datetime(frame.index, utc=True)
        else:
            raise ValueError("Serve una colonna date o un DatetimeIndex.")
        if frame.index.has_duplicates:
            raise ValueError("Sono presenti timestamp duplicati.")
        frame = frame.sort_index()

        for column in LongBurstSignalEngine.required_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[["open", "high", "low", "close"]].isna().any().any():
            raise ValueError("Sono presenti prezzi OHLC mancanti o non numerici.")
        if (frame[["open", "high", "low", "close"]] <= 0.0).any().any():
            raise ValueError("I prezzi OHLC devono essere positivi.")
        invalid_range = (
            (frame["high"] < frame["low"])
            | (frame["high"] < frame[["open", "close"]].max(axis=1))
            | (frame["low"] > frame[["open", "close"]].min(axis=1))
        )
        if bool(invalid_range.any()):
            raise ValueError("Relazione OHLC non valida in una o più barre.")

        valid_volume = frame["volume"].notna() & (frame["volume"] >= 0.0)
        frame["volume"] = frame["volume"].where(valid_volume, 0.0)
        return frame[list(LongBurstSignalEngine.required_columns)], valid_volume

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
        ).max(axis=1)

    @staticmethod
    def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        average_gain = gain.ewm(
            alpha=1.0 / window,
            adjust=False,
            min_periods=window,
        ).mean()
        average_loss = loss.ewm(
            alpha=1.0 / window,
            adjust=False,
            min_periods=window,
        ).mean()
        relative_strength = average_gain / average_loss.replace(0.0, np.nan)
        rsi = 100.0 - 100.0 / (1.0 + relative_strength)
        rsi = rsi.where(average_loss > 0.0, 100.0)
        return rsi.where(average_gain > 0.0, 0.0).clip(0.0, 100.0)

    def _lorentzian_classifier(
        self,
        features: pd.DataFrame,
        close: pd.Series,
        daily_volatility: pd.Series,
    ) -> pd.DataFrame:
        """k-NN causale: ogni label storica deve essere già maturata."""

        config = self.config
        horizon = config.forecast_horizon_bars
        feature_values = features.to_numpy(dtype=float)
        prices = close.to_numpy(dtype=float)
        volatility = daily_volatility.to_numpy(dtype=float)
        rows = len(features)
        scores = np.zeros(rows, dtype=float)
        expected = np.zeros(rows, dtype=float)
        uncertainty = np.zeros(rows, dtype=float)
        samples = np.zeros(rows, dtype=int)

        for position in range(rows):
            current = feature_values[position]
            if not np.isfinite(current).all() or position < horizon:
                continue
            first = max(0, position - config.lorentzian_lookback)
            last = position - horizon
            candidates = np.arange(
                first,
                last + 1,
                config.lorentzian_sample_stride,
                dtype=int,
            )
            if candidates.size == 0:
                continue
            candidate_features = feature_values[candidates]
            labels = prices[candidates + horizon] / prices[candidates] - 1.0
            valid = np.isfinite(candidate_features).all(axis=1) & np.isfinite(labels)
            candidate_features = candidate_features[valid]
            labels = labels[valid]
            samples[position] = int(labels.size)
            if labels.size < config.lorentzian_minimum_samples:
                continue

            distances = np.log1p(np.abs(candidate_features - current)).sum(axis=1)
            neighbours = min(config.lorentzian_neighbors, labels.size)
            nearest = np.argpartition(distances, neighbours - 1)[:neighbours]
            neighbour_distances = distances[nearest]
            neighbour_labels = labels[nearest]
            weights = 1.0 / (1.0 + neighbour_distances)
            weight_total = float(weights.sum())
            if not math.isfinite(weight_total) or weight_total <= 0.0:
                continue

            horizon_volatility = (
                volatility[position] * math.sqrt(horizon)
                if math.isfinite(volatility[position])
                else 0.0
            )
            label_scale = max(
                float(np.std(labels, ddof=1)) if labels.size > 1 else 0.0,
                horizon_volatility,
                1e-6,
            )
            directional_labels = np.tanh(neighbour_labels / label_scale)
            scores[position] = float(
                np.clip(np.average(directional_labels, weights=weights), -1.0, 1.0)
            )
            prediction = float(np.average(neighbour_labels, weights=weights))
            expected[position] = float(
                np.clip(prediction, -3.0 * label_scale, 3.0 * label_scale)
            )
            uncertainty[position] = float(
                math.sqrt(
                    max(
                        0.0,
                        np.average(
                            (neighbour_labels - prediction) ** 2,
                            weights=weights,
                        ),
                    )
                )
            )

        return pd.DataFrame(
            {
                "lorentzian_score": scores,
                "lorentzian_expected_return": expected,
                "lorentzian_uncertainty": uncertainty,
                "lorentzian_samples": samples,
            },
            index=features.index,
        )

    @staticmethod
    def _agreement(values: np.ndarray, weights: np.ndarray) -> float:
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        active = np.abs(values) >= 0.05
        if not bool(active.any()):
            return 0.0
        active_weights = weights[active]
        sign_consistency = abs(
            float(np.dot(active_weights, np.sign(values[active])))
        ) / float(active_weights.sum())
        magnitude_coherence = float(
            np.clip(1.0 - np.std(values) / 0.90, 0.0, 1.0)
        )
        return float(np.clip(0.70 * sign_consistency + 0.30 * magnitude_coherence, 0.0, 1.0))

    def evaluate_history(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Restituisce tutte le decisioni storiche usando solo dati fino a t."""

        config = self.config
        frame, valid_volume = self._prepare(data)
        close = frame["close"]
        high = frame["high"]
        low = frame["low"]
        volume = frame["volume"]
        true_range = self._true_range(frame)
        atr = true_range.rolling(config.keltner_window).mean()
        returns = close.pct_change()
        daily_volatility = returns.rolling(config.volatility_window).std(ddof=1)
        long_volatility = returns.rolling(config.long_volatility_window).std(ddof=1)
        expected_risk = daily_volatility * math.sqrt(config.forecast_horizon_bars)

        rolling_high = high.rolling(config.chop_window).max()
        rolling_low = low.rolling(config.chop_window).min()
        price_range = (rolling_high - rolling_low).replace(0.0, np.nan)
        choppiness = (
            100.0
            * np.log10(true_range.rolling(config.chop_window).sum() / price_range)
            / math.log10(config.chop_window)
        ).clip(0.0, 100.0)

        bollinger_basis = close.rolling(config.bollinger_window).mean()
        bollinger_width = (
            close.rolling(config.bollinger_window).std(ddof=0)
            * config.bollinger_deviations
        )
        keltner_basis = close.ewm(
            span=config.keltner_window,
            adjust=False,
            min_periods=config.keltner_window,
        ).mean()
        keltner_width = atr * config.keltner_atr_multiplier
        squeeze_on = (
            (bollinger_basis - bollinger_width > keltner_basis - keltner_width)
            & (bollinger_basis + bollinger_width < keltner_basis + keltner_width)
        )
        squeeze_release = squeeze_on.shift(1, fill_value=False) & ~squeeze_on
        momentum_midpoint = (
            (high.rolling(config.bollinger_window).max() + low.rolling(config.bollinger_window).min())
            / 2.0
            + bollinger_basis
        ) / 2.0
        detrended = close - momentum_midpoint
        normalized_momentum = detrended / atr.replace(0.0, np.nan)
        momentum_acceleration = (
            detrended - detrended.shift(3)
        ) / atr.replace(0.0, np.nan)
        squeeze_score = np.tanh(
            0.65 * normalized_momentum + 0.35 * momentum_acceleration
        )
        squeeze_score = pd.Series(squeeze_score, index=frame.index, dtype=float)
        squeeze_score = squeeze_score.where(~squeeze_on, squeeze_score * 0.50)
        squeeze_score = (
            squeeze_score
            + 0.15 * (squeeze_release & (squeeze_score > 0.0)).astype(float)
        ).clip(-1.0, 1.0)

        candle_range = (high - low).replace(0.0, np.nan)
        money_flow_multiplier = (
            ((close - low) - (high - close)) / candle_range
        ).fillna(0.0)
        money_flow_volume = money_flow_multiplier * volume
        cmf = (
            money_flow_volume.rolling(config.cmf_window).sum()
            / volume.rolling(config.cmf_window).sum().replace(0.0, np.nan)
        ).fillna(0.0)
        cmf_acceleration = cmf - cmf.shift(5)
        cmf_score = pd.Series(
            np.tanh(2.5 * cmf + 1.5 * cmf_acceleration.fillna(0.0)),
            index=frame.index,
            dtype=float,
        ).clip(-1.0, 1.0)

        prior_high = high.shift(1).rolling(config.breakout_window).max()
        breakout_strength = pd.Series(
            np.tanh((close - prior_high) / atr.replace(0.0, np.nan)),
            index=frame.index,
            dtype=float,
        ).fillna(0.0).clip(-1.0, 1.0)
        log_close = np.log(close)
        recent_momentum = log_close - log_close.shift(3)
        previous_momentum = log_close.shift(3) - log_close.shift(6)
        price_acceleration = pd.Series(
            np.tanh(
                (recent_momentum - previous_momentum)
                / (daily_volatility.replace(0.0, np.nan) * math.sqrt(3.0))
            ),
            index=frame.index,
            dtype=float,
        ).fillna(0.0).clip(-1.0, 1.0)

        log_volume = np.log1p(volume.clip(lower=0.0))
        volume_deviation = log_volume.rolling(config.volatility_window).std(ddof=1)
        volume_zscore = (
            (log_volume - log_volume.rolling(config.volatility_window).mean())
            / volume_deviation.replace(0.0, np.nan)
        ).fillna(0.0)
        ema_fast = close.ewm(span=8, adjust=False, min_periods=8).mean()
        ema_slow = close.ewm(span=21, adjust=False, min_periods=21).mean()
        trend_feature = pd.Series(
            np.tanh((ema_fast - ema_slow) / atr.replace(0.0, np.nan)),
            index=frame.index,
            dtype=float,
        )
        feature_frame = pd.DataFrame(
            {
                "momentum_3": np.tanh(
                    (log_close - log_close.shift(3))
                    / (daily_volatility.replace(0.0, np.nan) * math.sqrt(3.0))
                ),
                "momentum_5": np.tanh(
                    (log_close - log_close.shift(5))
                    / (daily_volatility.replace(0.0, np.nan) * math.sqrt(5.0))
                ),
                "rsi": (self._rsi(close) - 50.0) / 50.0,
                "trend": trend_feature,
                "volume": np.tanh(volume_zscore / 2.0),
            },
            index=frame.index,
        ).replace([np.inf, -np.inf], np.nan)
        lorentzian = self._lorentzian_classifier(
            feature_frame,
            close,
            daily_volatility,
        )

        component_weights = np.asarray(
            [
                config.lorentzian_weight,
                config.squeeze_weight,
                config.cmf_weight,
            ],
            dtype=float,
        )
        squeeze_component = squeeze_score.fillna(0.0).clip(-1.0, 1.0)
        components = np.column_stack(
            [
                lorentzian["lorentzian_score"].to_numpy(dtype=float),
                squeeze_component.to_numpy(dtype=float),
                cmf_score.to_numpy(dtype=float),
            ]
        )
        raw_score = pd.Series(
            np.clip(components @ component_weights, -1.0, 1.0),
            index=frame.index,
            dtype=float,
        )
        agreement = pd.Series(
            [self._agreement(values, component_weights) for values in components],
            index=frame.index,
            dtype=float,
        )

        correlations = pd.concat(
            [
                lorentzian["lorentzian_score"].rolling(config.correlation_window).corr(squeeze_component).rename("lorentzian_squeeze"),
                lorentzian["lorentzian_score"].rolling(config.correlation_window).corr(cmf_score).rename("lorentzian_cmf"),
                squeeze_component.rolling(config.correlation_window).corr(cmf_score).rename("squeeze_cmf"),
            ],
            axis=1,
        )
        maximum_positive_correlation = correlations.clip(lower=0.0).max(axis=1).fillna(0.0)
        correlation_penalty = (
            ((maximum_positive_correlation - 0.75) / 0.25).clip(0.0, 1.0)
            * 0.20
        )

        shock_score = (
            returns.abs() / daily_volatility.replace(0.0, np.nan)
        ).fillna(0.0)
        breakout_regime = (
            squeeze_release
            & (breakout_strength > 0.0)
            & (squeeze_score > 0.0)
        )
        shock_regime = shock_score >= config.shock_threshold
        trend_regime = (
            (choppiness <= 55.0)
            & (ema_fast > ema_slow)
            & ~breakout_regime
            & ~shock_regime
        )
        range_regime = (
            (choppiness >= 61.8)
            & ~breakout_regime
            & ~shock_regime
        )
        regimes = np.full(len(frame), LongBurstRegime.MIXED.value, dtype=object)
        regimes[range_regime.fillna(False).to_numpy()] = LongBurstRegime.RANGE.value
        regimes[trend_regime.fillna(False).to_numpy()] = LongBurstRegime.TREND.value
        regimes[breakout_regime.fillna(False).to_numpy()] = LongBurstRegime.BREAKOUT.value
        regimes[shock_regime.fillna(False).to_numpy()] = LongBurstRegime.SHOCK.value

        trend_quality = ((65.0 - choppiness) / 30.0).clip(0.35, 1.0)
        regime_quality = pd.Series(0.45, index=frame.index, dtype=float)
        regime_quality = regime_quality.where(~range_regime, 0.15)
        regime_quality = regime_quality.where(~trend_regime, trend_quality)
        regime_quality = regime_quality.where(~breakout_regime, 1.0)
        regime_quality = regime_quality.where(~shock_regime, 0.0)
        upward_context = (0.50 + 0.50 * trend_feature.fillna(0.0)).clip(0.0, 1.0)
        regime_quality = (
            regime_quality * (0.65 + 0.35 * upward_context)
        ).clip(0.0, 1.0)

        volume_completeness = valid_volume.astype(float).rolling(
            config.volatility_window,
            min_periods=1,
        ).mean()
        data_quality = (0.75 + 0.25 * volume_completeness).clip(0.0, 1.0)
        model_certainty = (
            1.0
            - lorentzian["lorentzian_uncertainty"]
            / (2.0 * expected_risk.replace(0.0, np.nan))
        ).clip(0.0, 1.0).fillna(0.0)
        confidence = (
            raw_score.clip(lower=0.0)
            * agreement
            * regime_quality
            * data_quality
            * (1.0 - correlation_penalty)
            * (0.60 + 0.40 * model_certainty)
        ).clip(0.0, 1.0)

        model_edge = lorentzian["lorentzian_expected_return"].clip(lower=0.0)
        technical_edge = raw_score.clip(lower=0.0) * expected_risk * 0.75
        expected_return = (
            0.65 * model_edge + 0.35 * technical_edge
        ).clip(lower=0.0)
        expected_return = pd.concat(
            [expected_return, 3.0 * expected_risk],
            axis=1,
        ).min(axis=1).fillna(0.0)
        estimated_cost = config.round_trip_cost_bps / 10_000.0
        net_edge = expected_return * confidence - estimated_cost
        opportunity_score = (
            net_edge / expected_risk.replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(-1.0)
        component_support = (components > 0.05).sum(axis=1)

        enough_history = np.arange(len(frame)) >= config.minimum_history - 1
        eligible = (
            enough_history
            & (lorentzian["lorentzian_samples"].to_numpy() >= config.lorentzian_minimum_samples)
            & (data_quality.to_numpy() >= config.minimum_data_quality)
            & ~shock_regime.fillna(False).to_numpy()
            & (choppiness.fillna(100.0).to_numpy() <= config.maximum_choppiness)
            & (lorentzian["lorentzian_score"].to_numpy() >= config.minimum_lorentzian_score)
            & (component_support >= config.minimum_component_support)
            & (raw_score.to_numpy() >= config.minimum_raw_score)
            & (confidence.to_numpy() >= config.minimum_confidence)
            & (net_edge.to_numpy() * 10_000.0 >= config.minimum_net_edge_bps)
            & (opportunity_score.to_numpy() >= config.minimum_opportunity_score)
            & ((squeeze_component.to_numpy() > 0.0) | (breakout_strength.to_numpy() > 0.0))
        )
        direction = np.where(eligible, Direction.LONG.value, Direction.FLAT.value)

        reason_code = np.full(len(frame), "LONG_BURST", dtype=object)
        for position in range(len(frame)):
            if eligible[position]:
                continue
            if not enough_history[position]:
                reason_code[position] = "INSUFFICIENT_HISTORY"
                regimes[position] = LongBurstRegime.INSUFFICIENT_DATA.value
            elif lorentzian.iloc[position]["lorentzian_samples"] < config.lorentzian_minimum_samples:
                reason_code[position] = "INSUFFICIENT_LABELS"
            elif data_quality.iloc[position] < config.minimum_data_quality:
                reason_code[position] = "DATA_QUALITY"
            elif bool(shock_regime.iloc[position]):
                reason_code[position] = "SHOCK_REGIME"
            elif float(choppiness.fillna(100.0).iloc[position]) > config.maximum_choppiness:
                reason_code[position] = "CHOPPY_REGIME"
            elif lorentzian.iloc[position]["lorentzian_score"] < config.minimum_lorentzian_score:
                reason_code[position] = "LORENTZIAN_NOT_BULLISH"
            elif component_support[position] < config.minimum_component_support:
                reason_code[position] = "COMPONENT_DISAGREEMENT"
            elif raw_score.iloc[position] < config.minimum_raw_score:
                reason_code[position] = "RAW_SCORE"
            elif confidence.iloc[position] < config.minimum_confidence:
                reason_code[position] = "CONFIDENCE"
            elif net_edge.iloc[position] * 10_000.0 < config.minimum_net_edge_bps:
                reason_code[position] = "NET_EDGE"
            elif opportunity_score.iloc[position] < config.minimum_opportunity_score:
                reason_code[position] = "OPPORTUNITY_SCORE"
            else:
                reason_code[position] = "NO_POSITIVE_TIMING"

        output = pd.DataFrame(
            {
                "ticker": str(ticker).upper().strip(),
                "direction": direction,
                "horizon_bars": config.forecast_horizon_bars,
                "regime": regimes,
                "choppiness": choppiness.fillna(100.0).clip(0.0, 100.0),
                "lorentzian_score": lorentzian["lorentzian_score"].clip(-1.0, 1.0),
                "squeeze_score": squeeze_component,
                "cmf_score": cmf_score.fillna(0.0).clip(-1.0, 1.0),
                "breakout_strength": breakout_strength,
                "price_acceleration": price_acceleration,
                "raw_score": raw_score.fillna(0.0),
                "agreement": agreement.fillna(0.0),
                "correlation_penalty": correlation_penalty.clip(0.0, 1.0),
                "regime_quality": regime_quality,
                "data_quality": data_quality,
                "confidence": confidence,
                "expected_return": expected_return,
                "expected_risk": expected_risk.fillna(0.0).clip(lower=0.0),
                "estimated_cost_return": estimated_cost,
                "net_edge": net_edge,
                "opportunity_score": opportunity_score,
                "lorentzian_samples": lorentzian["lorentzian_samples"].astype(int),
                "reason_code": reason_code,
                "squeeze_on": squeeze_on.astype(bool),
                "squeeze_release": squeeze_release.astype(bool),
                "cmf": cmf,
                "daily_volatility": daily_volatility.fillna(0.0),
                "long_volatility": long_volatility.fillna(0.0),
                "paper_only": True,
            },
            index=frame.index,
        )
        output.index.name = "timestamp"
        return output

    def analyze(self, data: pd.DataFrame, ticker: str) -> LongBurstSignal:
        """Restituisce la sola decisione più recente."""

        history = self.evaluate_history(data, ticker)
        row = history.iloc[-1]
        timestamp = pd.Timestamp(history.index[-1]).to_pydatetime()
        return LongBurstSignal(
            ticker=str(row["ticker"]),
            timestamp=timestamp,
            direction=Direction(str(row["direction"])),
            horizon_bars=int(row["horizon_bars"]),
            regime=LongBurstRegime(str(row["regime"])),
            choppiness=float(row["choppiness"]),
            lorentzian_score=float(row["lorentzian_score"]),
            squeeze_score=float(row["squeeze_score"]),
            cmf_score=float(row["cmf_score"]),
            breakout_strength=float(row["breakout_strength"]),
            price_acceleration=float(row["price_acceleration"]),
            raw_score=float(row["raw_score"]),
            agreement=float(row["agreement"]),
            correlation_penalty=float(row["correlation_penalty"]),
            regime_quality=float(row["regime_quality"]),
            data_quality=float(row["data_quality"]),
            confidence=float(row["confidence"]),
            expected_return=float(row["expected_return"]),
            expected_risk=float(row["expected_risk"]),
            estimated_cost_return=float(row["estimated_cost_return"]),
            net_edge=float(row["net_edge"]),
            opportunity_score=float(row["opportunity_score"]),
            lorentzian_samples=int(row["lorentzian_samples"]),
            reason_code=str(row["reason_code"]),
        )
