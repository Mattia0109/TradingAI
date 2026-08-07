"""Feature engine deterministico e privo di look-ahead."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from adaptive.models import FeatureSnapshot


@dataclass(frozen=True)
class FeatureEngineConfig:
    annualization_factor: int = 252
    short_volatility_span: int = 20
    long_volatility_span: int = 60
    trend_window: int = 63
    zscore_window: int = 63
    maximum_liquid_spread_bps: float = 50.0

    def __post_init__(self) -> None:
        for name in (
            "annualization_factor",
            "short_volatility_span",
            "long_volatility_span",
            "trend_window",
            "zscore_window",
        ):
            if int(getattr(self, name)) <= 1:
                raise ValueError(f"{name} deve essere maggiore di 1.")
        if self.maximum_liquid_spread_bps <= 0.0:
            raise ValueError("maximum_liquid_spread_bps deve essere positivo.")

    @property
    def minimum_history(self) -> int:
        return max(
            self.long_volatility_span + 1,
            self.trend_window,
            self.zscore_window + 5,
            22,
        )


class AdaptiveFeatureEngine:
    """Converte una serie OHLCV in una fotografia quantitativa compatta."""

    def __init__(self, config: FeatureEngineConfig | None = None) -> None:
        self.config = config or FeatureEngineConfig()

    @staticmethod
    def _prepare_data(data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data deve essere un pandas DataFrame.")
        if data.empty:
            raise ValueError("data non può essere vuoto.")

        prepared = data.copy()
        prepared.columns = [str(column).lower().strip() for column in prepared.columns]
        if "close" not in prepared.columns:
            raise ValueError("Manca la colonna close.")

        prepared["close"] = pd.to_numeric(prepared["close"], errors="coerce")
        prepared = prepared.dropna(subset=["close"])
        prepared = prepared[prepared["close"] > 0.0]
        if "date" in prepared.columns:
            prepared["date"] = pd.to_datetime(
                prepared["date"], errors="coerce", utc=True
            )
            prepared = prepared.dropna(subset=["date"])
            prepared = prepared.sort_values("date")
            prepared = prepared.drop_duplicates(subset=["date"], keep="last")
        else:
            prepared = prepared.sort_index()

        for optional in ("volume", "bid", "ask", "spread_bps"):
            if optional in prepared.columns:
                prepared[optional] = pd.to_numeric(
                    prepared[optional], errors="coerce"
                )

        if prepared.empty:
            raise ValueError("Nessun prezzo valido disponibile.")
        return prepared.reset_index(drop=True)

    @staticmethod
    def _safe_standard_deviation(values: pd.Series) -> float:
        cleaned = pd.to_numeric(values, errors="coerce").dropna()
        if len(cleaned) < 2:
            return 0.0
        deviation = float(cleaned.std(ddof=1))
        return deviation if math.isfinite(deviation) else 0.0

    def _annualized_volatility(self, returns: pd.Series, span: int) -> float:
        estimate = (
            returns.ewm(span=span, adjust=False, min_periods=span).std().iloc[-1]
        )
        if pd.isna(estimate) or float(estimate) <= 0.0:
            estimate = self._safe_standard_deviation(returns.tail(span))
        return max(0.0, float(estimate)) * math.sqrt(
            self.config.annualization_factor
        )

    def _trend_score(self, close: pd.Series) -> float:
        window = min(self.config.trend_window, len(close))
        log_prices = np.log(np.asarray(close.tail(window), dtype=float))
        time_axis = np.arange(window, dtype=float)
        time_axis -= time_axis.mean()
        denominator = float(np.dot(time_axis, time_axis))
        if denominator <= 0.0:
            return 0.0
        slope = float(np.dot(time_axis, log_prices - log_prices.mean()) / denominator)
        residual_daily_volatility = self._safe_standard_deviation(
            pd.Series(np.diff(log_prices))
        )
        if residual_daily_volatility <= 0.0:
            return float(np.sign(slope)) if slope != 0.0 else 0.0
        normalized = slope * math.sqrt(window) / residual_daily_volatility
        return float(np.tanh(normalized))

    def _return_zscore(self, close: pd.Series) -> float:
        five_bar_returns = np.log(close / close.shift(5)).dropna()
        reference = five_bar_returns.tail(self.config.zscore_window)
        if reference.empty:
            return 0.0
        deviation = self._safe_standard_deviation(reference)
        if deviation <= 0.0:
            return 0.0
        return float((reference.iloc[-1] - reference.mean()) / deviation)

    @staticmethod
    def _volume_zscore(prepared: pd.DataFrame, window: int) -> tuple[float, bool]:
        if "volume" not in prepared.columns:
            return 0.0, False
        volume = prepared["volume"].dropna().tail(window)
        if len(volume) < 2:
            return 0.0, False
        transformed = np.log1p(volume.clip(lower=0.0))
        deviation = float(transformed.std(ddof=1))
        if not math.isfinite(deviation) or deviation <= 0.0:
            return 0.0, True
        return float((transformed.iloc[-1] - transformed.mean()) / deviation), True

    @staticmethod
    def _spread_bps(prepared: pd.DataFrame) -> float | None:
        latest = prepared.iloc[-1]
        if "spread_bps" in prepared.columns and pd.notna(latest["spread_bps"]):
            return max(0.0, float(latest["spread_bps"]))
        if {"bid", "ask"}.issubset(prepared.columns):
            bid = latest["bid"]
            ask = latest["ask"]
            if pd.notna(bid) and pd.notna(ask) and float(ask) >= float(bid) > 0.0:
                midpoint = (float(ask) + float(bid)) / 2.0
                return (float(ask) - float(bid)) / midpoint * 10_000.0
        return None

    def _liquidity_score(
        self, spread_bps: float | None, volume_zscore: float, has_volume: bool
    ) -> float:
        spread_score = 0.65
        if spread_bps is not None:
            spread_score = max(
                0.0,
                1.0 - spread_bps / self.config.maximum_liquid_spread_bps,
            )
        volume_score = 0.5
        if has_volume:
            volume_score = 1.0 / (1.0 + math.exp(-volume_zscore))
        return float(np.clip(0.75 * spread_score + 0.25 * volume_score, 0.0, 1.0))

    def _market_correlation(
        self, log_returns: pd.Series, market_returns: pd.Series | None
    ) -> float | None:
        if market_returns is None:
            return None
        reference = pd.to_numeric(market_returns, errors="coerce")
        aligned = pd.concat(
            [log_returns.rename("asset"), reference.rename("market")], axis=1
        ).dropna()
        aligned = aligned.tail(self.config.long_volatility_span)
        if len(aligned) < 20:
            return None
        correlation = float(aligned["asset"].corr(aligned["market"]))
        return correlation if math.isfinite(correlation) else None

    @staticmethod
    def _timestamp(prepared: pd.DataFrame, fallback: datetime | None) -> datetime:
        if fallback is not None:
            if fallback.tzinfo is None:
                return fallback.replace(tzinfo=timezone.utc)
            return fallback.astimezone(timezone.utc)
        if "date" in prepared.columns:
            value = prepared.iloc[-1]["date"]
            if isinstance(value, pd.Timestamp):
                if value.tzinfo is None:
                    value = value.tz_localize("UTC")
                else:
                    value = value.tz_convert("UTC")
                return value.to_pydatetime()
        return datetime.now(timezone.utc)

    def build(
        self,
        data: pd.DataFrame,
        ticker: str,
        asset_class: str = "UNKNOWN",
        *,
        market_returns: pd.Series | None = None,
        news_sentiment: float = 0.0,
        news_relevance: float = 0.0,
        timestamp: datetime | None = None,
    ) -> FeatureSnapshot:
        prepared = self._prepare_data(data)
        if len(prepared) < self.config.minimum_history:
            raise ValueError(
                "Storico insufficiente: servono almeno "
                f"{self.config.minimum_history} osservazioni."
            )

        close = prepared["close"].astype(float)
        log_returns = np.log(close / close.shift(1)).dropna()
        return_1 = float(log_returns.iloc[-1])
        return_5 = float(log_returns.tail(5).sum())
        return_21 = float(log_returns.tail(21).sum())
        short_volatility = self._annualized_volatility(
            log_returns, self.config.short_volatility_span
        )
        long_volatility = self._annualized_volatility(
            log_returns, self.config.long_volatility_span
        )
        daily_volatility = short_volatility / math.sqrt(
            self.config.annualization_factor
        )
        shock_score = abs(return_1) / daily_volatility if daily_volatility > 0.0 else 0.0
        volume_zscore, has_volume = self._volume_zscore(
            prepared, self.config.long_volatility_span
        )
        spread_bps = self._spread_bps(prepared)
        liquidity_score = self._liquidity_score(
            spread_bps, volume_zscore, has_volume
        )
        market_correlation = self._market_correlation(log_returns, market_returns)

        data_quality = 1.0
        if not has_volume:
            data_quality -= 0.05
        if spread_bps is None:
            data_quality -= 0.10
        if market_correlation is None:
            data_quality -= 0.05

        return FeatureSnapshot(
            ticker=ticker,
            asset_class=asset_class,
            timestamp=self._timestamp(prepared, timestamp),
            price=float(close.iloc[-1]),
            return_1=return_1,
            return_5=return_5,
            return_21=return_21,
            realized_volatility_short=short_volatility,
            realized_volatility_long=long_volatility,
            trend_score=self._trend_score(close),
            return_zscore=self._return_zscore(close),
            shock_score=shock_score,
            volume_zscore=volume_zscore,
            spread_bps=spread_bps,
            liquidity_score=liquidity_score,
            market_correlation=market_correlation,
            news_sentiment=news_sentiment,
            news_relevance=news_relevance,
            data_quality=max(0.0, data_quality),
        )
