import pandas as pd

from strategies.base_strategy import BaseStrategy


class TrendFollowingStrategy(BaseStrategy):
    """
    Strategia trend-following basata su:

    - EMA veloce e lenta;
    - pendenza della EMA veloce;
    - ADX;
    - ATR;
    - RSI;
    - distanza minima tra le EMA.

    Supporta segnali BUY, SELL e WAIT.
    """

    name = "trend_following"

    def __init__(
        self,
        fast_ema_period=20,
        slow_ema_period=50,
        adx_period=14,
        atr_period=14,
        rsi_period=14,
        minimum_adx=20.0,
        minimum_atr_percent=0.0005,
        maximum_atr_percent=0.05,
        minimum_ema_distance_percent=0.0005,
        minimum_slope_percent=0.0001
    ):
        if fast_ema_period <= 0:
            raise ValueError(
                "fast_ema_period deve essere positivo."
            )

        if slow_ema_period <= fast_ema_period:
            raise ValueError(
                "slow_ema_period deve essere maggiore "
                "di fast_ema_period."
            )

        if adx_period <= 1:
            raise ValueError(
                "adx_period deve essere maggiore di 1."
            )

        if atr_period <= 1:
            raise ValueError(
                "atr_period deve essere maggiore di 1."
            )

        if rsi_period <= 1:
            raise ValueError(
                "rsi_period deve essere maggiore di 1."
            )

        if minimum_adx < 0:
            raise ValueError(
                "minimum_adx non può essere negativo."
            )

        if minimum_atr_percent < 0:
            raise ValueError(
                "minimum_atr_percent non può essere negativo."
            )

        if maximum_atr_percent <= minimum_atr_percent:
            raise ValueError(
                "maximum_atr_percent deve essere maggiore "
                "di minimum_atr_percent."
            )

        if minimum_ema_distance_percent < 0:
            raise ValueError(
                "minimum_ema_distance_percent non può essere negativo."
            )

        if minimum_slope_percent < 0:
            raise ValueError(
                "minimum_slope_percent non può essere negativo."
            )

        self.fast_ema_period = int(
            fast_ema_period
        )

        self.slow_ema_period = int(
            slow_ema_period
        )

        self.adx_period = int(
            adx_period
        )

        self.atr_period = int(
            atr_period
        )

        self.rsi_period = int(
            rsi_period
        )

        self.minimum_adx = float(
            minimum_adx
        )

        self.minimum_atr_percent = float(
            minimum_atr_percent
        )

        self.maximum_atr_percent = float(
            maximum_atr_percent
        )

        self.minimum_ema_distance_percent = float(
            minimum_ema_distance_percent
        )

        self.minimum_slope_percent = float(
            minimum_slope_percent
        )


    @staticmethod
    def calculate_true_range(
        data
    ):
        previous_close = data[
            "close"
        ].shift(1)

        high_low = (
            data["high"] -
            data["low"]
        )

        high_previous_close = (
            data["high"] -
            previous_close
        ).abs()

        low_previous_close = (
            data["low"] -
            previous_close
        ).abs()

        return pd.concat(
            [
                high_low,
                high_previous_close,
                low_previous_close
            ],
            axis=1
        ).max(
            axis=1
        )


    def calculate_atr(
        self,
        data
    ):
        true_range = self.calculate_true_range(
            data
        )

        return true_range.ewm(
            alpha=1 / self.atr_period,
            adjust=False,
            min_periods=self.atr_period
        ).mean()


    def calculate_rsi(
        self,
        close
    ):
        delta = close.diff()

        gains = delta.clip(
            lower=0
        )

        losses = -delta.clip(
            upper=0
        )

        average_gain = gains.ewm(
            alpha=1 / self.rsi_period,
            adjust=False,
            min_periods=self.rsi_period
        ).mean()

        average_loss = losses.ewm(
            alpha=1 / self.rsi_period,
            adjust=False,
            min_periods=self.rsi_period
        ).mean()

        relative_strength = (
            average_gain /
            average_loss.replace(
                0,
                float("nan")
            )
        )

        rsi = 100 - (
            100 /
            (
                1 +
                relative_strength
            )
        )

        rsi = rsi.where(
            average_loss != 0,
            100.0
        )

        rsi = rsi.where(
            average_gain != 0,
            0.0
        )

        return rsi


    def calculate_adx(
        self,
        data
    ):
        high_difference = data[
            "high"
        ].diff()

        low_difference = -data[
            "low"
        ].diff()

        positive_directional_movement = (
            high_difference.where(
                (
                    high_difference >
                    low_difference
                )
                &
                (
                    high_difference > 0
                ),
                0.0
            )
        )

        negative_directional_movement = (
            low_difference.where(
                (
                    low_difference >
                    high_difference
                )
                &
                (
                    low_difference > 0
                ),
                0.0
            )
        )

        true_range = self.calculate_true_range(
            data
        )

        smoothed_true_range = true_range.ewm(
            alpha=1 / self.adx_period,
            adjust=False,
            min_periods=self.adx_period
        ).mean()

        positive_dm_smoothed = (
            positive_directional_movement.ewm(
                alpha=1 / self.adx_period,
                adjust=False,
                min_periods=self.adx_period
            ).mean()
        )

        negative_dm_smoothed = (
            negative_directional_movement.ewm(
                alpha=1 / self.adx_period,
                adjust=False,
                min_periods=self.adx_period
            ).mean()
        )

        positive_di = (
            100 *
            positive_dm_smoothed /
            smoothed_true_range.replace(
                0,
                float("nan")
            )
        )

        negative_di = (
            100 *
            negative_dm_smoothed /
            smoothed_true_range.replace(
                0,
                float("nan")
            )
        )

        directional_index = (
            100 *
            (
                positive_di -
                negative_di
            ).abs()
            /
            (
                positive_di +
                negative_di
            ).replace(
                0,
                float("nan")
            )
        )

        return directional_index.ewm(
            alpha=1 / self.adx_period,
            adjust=False,
            min_periods=self.adx_period
        ).mean()


    def prepare_indicators(
        self,
        data
    ):
        prepared = self.validate_ohlcv_data(
            data
        )

        prepared["EMA_FAST"] = (
            prepared["close"].ewm(
                span=self.fast_ema_period,
                adjust=False
            ).mean()
        )

        prepared["EMA_SLOW"] = (
            prepared["close"].ewm(
                span=self.slow_ema_period,
                adjust=False
            ).mean()
        )

        prepared["EMA_FAST_SLOPE"] = (
            prepared["EMA_FAST"].diff(
                periods=3
            )
        )

        prepared["ATR"] = self.calculate_atr(
            prepared
        )

        prepared["ATR_PERCENT"] = (
            prepared["ATR"] /
            prepared["close"]
        )

        prepared["RSI"] = self.calculate_rsi(
            prepared["close"]
        )

        prepared["ADX"] = self.calculate_adx(
            prepared
        )

        prepared["EMA_DISTANCE_PERCENT"] = (
            (
                prepared["EMA_FAST"] -
                prepared["EMA_SLOW"]
            ).abs()
            /
            prepared["close"]
        )

        prepared["EMA_SLOPE_PERCENT"] = (
            prepared["EMA_FAST_SLOPE"].abs()
            /
            prepared["close"]
        )

        return prepared


    @staticmethod
    def clamp(
        value,
        minimum,
        maximum
    ):
        return max(
            minimum,
            min(
                maximum,
                value
            )
        )


    def calculate_confidence(
        self,
        latest
    ):
        adx = float(
            latest["ADX"]
        )

        ema_distance_percent = float(
            latest[
                "EMA_DISTANCE_PERCENT"
            ]
        )

        slope_percent = float(
            latest[
                "EMA_SLOPE_PERCENT"
            ]
        )

        adx_bonus = self.clamp(
            (
                adx -
                self.minimum_adx
            ) * 0.8,
            0,
            20
        )

        distance_bonus = self.clamp(
            ema_distance_percent * 2000,
            0,
            20
        )

        slope_bonus = self.clamp(
            slope_percent * 4000,
            0,
            10
        )

        confidence = (
            50 +
            adx_bonus +
            distance_bonus +
            slope_bonus
        )

        return round(
            self.clamp(
                confidence,
                0,
                100
            ),
            2
        )


    def generate_decision(
        self,
        data,
        ticker
    ):
        prepared = self.prepare_indicators(
            data
        )

        minimum_rows = max(
            self.slow_ema_period,
            self.adx_period * 2,
            self.atr_period,
            self.rsi_period
        ) + 5

        if len(prepared) < minimum_rows:
            return self.wait_decision(
                ticker=ticker,
                strategy=self.name,
                reasons=[
                    "Storico insufficiente"
                ]
            )

        latest = prepared.iloc[-1]

        required_values = [
            latest["EMA_FAST"],
            latest["EMA_SLOW"],
            latest["EMA_FAST_SLOPE"],
            latest["ATR"],
            latest["ATR_PERCENT"],
            latest["RSI"],
            latest["ADX"],
            latest["EMA_DISTANCE_PERCENT"],
            latest["EMA_SLOPE_PERCENT"]
        ]

        if any(
            pd.isna(value)
            for value in required_values
        ):
            return self.wait_decision(
                ticker=ticker,
                strategy=self.name,
                reasons=[
                    "Indicatori non disponibili"
                ]
            )

        close = float(
            latest["close"]
        )

        ema_fast = float(
            latest["EMA_FAST"]
        )

        ema_slow = float(
            latest["EMA_SLOW"]
        )

        ema_slope = float(
            latest["EMA_FAST_SLOPE"]
        )

        atr = float(
            latest["ATR"]
        )

        atr_percent = float(
            latest["ATR_PERCENT"]
        )

        rsi = float(
            latest["RSI"]
        )

        adx = float(
            latest["ADX"]
        )

        ema_distance_percent = float(
            latest[
                "EMA_DISTANCE_PERCENT"
            ]
        )

        slope_percent = float(
            latest[
                "EMA_SLOPE_PERCENT"
            ]
        )

        reasons = [
            f"ADX: {adx:.2f}",
            f"RSI: {rsi:.2f}",
            f"ATR%: {atr_percent * 100:.3f}%",
            (
                "Distanza EMA: "
                f"{ema_distance_percent * 100:.3f}%"
            )
        ]

        if atr_percent < self.minimum_atr_percent:
            reasons.append(
                "Volatilità troppo bassa"
            )

            return self.wait_decision(
                ticker=ticker,
                strategy=self.name,
                reasons=reasons,
                atr=atr
            )

        if atr_percent > self.maximum_atr_percent:
            reasons.append(
                "Volatilità troppo elevata"
            )

            return self.wait_decision(
                ticker=ticker,
                strategy=self.name,
                reasons=reasons,
                atr=atr
            )

        if adx < self.minimum_adx:
            reasons.append(
                "Trend non abbastanza forte"
            )

            return self.wait_decision(
                ticker=ticker,
                strategy=self.name,
                reasons=reasons,
                atr=atr
            )

        if (
            ema_distance_percent <
            self.minimum_ema_distance_percent
        ):
            reasons.append(
                "EMA troppo vicine"
            )

            return self.wait_decision(
                ticker=ticker,
                strategy=self.name,
                reasons=reasons,
                atr=atr
            )

        if (
            slope_percent <
            self.minimum_slope_percent
        ):
            reasons.append(
                "Pendenza EMA insufficiente"
            )

            return self.wait_decision(
                ticker=ticker,
                strategy=self.name,
                reasons=reasons,
                atr=atr
            )

        bullish_conditions = (
            ema_fast > ema_slow
            and ema_slope > 0
            and close > ema_fast
            and rsi >= 50
        )

        bearish_conditions = (
            ema_fast < ema_slow
            and ema_slope < 0
            and close < ema_fast
            and rsi <= 50
        )

        if bullish_conditions:
            reasons.extend(
                [
                    "EMA veloce sopra EMA lenta",
                    "Pendenza EMA positiva",
                    "Prezzo sopra EMA veloce",
                    "Trend rialzista confermato"
                ]
            )

            return {
                "ticker": ticker,
                "strategy": self.name,
                "action": "BUY",
                "confidence": (
                    self.calculate_confidence(
                        latest
                    )
                ),
                "atr": atr,
                "reasons": reasons
            }

        if bearish_conditions:
            reasons.extend(
                [
                    "EMA veloce sotto EMA lenta",
                    "Pendenza EMA negativa",
                    "Prezzo sotto EMA veloce",
                    "Trend ribassista confermato"
                ]
            )

            return {
                "ticker": ticker,
                "strategy": self.name,
                "action": "SELL",
                "confidence": (
                    self.calculate_confidence(
                        latest
                    )
                ),
                "atr": atr,
                "reasons": reasons
            }

        reasons.append(
            "Condizioni di entrata incomplete"
        )

        return self.wait_decision(
            ticker=ticker,
            strategy=self.name,
            reasons=reasons,
            atr=atr
        )