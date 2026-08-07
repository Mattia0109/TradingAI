import math

import numpy as np
import pandas as pd


class MultiAssetTimeSeriesMomentumStrategy:
    """
    Strategia multi-asset Time-Series Momentum.

    Il segnale combina più orizzonti temporali.
    I componenti possono usare rendimenti
    normalizzati oppure trend lineari significativi
    con errore standard Newey-West.

    Questa classe genera target direzionali.
    Non apre direttamente operazioni.
    """

    name = "multi_asset_tsmom"

    DEFAULT_LOOKBACK_WEIGHTS = {
        21: 0.10,
        63: 0.20,
        126: 0.30,
        252: 0.40
    }

    VALID_ACTIONS = {
        "LONG",
        "SHORT",
        "FLAT"
    }

    def __init__(
        self,
        lookback_weights=None,
        volatility_span=60,
        annualization_factor=252,
        no_trade_threshold=0.15,
        component_clip=1.0,
        minimum_annualized_volatility=0.01,
        maximum_annualized_volatility=3.0,
        minimum_history=None,
        signal_sizing="continuous",
        component_method="return",
        trend_significance_threshold=2.0
    ):
        selected_weights = (
            dict(lookback_weights)
            if lookback_weights is not None
            else dict(
                self.DEFAULT_LOOKBACK_WEIGHTS
            )
        )

        if not selected_weights:
            raise ValueError(
                "lookback_weights non può essere vuoto."
            )

        normalized_weights = {}

        for lookback, weight in (
            selected_weights.items()
        ):
            normalized_lookback = int(
                lookback
            )

            normalized_weight = float(
                weight
            )

            if normalized_lookback <= 1:
                raise ValueError(
                    "Ogni lookback deve essere "
                    "maggiore di uno."
                )

            if normalized_weight <= 0:
                raise ValueError(
                    "Ogni peso deve essere positivo."
                )

            normalized_weights[
                normalized_lookback
            ] = normalized_weight

        total_weight = sum(
            normalized_weights.values()
        )

        self.lookback_weights = {
            lookback: weight / total_weight
            for lookback, weight
            in normalized_weights.items()
        }

        if volatility_span <= 1:
            raise ValueError(
                "volatility_span deve essere "
                "maggiore di uno."
            )

        if annualization_factor <= 0:
            raise ValueError(
                "annualization_factor deve essere positivo."
            )

        if not 0 <= no_trade_threshold < 1:
            raise ValueError(
                "no_trade_threshold deve essere "
                "compreso tra 0 e 1 escluso."
            )

        if component_clip <= 0:
            raise ValueError(
                "component_clip deve essere positivo."
            )

        if minimum_annualized_volatility <= 0:
            raise ValueError(
                "minimum_annualized_volatility "
                "deve essere positiva."
            )

        if (
            maximum_annualized_volatility
            <= minimum_annualized_volatility
        ):
            raise ValueError(
                "maximum_annualized_volatility deve "
                "superare il minimo."
            )

        normalized_signal_sizing = (
            str(signal_sizing)
            .lower()
            .strip()
        )

        if normalized_signal_sizing not in {
            "continuous",
            "directional"
        }:
            raise ValueError(
                "signal_sizing deve essere continuous "
                "oppure directional."
            )

        normalized_component_method = (
            str(component_method)
            .lower()
            .strip()
        )

        if normalized_component_method not in {
            "return",
            "linear_trend"
        }:
            raise ValueError(
                "component_method deve essere return "
                "oppure linear_trend."
            )

        trend_significance_threshold = float(
            trend_significance_threshold
        )

        if (
            not math.isfinite(
                trend_significance_threshold
            )
            or trend_significance_threshold <= 0
        ):
            raise ValueError(
                "trend_significance_threshold deve essere "
                "positivo e finito."
            )

        calculated_minimum_history = max(
            max(
                self.lookback_weights
            ) + 1,
            int(
                volatility_span
            ) + 2
        )

        if minimum_history is None:
            minimum_history = (
                calculated_minimum_history
            )

        if (
            int(minimum_history)
            <
            calculated_minimum_history
        ):
            raise ValueError(
                "minimum_history è insufficiente per "
                "i lookback e la volatilità configurati."
            )

        self.volatility_span = int(
            volatility_span
        )

        self.annualization_factor = int(
            annualization_factor
        )

        self.no_trade_threshold = float(
            no_trade_threshold
        )

        self.component_clip = float(
            component_clip
        )

        self.minimum_annualized_volatility = float(
            minimum_annualized_volatility
        )

        self.maximum_annualized_volatility = float(
            maximum_annualized_volatility
        )

        self.signal_sizing = (
            normalized_signal_sizing
        )

        self.component_method = (
            normalized_component_method
        )

        self.trend_significance_threshold = (
            trend_significance_threshold
        )

        self.minimum_history = int(
            minimum_history
        )


    @staticmethod
    def normalize_ticker(
        ticker
    ):
        normalized = (
            str(ticker)
            .upper()
            .strip()
        )

        if not normalized:
            raise ValueError(
                "ticker non può essere vuoto."
            )

        return normalized


    @staticmethod
    def prepare_data(
        data
    ):
        if not isinstance(
            data,
            pd.DataFrame
        ):
            raise TypeError(
                "data deve essere un pandas DataFrame."
            )

        if data.empty:
            raise ValueError(
                "Il DataFrame è vuoto."
            )

        normalized = data.copy()

        normalized.columns = [
            str(column).lower().strip()
            for column in normalized.columns
        ]

        if "close" not in normalized.columns:
            raise ValueError(
                "Manca la colonna close."
            )

        if "date" not in normalized.columns:
            normalized["date"] = (
                normalized.index
            )

        normalized["date"] = pd.to_datetime(
            normalized["date"],
            errors="coerce",
            utc=True
        )

        normalized["close"] = pd.to_numeric(
            normalized["close"],
            errors="coerce"
        )

        normalized = normalized.dropna(
            subset=[
                "date",
                "close"
            ]
        )

        normalized = normalized[
            normalized["close"] > 0
        ]

        normalized = normalized.sort_values(
            "date"
        )

        normalized = normalized.drop_duplicates(
            subset=["date"],
            keep="last"
        )

        normalized["date"] = (
            normalized["date"]
            .dt
            .tz_convert(None)
        )

        normalized = normalized.reset_index(
            drop=True
        )

        if normalized.empty:
            raise ValueError(
                "Nessun prezzo valido disponibile."
            )

        return normalized


    def calculate_annualized_volatility(
        self,
        close_prices
    ):
        log_returns = np.log(
            close_prices
            /
            close_prices.shift(1)
        )

        daily_volatility = (
            log_returns
            .ewm(
                span=self.volatility_span,
                adjust=False,
                min_periods=self.volatility_span
            )
            .std()
            .iloc[-1]
        )

        if (
            pd.isna(daily_volatility)
            or not math.isfinite(
                float(daily_volatility)
            )
            or float(daily_volatility) <= 0
        ):
            return None

        annualized = (
            float(daily_volatility)
            *
            math.sqrt(
                self.annualization_factor
            )
        )

        return min(
            max(
                annualized,
                self.minimum_annualized_volatility
            ),
            self.maximum_annualized_volatility
        )


    def calculate_component(
        self,
        close_prices,
        lookback,
        annualized_volatility
    ):
        current_price = float(
            close_prices.iloc[-1]
        )

        previous_price = float(
            close_prices.iloc[
                -lookback - 1
            ]
        )

        logarithmic_return = math.log(
            current_price
            /
            previous_price
        )

        daily_volatility = (
            annualized_volatility
            /
            math.sqrt(
                self.annualization_factor
            )
        )

        expected_volatility = (
            daily_volatility
            *
            math.sqrt(
                lookback
            )
        )

        if expected_volatility <= 0:
            return 0.0

        normalized_momentum = (
            logarithmic_return
            /
            expected_volatility
        )

        clipped = float(
            np.clip(
                normalized_momentum,
                -self.component_clip,
                self.component_clip
            )
        )

        return (
            clipped
            /
            self.component_clip
        )


    @staticmethod
    def calculate_linear_trend_t_statistic(
        close_prices,
        lookback
    ):
        price_path = np.asarray(
            close_prices.iloc[
                -lookback - 1:
            ],
            dtype=float
        )

        observation_count = len(
            price_path
        )

        if observation_count < 3:
            return 0.0

        normalized_path = (
            price_path
            /
            price_path[0]
        )

        time_axis = np.arange(
            observation_count,
            dtype=float
        )

        time_axis -= float(
            time_axis.mean()
        )

        design = np.column_stack(
            [
                np.ones(
                    observation_count,
                    dtype=float
                ),
                time_axis
            ]
        )

        inverse_information = np.linalg.pinv(
            design.T
            @
            design
        )

        coefficients = (
            inverse_information
            @
            design.T
            @
            normalized_path
        )

        residuals = (
            normalized_path
            -
            design
            @
            coefficients
        )

        scores = (
            design
            *
            residuals[:, None]
        )

        covariance_meat = (
            scores.T
            @
            scores
        )

        maximum_lag = int(
            math.floor(
                4
                *
                (
                    observation_count
                    /
                    100
                ) ** (
                    2
                    /
                    9
                )
            )
        )

        maximum_lag = min(
            max(
                maximum_lag,
                1
            ),
            observation_count - 1
        )

        for lag in range(
            1,
            maximum_lag + 1
        ):
            weight = (
                1
                -
                lag
                /
                (
                    maximum_lag
                    +
                    1
                )
            )

            lagged_covariance = (
                scores[lag:].T
                @
                scores[:-lag]
            )

            covariance_meat += (
                weight
                *
                (
                    lagged_covariance
                    +
                    lagged_covariance.T
                )
            )

        covariance = (
            inverse_information
            @
            covariance_meat
            @
            inverse_information
        )

        covariance *= (
            observation_count
            /
            (
                observation_count
                -
                design.shape[1]
            )
        )

        slope = float(
            coefficients[1]
        )

        slope_variance = max(
            float(
                covariance[1, 1]
            ),
            0.0
        )

        if slope_variance <= np.finfo(float).eps:
            if abs(slope) <= np.finfo(float).eps:
                return 0.0

            return float(
                math.copysign(
                    100.0,
                    slope
                )
            )

        statistic = (
            slope
            /
            math.sqrt(
                slope_variance
            )
        )

        return float(
            np.clip(
                statistic,
                -100.0,
                100.0
            )
        )


    def build_flat_result(
        self,
        ticker,
        asset_class,
        reason,
        as_of=None,
        annualized_volatility=None,
        history_ready=False
    ):
        return {
            "ticker": ticker,
            "asset_class": asset_class,
            "strategy": self.name,
            "signal_sizing": self.signal_sizing,
            "component_method": self.component_method,
            "action": "FLAT",
            "signal": 0.0,
            "strength": 0.0,
            "annualized_volatility": (
                annualized_volatility
            ),
            "components": {},
            "history_ready": bool(
                history_ready
            ),
            "as_of": as_of,
            "reasons": [
                str(reason)
            ]
        }


    def generate_signal(
        self,
        data,
        ticker,
        asset_class="UNKNOWN"
    ):
        normalized_ticker = (
            self.normalize_ticker(
                ticker
            )
        )

        normalized_asset_class = (
            str(asset_class)
            .upper()
            .strip()
            or "UNKNOWN"
        )

        prepared = self.prepare_data(
            data
        )

        as_of = prepared.iloc[-1][
            "date"
        ]

        if len(prepared) < self.minimum_history:
            return self.build_flat_result(
                ticker=normalized_ticker,
                asset_class=(
                    normalized_asset_class
                ),
                reason=(
                    "Storico insufficiente per "
                    "il segnale TSMOM."
                ),
                as_of=as_of,
                history_ready=False
            )

        close_prices = prepared[
            "close"
        ].astype(
            float
        )

        annualized_volatility = (
            self.calculate_annualized_volatility(
                close_prices
            )
        )

        if annualized_volatility is None:
            return self.build_flat_result(
                ticker=normalized_ticker,
                asset_class=(
                    normalized_asset_class
                ),
                reason=(
                    "Volatilità non disponibile "
                    "o nulla."
                ),
                as_of=as_of,
                history_ready=True
            )

        components = {}
        weighted_signal = 0.0

        for lookback, weight in (
            sorted(
                self.lookback_weights.items()
            )
        ):
            trend_t_statistic = None

            if (
                self.component_method
                ==
                "linear_trend"
            ):
                trend_t_statistic = (
                    self.calculate_linear_trend_t_statistic(
                        close_prices=(
                            close_prices
                        ),
                        lookback=lookback
                    )
                )

                if (
                    trend_t_statistic
                    >
                    self.trend_significance_threshold
                ):
                    component = 1.0

                elif (
                    trend_t_statistic
                    <
                    -self.trend_significance_threshold
                ):
                    component = -1.0

                else:
                    component = 0.0

            else:
                component = self.calculate_component(
                    close_prices=close_prices,
                    lookback=lookback,
                    annualized_volatility=(
                        annualized_volatility
                    )
                )

            components[
                str(lookback)
            ] = {
                "lookback": lookback,
                "weight": weight,
                "normalized_momentum": (
                    component
                ),
                "trend_t_statistic": (
                    trend_t_statistic
                ),
                "weighted_contribution": (
                    component
                    *
                    weight
                )
            }

            weighted_signal += (
                component
                *
                weight
            )

        weighted_signal = float(
            np.clip(
                weighted_signal,
                -1.0,
                1.0
            )
        )

        absolute_signal = abs(
            weighted_signal
        )

        if (
            absolute_signal
            <
            self.no_trade_threshold
        ):
            action = "FLAT"
            effective_signal = 0.0

            reasons = [
                (
                    "Segnale inferiore alla "
                    "no-trade band."
                )
            ]

        elif weighted_signal > 0:
            action = "LONG"
            effective_signal = 1.0

            reasons = [
                (
                    "Momentum aggregato positivo "
                    "su più orizzonti."
                )
            ]

        else:
            action = "SHORT"
            effective_signal = -1.0

            reasons = [
                (
                    "Momentum aggregato negativo "
                    "su più orizzonti."
                )
            ]

        if (
            action != "FLAT"
            and
            self.signal_sizing
            ==
            "continuous"
        ):
            effective_signal = (
                weighted_signal
            )

        if action == "FLAT":
            strength = 0.0

        else:
            strength = (
                absolute_signal
                -
                self.no_trade_threshold
            ) / (
                1.0
                -
                self.no_trade_threshold
            )

            strength = float(
                np.clip(
                    strength,
                    0.0,
                    1.0
                )
            )

        return {
            "ticker": normalized_ticker,
            "asset_class": (
                normalized_asset_class
            ),
            "strategy": self.name,
            "signal_sizing": self.signal_sizing,
            "component_method": self.component_method,
            "action": action,
            "signal": float(
                effective_signal
            ),
            "raw_signal": float(
                weighted_signal
            ),
            "strength": strength,
            "annualized_volatility": float(
                annualized_volatility
            ),
            "components": components,
            "history_ready": True,
            "as_of": as_of,
            "reasons": reasons
        }


    def generate_signals(
        self,
        market_data,
        asset_classes=None
    ):
        if not isinstance(
            market_data,
            dict
        ):
            raise TypeError(
                "market_data deve essere un dizionario "
                "ticker -> DataFrame."
            )

        if not market_data:
            raise ValueError(
                "market_data non può essere vuoto."
            )

        if asset_classes is None:
            asset_classes = {}

        if not isinstance(
            asset_classes,
            dict
        ):
            raise TypeError(
                "asset_classes deve essere un dizionario."
            )

        results = {}

        for ticker in sorted(
            market_data
        ):
            results[
                self.normalize_ticker(
                    ticker
                )
            ] = self.generate_signal(
                data=market_data[ticker],
                ticker=ticker,
                asset_class=asset_classes.get(
                    ticker,
                    asset_classes.get(
                        self.normalize_ticker(
                            ticker
                        ),
                        "UNKNOWN"
                    )
                )
            )

        return results
