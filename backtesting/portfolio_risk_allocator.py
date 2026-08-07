import math

import numpy as np


class PortfolioRiskAllocator:
    """
    Converte i segnali TSMOM in pesi target.

    Il peso rappresenta la quota del capitale
    nozionale assegnata allo strumento.

    Funzioni principali:

    - inverse-volatility allocation;
    - volatility targeting;
    - limite di esposizione lorda;
    - limite per singolo asset;
    - limite per asset class;
    - no-trade turnover buffer;
    - drawdown governor;
    - portfolio kill switch.
    """

    def __init__(
        self,
        target_portfolio_volatility=0.10,
        max_gross_exposure=1.0,
        max_asset_weight=0.10,
        max_asset_class_weight=0.30,
        minimum_trade_weight=0.005,
        turnover_buffer=0.005,
        drawdown_start=0.05,
        drawdown_medium=0.10,
        drawdown_severe=0.15,
        drawdown_kill_switch=0.20,
        minimum_drawdown_multiplier=0.25
    ):
        if target_portfolio_volatility <= 0:
            raise ValueError(
                "target_portfolio_volatility "
                "deve essere positiva."
            )

        if not 0 < max_gross_exposure <= 1:
            raise ValueError(
                "max_gross_exposure deve essere "
                "compresa tra 0 e 1."
            )

        if not 0 < max_asset_weight <= 1:
            raise ValueError(
                "max_asset_weight deve essere "
                "compreso tra 0 e 1."
            )

        if not 0 < max_asset_class_weight <= 1:
            raise ValueError(
                "max_asset_class_weight deve essere "
                "compreso tra 0 e 1."
            )

        if minimum_trade_weight < 0:
            raise ValueError(
                "minimum_trade_weight non può essere negativo."
            )

        if turnover_buffer < 0:
            raise ValueError(
                "turnover_buffer non può essere negativo."
            )

        minimum_drawdown_multiplier = float(
            minimum_drawdown_multiplier
        )

        if (
            not math.isfinite(
                minimum_drawdown_multiplier
            )
            or
            not 0 < minimum_drawdown_multiplier <= 0.50
        ):
            raise ValueError(
                "minimum_drawdown_multiplier deve essere "
                "compreso tra 0 e 0.50."
            )

        drawdown_levels = [
            float(drawdown_start),
            float(drawdown_medium),
            float(drawdown_severe),
            float(drawdown_kill_switch)
        ]

        if any(
            level < 0
            for level in drawdown_levels
        ):
            raise ValueError(
                "Le soglie di drawdown non possono "
                "essere negative."
            )

        if drawdown_levels != sorted(
            drawdown_levels
        ):
            raise ValueError(
                "Le soglie di drawdown devono "
                "essere crescenti."
            )

        if (
            drawdown_start
            ==
            drawdown_kill_switch
        ):
            raise ValueError(
                "Le soglie di drawdown devono "
                "essere distinte."
            )

        self.target_portfolio_volatility = float(
            target_portfolio_volatility
        )

        self.max_gross_exposure = float(
            max_gross_exposure
        )

        self.max_asset_weight = float(
            max_asset_weight
        )

        self.max_asset_class_weight = float(
            max_asset_class_weight
        )

        self.minimum_trade_weight = float(
            minimum_trade_weight
        )

        self.turnover_buffer = float(
            turnover_buffer
        )

        self.drawdown_start = float(
            drawdown_start
        )

        self.drawdown_medium = float(
            drawdown_medium
        )

        self.drawdown_severe = float(
            drawdown_severe
        )

        self.drawdown_kill_switch = float(
            drawdown_kill_switch
        )

        self.minimum_drawdown_multiplier = (
            minimum_drawdown_multiplier
        )


    @staticmethod
    def normalize_previous_weights(
        previous_weights
    ):
        if previous_weights is None:
            return {}

        if not isinstance(
            previous_weights,
            dict
        ):
            raise TypeError(
                "previous_weights deve essere "
                "un dizionario."
            )

        normalized = {}

        for ticker, weight in (
            previous_weights.items()
        ):
            normalized_ticker = (
                str(ticker)
                .upper()
                .strip()
            )

            normalized_weight = float(
                weight
            )

            if not normalized_ticker:
                raise ValueError(
                    "Ticker vuoto in previous_weights."
                )

            if not math.isfinite(
                normalized_weight
            ):
                raise ValueError(
                    "Peso precedente non finito."
                )

            normalized[
                normalized_ticker
            ] = normalized_weight

        return normalized


    @staticmethod
    def normalize_signals(
        signals
    ):
        if isinstance(
            signals,
            dict
        ):
            source = list(
                signals.values()
            )

        elif isinstance(
            signals,
            list
        ):
            source = list(
                signals
            )

        else:
            raise TypeError(
                "signals deve essere un dizionario "
                "o una lista."
            )

        normalized = []

        for signal in source:
            if not isinstance(
                signal,
                dict
            ):
                raise TypeError(
                    "Ogni segnale deve essere "
                    "un dizionario."
                )

            ticker = (
                str(
                    signal.get(
                        "ticker",
                        ""
                    )
                )
                .upper()
                .strip()
            )

            if not ticker:
                raise ValueError(
                    "Ogni segnale deve avere un ticker."
                )

            normalized.append(
                {
                    **signal,
                    "ticker": ticker,
                    "asset_class": (
                        str(
                            signal.get(
                                "asset_class",
                                "UNKNOWN"
                            )
                        )
                        .upper()
                        .strip()
                        or "UNKNOWN"
                    )
                }
            )

        return normalized


    def calculate_drawdown_multiplier(
        self,
        current_drawdown
    ):
        drawdown = abs(
            float(current_drawdown)
        )

        if not math.isfinite(
            drawdown
        ):
            raise ValueError(
                "current_drawdown deve essere finito."
            )

        if (
            drawdown
            >=
            self.drawdown_kill_switch
        ):
            return 0.0

        if drawdown <= self.drawdown_start:
            return 1.0

        if drawdown <= self.drawdown_medium:
            progress = (
                drawdown
                -
                self.drawdown_start
            ) / (
                self.drawdown_medium
                -
                self.drawdown_start
            )

            return (
                1.0
                -
                0.25
                *
                progress
            )

        if drawdown <= self.drawdown_severe:
            progress = (
                drawdown
                -
                self.drawdown_medium
            ) / (
                self.drawdown_severe
                -
                self.drawdown_medium
            )

            return (
                0.75
                -
                0.25
                *
                progress
            )

        progress = (
            drawdown
            -
            self.drawdown_severe
        ) / (
            self.drawdown_kill_switch
            -
            self.drawdown_severe
        )

        return max(
            self.minimum_drawdown_multiplier,
            0.50
            -
            (
                0.50
                -
                self.minimum_drawdown_multiplier
            )
            *
            progress
        )


    @staticmethod
    def calculate_gross_exposure(
        weights
    ):
        return sum(
            abs(
                float(weight)
            )
            for weight in weights.values()
        )


    @staticmethod
    def calculate_net_exposure(
        weights
    ):
        return sum(
            float(weight)
            for weight in weights.values()
        )


    @staticmethod
    def calculate_class_exposures(
        weights,
        asset_classes
    ):
        exposures = {}

        for ticker, weight in (
            weights.items()
        ):
            asset_class = (
                asset_classes.get(
                    ticker,
                    "UNKNOWN"
                )
            )

            exposures[
                asset_class
            ] = (
                exposures.get(
                    asset_class,
                    0.0
                )
                +
                abs(
                    float(weight)
                )
            )

        return exposures


    @staticmethod
    def estimate_independent_volatility(
        weights,
        volatilities
    ):
        variance = sum(
            (
                float(
                    weights.get(
                        ticker,
                        0.0
                    )
                )
                *
                float(volatility)
            ) ** 2
            for ticker, volatility
            in volatilities.items()
        )

        return math.sqrt(
            max(
                variance,
                0.0
            )
        )


    @staticmethod
    def scale_weights(
        weights,
        multiplier
    ):
        return {
            ticker: float(weight)
            *
            float(multiplier)
            for ticker, weight
            in weights.items()
        }


    def enforce_asset_cap(
        self,
        weights
    ):
        return {
            ticker: float(
                np.clip(
                    weight,
                    -self.max_asset_weight,
                    self.max_asset_weight
                )
            )
            for ticker, weight
            in weights.items()
        }


    def enforce_class_cap(
        self,
        weights,
        asset_classes
    ):
        adjusted = dict(
            weights
        )

        exposures = (
            self.calculate_class_exposures(
                adjusted,
                asset_classes
            )
        )

        for asset_class, exposure in (
            exposures.items()
        ):
            if (
                exposure
                <=
                self.max_asset_class_weight
            ):
                continue

            scale = (
                self.max_asset_class_weight
                /
                exposure
            )

            for ticker in adjusted:
                if (
                    asset_classes.get(
                        ticker,
                        "UNKNOWN"
                    )
                    ==
                    asset_class
                ):
                    adjusted[ticker] *= (
                        scale
                    )

        return adjusted


    @staticmethod
    def remove_small_weights(
        weights,
        minimum_weight
    ):
        return {
            ticker: (
                float(weight)
                if abs(
                    float(weight)
                )
                >= minimum_weight
                else 0.0
            )
            for ticker, weight
            in weights.items()
        }


    def enforce_gross_limit(
        self,
        weights,
        gross_limit
    ):
        gross = (
            self.calculate_gross_exposure(
                weights
            )
        )

        if gross <= gross_limit:
            return dict(
                weights
            )

        if gross == 0:
            return dict(
                weights
            )

        return self.scale_weights(
            weights,
            gross_limit / gross
        )


    def enforce_volatility_target(
        self,
        weights,
        volatilities,
        target_volatility
    ):
        estimated_volatility = (
            self.estimate_independent_volatility(
                weights,
                volatilities
            )
        )

        if (
            estimated_volatility <= 0
            or estimated_volatility
            <= target_volatility
        ):
            return dict(
                weights
            )

        scale = (
            target_volatility
            /
            estimated_volatility
        )

        return self.scale_weights(
            weights,
            scale
        )


    def apply_turnover_buffer(
        self,
        target_weights,
        previous_weights
    ):
        adjusted = {}

        all_tickers = set(
            target_weights
        ) | set(
            previous_weights
        )

        for ticker in all_tickers:
            target = float(
                target_weights.get(
                    ticker,
                    0.0
                )
            )

            previous = float(
                previous_weights.get(
                    ticker,
                    0.0
                )
            )

            difference = abs(
                target
                -
                previous
            )

            same_direction = (
                target
                *
                previous
                > 0
            )

            preserve_position = (
                same_direction
                and
                difference
                <
                self.turnover_buffer
            )

            if preserve_position:
                adjusted[ticker] = (
                    previous
                )

            else:
                adjusted[ticker] = (
                    target
                )

        return adjusted


    def finalize_weights(
        self,
        weights,
        asset_classes,
        volatilities,
        gross_limit,
        volatility_target
    ):
        finalized = self.enforce_asset_cap(
            weights
        )

        finalized = self.enforce_class_cap(
            finalized,
            asset_classes
        )

        finalized = self.enforce_gross_limit(
            finalized,
            gross_limit
        )

        finalized = (
            self.enforce_volatility_target(
                finalized,
                volatilities,
                volatility_target
            )
        )

        finalized = self.remove_small_weights(
            finalized,
            self.minimum_trade_weight
        )

        finalized = self.enforce_gross_limit(
            finalized,
            gross_limit
        )

        return finalized


    def allocate(
        self,
        signals,
        previous_weights=None,
        current_drawdown=0.0
    ):
        normalized_signals = (
            self.normalize_signals(
                signals
            )
        )

        normalized_previous = (
            self.normalize_previous_weights(
                previous_weights
            )
        )

        drawdown_multiplier = (
            self.calculate_drawdown_multiplier(
                current_drawdown
            )
        )

        asset_classes = {}
        volatilities = {}
        raw_weights = {}
        rejected = {}

        for signal in normalized_signals:
            ticker = signal[
                "ticker"
            ]

            action = (
                str(
                    signal.get(
                        "action",
                        "FLAT"
                    )
                )
                .upper()
                .strip()
            )

            signed_signal = float(
                signal.get(
                    "signal",
                    0.0
                )
            )

            volatility = signal.get(
                "annualized_volatility"
            )

            asset_classes[
                ticker
            ] = signal[
                "asset_class"
            ]

            if action not in {
                "LONG",
                "SHORT"
            }:
                rejected[ticker] = (
                    "Segnale FLAT."
                )

                continue

            if volatility is None:
                rejected[ticker] = (
                    "Volatilità mancante."
                )

                continue

            volatility = float(
                volatility
            )

            if (
                not math.isfinite(
                    volatility
                )
                or volatility <= 0
            ):
                rejected[ticker] = (
                    "Volatilità non valida."
                )

                continue

            if (
                not math.isfinite(
                    signed_signal
                )
                or signed_signal == 0
            ):
                rejected[ticker] = (
                    "Segnale numerico non valido."
                )

                continue

            if (
                action == "LONG"
                and signed_signal < 0
            ):
                rejected[ticker] = (
                    "Direzione LONG incoerente."
                )

                continue

            if (
                action == "SHORT"
                and signed_signal > 0
            ):
                rejected[ticker] = (
                    "Direzione SHORT incoerente."
                )

                continue

            volatilities[
                ticker
            ] = volatility

            raw_weights[
                ticker
            ] = (
                signed_signal
                /
                volatility
            )

        if (
            not raw_weights
            or drawdown_multiplier == 0
        ):
            zero_weights = {
                ticker: 0.0
                for ticker in set(
                    normalized_previous
                ) | {
                    signal["ticker"]
                    for signal
                    in normalized_signals
                }
            }

            return {
                "weights": zero_weights,
                "gross_exposure": 0.0,
                "net_exposure": 0.0,
                "cash_weight": 1.0,
                "estimated_volatility": 0.0,
                "target_volatility": (
                    self.target_portfolio_volatility
                    *
                    drawdown_multiplier
                ),
                "drawdown_multiplier": (
                    drawdown_multiplier
                ),
                "turnover": sum(
                    abs(weight)
                    for weight
                    in normalized_previous.values()
                ),
                "asset_class_exposures": {},
                "rejected": rejected,
                "kill_switch_active": (
                    drawdown_multiplier == 0
                )
            }

        absolute_raw_sum = sum(
            abs(weight)
            for weight
            in raw_weights.values()
        )

        gross_limit = (
            self.max_gross_exposure
            *
            drawdown_multiplier
        )

        volatility_target = (
            self.target_portfolio_volatility
            *
            drawdown_multiplier
        )

        target_weights = {
            ticker: (
                raw_weight
                /
                absolute_raw_sum
                *
                gross_limit
            )
            for ticker, raw_weight
            in raw_weights.items()
        }

        target_weights = self.finalize_weights(
            weights=target_weights,
            asset_classes=asset_classes,
            volatilities=volatilities,
            gross_limit=gross_limit,
            volatility_target=(
                volatility_target
            )
        )

        target_weights = (
            self.apply_turnover_buffer(
                target_weights=target_weights,
                previous_weights=(
                    normalized_previous
                )
            )
        )

        target_weights = self.finalize_weights(
            weights=target_weights,
            asset_classes=asset_classes,
            volatilities=volatilities,
            gross_limit=gross_limit,
            volatility_target=(
                volatility_target
            )
        )

        all_tickers = set(
            target_weights
        ) | set(
            normalized_previous
        )

        complete_weights = {
            ticker: float(
                target_weights.get(
                    ticker,
                    0.0
                )
            )
            for ticker in sorted(
                all_tickers
            )
        }

        gross_exposure = (
            self.calculate_gross_exposure(
                complete_weights
            )
        )

        net_exposure = (
            self.calculate_net_exposure(
                complete_weights
            )
        )

        estimated_volatility = (
            self.estimate_independent_volatility(
                complete_weights,
                volatilities
            )
        )

        turnover = sum(
            abs(
                complete_weights.get(
                    ticker,
                    0.0
                )
                -
                normalized_previous.get(
                    ticker,
                    0.0
                )
            )
            for ticker in all_tickers
        )

        class_exposures = (
            self.calculate_class_exposures(
                complete_weights,
                asset_classes
            )
        )

        return {
            "weights": complete_weights,
            "gross_exposure": (
                gross_exposure
            ),
            "net_exposure": net_exposure,
            "cash_weight": max(
                0.0,
                1.0
                -
                gross_exposure
            ),
            "estimated_volatility": (
                estimated_volatility
            ),
            "target_volatility": (
                volatility_target
            ),
            "drawdown_multiplier": (
                drawdown_multiplier
            ),
            "turnover": turnover,
            "asset_class_exposures": (
                class_exposures
            ),
            "rejected": rejected,
            "kill_switch_active": False
        }
