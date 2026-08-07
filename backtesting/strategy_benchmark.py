import copy
import statistics

import pandas as pd

from backtesting.walk_forward import (
    WalkForwardValidator
)
from market.specifications import (
    MarketSpecificationRegistry
)


class StrategyBenchmarkRunner:
    """
    Benchmark multi-asset e multi-timeframe.

    Modalità supportate:

    gross:
        ricerca dell'edge fuori campione
        senza commissioni e slippage.

    market:
        usa commissioni, spread, slippage,
        quantità e moltiplicatori specifici
        dello strumento.
    """

    ALLOWED_COST_MODES = {
        "gross",
        "market"
    }

    GROSS_CANDIDATE_CLASSIFICATIONS = {
        "WEAK_GROSS_EDGE",
        "GROSS_EDGE_CANDIDATE"
    }

    MARKET_CANDIDATE_CLASSIFICATIONS = {
        "WEAK_NET_EDGE",
        "NET_EDGE_CANDIDATE"
    }

    def __init__(
        self,
        strategies,
        data_loader,
        initial_capital=10000.0,
        max_position_percent=0.10,
        minimum_history=30,
        max_holding_bars=78,
        cooldown_bars=12,
        max_trades_per_day=3,
        cost_mode="gross",
        margin_overrides=None,
        validator_class=WalkForwardValidator,
        progress_callback=None
    ):
        if not isinstance(
            strategies,
            dict
        ):
            raise TypeError(
                "strategies deve essere un dizionario."
            )

        if not strategies:
            raise ValueError(
                "Deve essere presente almeno una strategia."
            )

        for (
            strategy_name,
            strategy
        ) in strategies.items():
            if not strategy_name:
                raise ValueError(
                    "Il nome della strategia non può essere vuoto."
                )

            if (
                strategy is not None
                and not hasattr(
                    strategy,
                    "generate_decision"
                )
            ):
                raise TypeError(
                    f"La strategia {strategy_name} deve "
                    "implementare generate_decision()."
                )

        if not callable(
            data_loader
        ):
            raise TypeError(
                "data_loader deve essere chiamabile."
            )

        if initial_capital <= 0:
            raise ValueError(
                "initial_capital deve essere positivo."
            )

        if not 0 < max_position_percent <= 0.10:
            raise ValueError(
                "max_position_percent deve essere "
                "compreso tra 0 e 0.10."
            )

        if minimum_history < 26:
            raise ValueError(
                "minimum_history deve essere almeno 26."
            )

        if max_holding_bars <= 0:
            raise ValueError(
                "max_holding_bars deve essere positivo."
            )

        if cooldown_bars < 0:
            raise ValueError(
                "cooldown_bars non può essere negativo."
            )

        if max_trades_per_day <= 0:
            raise ValueError(
                "max_trades_per_day deve essere positivo."
            )

        normalized_cost_mode = (
            str(cost_mode)
            .lower()
            .strip()
        )

        if (
            normalized_cost_mode
            not in self.ALLOWED_COST_MODES
        ):
            raise ValueError(
                "cost_mode deve essere gross o market."
            )

        if margin_overrides is None:
            margin_overrides = {}

        if not isinstance(
            margin_overrides,
            dict
        ):
            raise TypeError(
                "margin_overrides deve essere un dizionario."
            )

        normalized_margin_overrides = {}

        for ticker, value in (
            margin_overrides.items()
        ):
            normalized_ticker = (
                str(ticker)
                .upper()
                .strip()
            )

            normalized_value = float(
                value
            )

            if not normalized_ticker:
                raise ValueError(
                    "Ticker vuoto in margin_overrides."
                )

            if normalized_value <= 0:
                raise ValueError(
                    "Ogni margin override deve essere positivo."
                )

            normalized_margin_overrides[
                normalized_ticker
            ] = normalized_value

        self.strategies = strategies
        self.data_loader = data_loader

        self.initial_capital = float(
            initial_capital
        )

        self.max_position_percent = float(
            max_position_percent
        )

        self.minimum_history = int(
            minimum_history
        )

        self.max_holding_bars = int(
            max_holding_bars
        )

        self.cooldown_bars = int(
            cooldown_bars
        )

        self.max_trades_per_day = int(
            max_trades_per_day
        )

        self.cost_mode = (
            normalized_cost_mode
        )

        self.use_market_costs = (
            self.cost_mode == "market"
        )

        self.margin_overrides = (
            normalized_margin_overrides
        )

        self.validator_class = (
            validator_class
        )

        self.progress_callback = (
            progress_callback
        )


    @property
    def cost_model_name(self):
        if self.cost_mode == "market":
            return "market_specification"

        return "zero_costs"


    def report_progress(
        self,
        message
    ):
        if self.progress_callback is not None:
            self.progress_callback(
                message
            )


    @staticmethod
    def validate_market_case(
        market_case
    ):
        if not isinstance(
            market_case,
            dict
        ):
            raise TypeError(
                "Ogni market case deve essere un dizionario."
            )

        required_fields = {
            "ticker",
            "period",
            "interval",
            "train_bars",
            "test_bars",
            "step_bars"
        }

        missing_fields = (
            required_fields
            -
            set(market_case)
        )

        if missing_fields:
            raise ValueError(
                "Campi mancanti nel market case: "
                f"{sorted(missing_fields)}"
            )

        if not market_case["ticker"]:
            raise ValueError(
                "ticker non può essere vuoto."
            )

        if not market_case["period"]:
            raise ValueError(
                "period non può essere vuoto."
            )

        if not market_case["interval"]:
            raise ValueError(
                "interval non può essere vuoto."
            )

        for field in [
            "train_bars",
            "test_bars",
            "step_bars"
        ]:
            if int(
                market_case[field]
            ) <= 0:
                raise ValueError(
                    f"{field} deve essere positivo."
                )


    def get_execution_issue(
        self,
        ticker
    ):
        """
        Controlla se lo strumento può essere
        simulato correttamente.

        I futures con margine esterno devono avere
        un margin override configurato.
        """

        normalized_ticker = (
            str(ticker)
            .upper()
            .strip()
        )

        try:
            specification = (
                MarketSpecificationRegistry.get(
                    normalized_ticker
                )
            )

        except Exception as error:
            return (
                "Specifiche di mercato non disponibili: "
                f"{error}"
            )

        if (
            specification.margin_model
            == "external"
            and normalized_ticker
            not in self.margin_overrides
        ):
            return (
                "Margine esterno non configurato "
                f"per {normalized_ticker}"
            )

        return None


    @staticmethod
    def classify_summary(
        summary,
        cost_mode
    ):
        """
        Classifica il risultato fuori campione.

        In gross valuta il vantaggio lordo.

        In market valuta il vantaggio netto
        dopo costi specifici dello strumento.
        """

        total_folds = int(
            summary.get(
                "total_folds",
                0
            )
        )

        total_trades = int(
            summary.get(
                "total_trades",
                0
            )
        )

        net_profit = float(
            summary.get(
                "total_net_profit",
                0.0
            )
        )

        profitable_percent = float(
            summary.get(
                "profitable_fold_percent",
                0.0
            )
        )

        efficiency = float(
            summary.get(
                "walk_forward_efficiency",
                0.0
            )
        )

        is_robust = bool(
            summary.get(
                "is_robust",
                False
            )
        )

        if total_folds < 3:
            return "INSUFFICIENT_FOLDS"

        if total_trades < 30:
            return "INSUFFICIENT_TRADES"

        if cost_mode == "market":
            if net_profit <= 0:
                return "REJECT_NO_NET_EDGE"

            if efficiency <= 1:
                return "REJECT_UNSTABLE_NET_EDGE"

            if profitable_percent < 50:
                return "REJECT_INCONSISTENT_NET_EDGE"

            if is_robust:
                return "NET_EDGE_CANDIDATE"

            return "WEAK_NET_EDGE"

        if net_profit <= 0:
            return "REJECT_NO_GROSS_EDGE"

        if efficiency <= 1:
            return "REJECT_UNSTABLE_EDGE"

        if profitable_percent < 50:
            return "REJECT_INCONSISTENT_EDGE"

        if is_robust:
            return "GROSS_EDGE_CANDIDATE"

        return "WEAK_GROSS_EDGE"


    def is_candidate(
        self,
        classification
    ):
        if self.cost_mode == "market":
            return (
                classification
                in self.MARKET_CANDIDATE_CLASSIFICATIONS
            )

        return (
            classification
            in self.GROSS_CANDIDATE_CLASSIFICATIONS
        )


    def build_validator(
        self,
        strategy,
        market_case
    ):
        strategy_copy = (
            copy.deepcopy(
                strategy
            )
            if strategy is not None
            else None
        )

        return self.validator_class(
            train_bars=int(
                market_case["train_bars"]
            ),
            test_bars=int(
                market_case["test_bars"]
            ),
            step_bars=int(
                market_case["step_bars"]
            ),
            initial_capital=(
                self.initial_capital
            ),
            max_position_percent=(
                self.max_position_percent
            ),
            minimum_history=(
                self.minimum_history
            ),
            commission_percent=0.0,
            slippage_percent=0.0,
            max_holding_bars=(
                self.max_holding_bars
            ),
            cooldown_bars=(
                self.cooldown_bars
            ),
            max_trades_per_day=(
                self.max_trades_per_day
            ),
            strategy=strategy_copy,
            use_market_costs=(
                self.use_market_costs
            ),
            margin_overrides=(
                self.margin_overrides
            )
        )


    def evaluate_strategy(
        self,
        strategy_name,
        strategy,
        market_case,
        data
    ):
        validator = self.build_validator(
            strategy=strategy,
            market_case=market_case
        )

        walk_forward_result = (
            validator.run(
                ticker=market_case[
                    "ticker"
                ],
                data=data
            )
        )

        summary = walk_forward_result[
            "summary"
        ]

        classification = (
            self.classify_summary(
                summary=summary,
                cost_mode=self.cost_mode
            )
        )

        candidate = self.is_candidate(
            classification
        )

        return {
            "status": "COMPLETED",
            "ticker": market_case[
                "ticker"
            ],
            "period": market_case[
                "period"
            ],
            "interval": market_case[
                "interval"
            ],
            "strategy": strategy_name,
            "cost_mode": self.cost_mode,
            "cost_model": (
                self.cost_model_name
            ),
            "total_folds": int(
                summary[
                    "total_folds"
                ]
            ),
            "profitable_folds": int(
                summary[
                    "profitable_folds"
                ]
            ),
            "profitable_fold_percent": float(
                summary[
                    "profitable_fold_percent"
                ]
            ),
            "total_trades": int(
                summary[
                    "total_trades"
                ]
            ),
            "total_net_profit": float(
                summary[
                    "total_net_profit"
                ]
            ),
            "total_commissions": float(
                summary.get(
                    "total_commissions",
                    0.0
                )
            ),
            "average_fold_return_percent": float(
                summary[
                    "average_fold_return_percent"
                ]
            ),
            "median_fold_return_percent": float(
                summary[
                    "median_fold_return_percent"
                ]
            ),
            "worst_fold_return_percent": float(
                summary[
                    "worst_fold_return_percent"
                ]
            ),
            "best_fold_return_percent": float(
                summary[
                    "best_fold_return_percent"
                ]
            ),
            "walk_forward_efficiency": float(
                summary[
                    "walk_forward_efficiency"
                ]
            ),
            "is_robust": bool(
                summary[
                    "is_robust"
                ]
            ),
            "classification": classification,
            "is_candidate": candidate,
            "advance_to_cost_test": (
                self.cost_mode == "gross"
                and candidate
            ),
            "advance_to_robustness_test": (
                self.cost_mode == "market"
                and candidate
            )
        }


    @staticmethod
    def result_rank_key(
        result
    ):
        classification_priority = {
            "NET_EDGE_CANDIDATE": 8,
            "GROSS_EDGE_CANDIDATE": 8,
            "WEAK_NET_EDGE": 7,
            "WEAK_GROSS_EDGE": 7,
            "INSUFFICIENT_TRADES": 6,
            "INSUFFICIENT_FOLDS": 5,
            "REJECT_INCONSISTENT_NET_EDGE": 4,
            "REJECT_INCONSISTENT_EDGE": 4,
            "REJECT_UNSTABLE_NET_EDGE": 3,
            "REJECT_UNSTABLE_EDGE": 3,
            "REJECT_NO_NET_EDGE": 2,
            "REJECT_NO_GROSS_EDGE": 2,
            "ERROR": 0,
            "SKIPPED": 0
        }

        return (
            classification_priority.get(
                result.get(
                    "classification"
                ),
                0
            ),
            bool(
                result.get(
                    "is_robust",
                    False
                )
            ),
            float(
                result.get(
                    "profitable_fold_percent",
                    0.0
                )
            ),
            float(
                result.get(
                    "walk_forward_efficiency",
                    0.0
                )
            ),
            float(
                result.get(
                    "total_net_profit",
                    0.0
                )
            )
        )


    @staticmethod
    def aggregate_strategy(
        strategy_name,
        results
    ):
        strategy_results = [
            result
            for result in results
            if result["strategy"] == strategy_name
        ]

        completed = [
            result
            for result in strategy_results
            if result["status"] == "COMPLETED"
        ]

        skipped = [
            result
            for result in strategy_results
            if result["status"] == "SKIPPED"
        ]

        errors = [
            result
            for result in strategy_results
            if result["status"] == "ERROR"
        ]

        positive = [
            result
            for result in completed
            if result[
                "total_net_profit"
            ] > 0
        ]

        candidates = [
            result
            for result in completed
            if result.get(
                "is_candidate",
                False
            )
        ]

        robust = [
            result
            for result in completed
            if result[
                "classification"
            ]
            in {
                "GROSS_EDGE_CANDIDATE",
                "NET_EDGE_CANDIDATE"
            }
        ]

        net_profits = [
            result[
                "total_net_profit"
            ]
            for result in completed
        ]

        profitable_percentages = [
            result[
                "profitable_fold_percent"
            ]
            for result in completed
        ]

        efficiencies = [
            result[
                "walk_forward_efficiency"
            ]
            for result in completed
        ]

        commissions = [
            result.get(
                "total_commissions",
                0.0
            )
            for result in completed
        ]

        completed_count = len(
            completed
        )

        positive_rate = (
            len(positive)
            /
            completed_count
            *
            100
            if completed_count > 0
            else 0.0
        )

        candidate_rate = (
            len(candidates)
            /
            completed_count
            *
            100
            if completed_count > 0
            else 0.0
        )

        if not completed:
            overall_status = "NO_DATA"

        elif not candidates:
            overall_status = "REJECTED"

        elif len(robust) >= 2:
            overall_status = (
                "STRONG_CANDIDATE"
            )

        else:
            overall_status = "CANDIDATE"

        return {
            "strategy": strategy_name,
            "total_cases": len(
                strategy_results
            ),
            "completed_cases": (
                completed_count
            ),
            "skipped_cases": len(
                skipped
            ),
            "error_cases": len(
                errors
            ),
            "positive_cases": len(
                positive
            ),
            "positive_case_percent": (
                positive_rate
            ),
            "candidate_cases": len(
                candidates
            ),
            "candidate_case_percent": (
                candidate_rate
            ),
            "robust_cases": len(
                robust
            ),
            "total_net_profit": (
                sum(net_profits)
                if net_profits
                else 0.0
            ),
            "total_commissions": (
                sum(commissions)
                if commissions
                else 0.0
            ),
            "median_net_profit": (
                statistics.median(
                    net_profits
                )
                if net_profits
                else 0.0
            ),
            "average_profitable_fold_percent": (
                sum(
                    profitable_percentages
                )
                /
                len(
                    profitable_percentages
                )
                if profitable_percentages
                else 0.0
            ),
            "median_walk_forward_efficiency": (
                statistics.median(
                    efficiencies
                )
                if efficiencies
                else 0.0
            ),
            "overall_status": (
                overall_status
            )
        }


    @staticmethod
    def aggregate_rank_key(
        aggregate
    ):
        status_priority = {
            "STRONG_CANDIDATE": 3,
            "CANDIDATE": 2,
            "REJECTED": 1,
            "NO_DATA": 0
        }

        return (
            status_priority.get(
                aggregate[
                    "overall_status"
                ],
                0
            ),
            aggregate[
                "robust_cases"
            ],
            aggregate[
                "candidate_cases"
            ],
            aggregate[
                "positive_case_percent"
            ],
            aggregate[
                "median_walk_forward_efficiency"
            ],
            aggregate[
                "median_net_profit"
            ]
        )


    def build_failed_result(
        self,
        status,
        ticker,
        period,
        interval,
        strategy_name,
        error
    ):
        return {
            "status": status,
            "ticker": ticker,
            "period": period,
            "interval": interval,
            "strategy": strategy_name,
            "cost_mode": self.cost_mode,
            "cost_model": (
                self.cost_model_name
            ),
            "classification": status,
            "is_candidate": False,
            "advance_to_cost_test": False,
            "advance_to_robustness_test": False,
            "error": str(
                error
            )
        }


    def run(
        self,
        market_cases
    ):
        if not isinstance(
            market_cases,
            list
        ):
            raise TypeError(
                "market_cases deve essere una lista."
            )

        if not market_cases:
            raise ValueError(
                "market_cases non può essere vuota."
            )

        results = []

        total_cases = len(
            market_cases
        )

        for (
            case_index,
            market_case
        ) in enumerate(
            market_cases,
            start=1
        ):
            self.validate_market_case(
                market_case
            )

            ticker = (
                str(
                    market_case[
                        "ticker"
                    ]
                )
                .upper()
                .strip()
            )

            period = market_case[
                "period"
            ]

            interval = market_case[
                "interval"
            ]

            execution_issue = (
                self.get_execution_issue(
                    ticker
                )
            )

            if execution_issue is not None:
                self.report_progress(
                    f"[{case_index}/{total_cases}] "
                    f"SKIP {ticker} {interval}: "
                    f"{execution_issue}"
                )

                for strategy_name in (
                    self.strategies
                ):
                    results.append(
                        self.build_failed_result(
                            status="SKIPPED",
                            ticker=ticker,
                            period=period,
                            interval=interval,
                            strategy_name=(
                                strategy_name
                            ),
                            error=execution_issue
                        )
                    )

                continue

            self.report_progress(
                f"[{case_index}/{total_cases}] "
                f"Caricamento {ticker} "
                f"{interval} {period}"
            )

            try:
                data = self.data_loader(
                    ticker=ticker,
                    period=period,
                    interval=interval
                )

            except Exception as error:
                for strategy_name in (
                    self.strategies
                ):
                    results.append(
                        self.build_failed_result(
                            status="ERROR",
                            ticker=ticker,
                            period=period,
                            interval=interval,
                            strategy_name=(
                                strategy_name
                            ),
                            error=error
                        )
                    )

                continue

            if (
                data is None
                or not isinstance(
                    data,
                    pd.DataFrame
                )
                or data.empty
            ):
                for strategy_name in (
                    self.strategies
                ):
                    results.append(
                        self.build_failed_result(
                            status="SKIPPED",
                            ticker=ticker,
                            period=period,
                            interval=interval,
                            strategy_name=(
                                strategy_name
                            ),
                            error=(
                                "Nessun dato disponibile"
                            )
                        )
                    )

                continue

            minimum_required = (
                int(
                    market_case[
                        "train_bars"
                    ]
                )
                +
                int(
                    market_case[
                        "test_bars"
                    ]
                )
            )

            if len(data) < minimum_required:
                for strategy_name in (
                    self.strategies
                ):
                    failure = (
                        self.build_failed_result(
                            status="SKIPPED",
                            ticker=ticker,
                            period=period,
                            interval=interval,
                            strategy_name=(
                                strategy_name
                            ),
                            error=(
                                "Storico insufficiente"
                            )
                        )
                    )

                    failure["available_bars"] = len(
                        data
                    )

                    failure["required_bars"] = (
                        minimum_required
                    )

                    results.append(
                        failure
                    )

                continue

            for (
                strategy_name,
                strategy
            ) in self.strategies.items():
                self.report_progress(
                    f"    Test strategia "
                    f"{strategy_name}"
                )

                try:
                    result = (
                        self.evaluate_strategy(
                            strategy_name=(
                                strategy_name
                            ),
                            strategy=strategy,
                            market_case=(
                                market_case
                            ),
                            data=data
                        )
                    )

                except Exception as error:
                    result = (
                        self.build_failed_result(
                            status="ERROR",
                            ticker=ticker,
                            period=period,
                            interval=interval,
                            strategy_name=(
                                strategy_name
                            ),
                            error=error
                        )
                    )

                results.append(
                    result
                )

        ranked_results = sorted(
            results,
            key=self.result_rank_key,
            reverse=True
        )

        candidates = [
            result
            for result in ranked_results
            if result.get(
                "is_candidate",
                False
            )
        ]

        aggregates = [
            self.aggregate_strategy(
                strategy_name=(
                    strategy_name
                ),
                results=results
            )
            for strategy_name in (
                self.strategies
            )
        ]

        aggregates = sorted(
            aggregates,
            key=self.aggregate_rank_key,
            reverse=True
        )

        return {
            "settings": {
                "initial_capital": (
                    self.initial_capital
                ),
                "max_position_percent": (
                    self.max_position_percent
                ),
                "cost_mode": (
                    self.cost_mode
                ),
                "cost_model": (
                    self.cost_model_name
                ),
                "use_market_costs": (
                    self.use_market_costs
                ),
                "margin_overrides": dict(
                    self.margin_overrides
                )
            },
            "results": results,
            "ranked_results": (
                ranked_results
            ),
            "candidates": candidates,
            "strategy_ranking": (
                aggregates
            )
        }