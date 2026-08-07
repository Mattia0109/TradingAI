import copy
import math
import statistics

import numpy as np
import pandas as pd

from backtesting.multi_asset_backtester import (
    MultiAssetBacktester
)
from backtesting.multi_asset_portfolio import (
    MultiAssetPortfolio
)
from market.research_universe import (
    register_research_universe_specifications
)


class ResearchMultiAssetBacktester(
    MultiAssetBacktester
):
    """
    Versione research del backtester multi-asset.

    Aggiunge un trade_start_time:

    - tutti i dati precedenti sono utilizzabili
      per il calcolo dei segnali;
    - nessun segnale viene eseguito prima
      dell'inizio ufficiale del test;
    - l'equity curve valutata parte dal giorno
      del primo segnale;
    - l'esecuzione avviene dalla barra seguente.
    """

    @staticmethod
    def normalize_timestamp(
        value
    ):
        timestamp = pd.to_datetime(
            value,
            errors="coerce",
            utc=True
        )

        if pd.isna(timestamp):
            raise ValueError(
                f"Timestamp non valido: {value}"
            )

        return timestamp.tz_convert(
            None
        )


    @staticmethod
    def create_flat_allocation(
        signals,
        previous_weights,
        current_drawdown
    ):
        tickers = set(
            previous_weights
        ) | set(
            signals
        )

        weights = {
            ticker: 0.0
            for ticker in tickers
        }

        turnover = sum(
            abs(
                float(weight)
            )
            for weight in previous_weights.values()
        )

        return {
            "weights": weights,
            "gross_exposure": 0.0,
            "net_exposure": 0.0,
            "cash_weight": 1.0,
            "estimated_volatility": 0.0,
            "target_volatility": 0.0,
            "drawdown_multiplier": 1.0,
            "turnover": turnover,
            "asset_class_exposures": {},
            "rejected": {
                "portfolio": (
                    "Numero insufficiente "
                    "di segnali attivi."
                )
            },
            "kill_switch_active": False,
            "current_drawdown": float(
                current_drawdown
            )
        }


    def run(
        self,
        market_data,
        asset_classes=None,
        trade_start_time=None
    ):
        prepared_data = (
            self.prepare_market_data(
                market_data
            )
        )

        price_panel = (
            self.build_price_panel(
                prepared_data
            )
        )

        normalized_asset_classes = (
            self.build_asset_classes(
                prepared_data=prepared_data,
                asset_classes=asset_classes
            )
        )

        minimum_history = int(
            getattr(
                self.strategy,
                "minimum_history",
                1
            )
        )

        if len(price_panel) <= minimum_history:
            raise ValueError(
                "Storico insufficiente per il "
                "backtest multi-asset."
            )

        if trade_start_time is None:
            normalized_start = (
                price_panel.index[
                    minimum_history - 1
                ]
            )

        else:
            normalized_start = (
                self.normalize_timestamp(
                    trade_start_time
                )
            )

        eligible_indices = [
            index
            for index, timestamp
            in enumerate(
                price_panel.index
            )
            if (
                index
                >=
                minimum_history - 1
                and timestamp
                >=
                normalized_start
            )
        ]

        if not eligible_indices:
            raise ValueError(
                "Nessuna barra disponibile dopo "
                "trade_start_time."
            )

        first_signal_index = (
            eligible_indices[0]
        )

        if (
            first_signal_index
            >=
            len(price_panel) - 1
        ):
            raise ValueError(
                "Serve almeno una barra successiva "
                "per eseguire il primo segnale."
            )

        evaluation_start_time = (
            price_panel.index[
                first_signal_index
            ]
        )

        portfolio = MultiAssetPortfolio(
            initial_capital=(
                self.initial_capital
            ),
            use_market_costs=(
                self.use_market_costs
            ),
            commission_percent=(
                self.commission_percent
            ),
            slippage_percent=(
                self.slippage_percent
            )
        )

        pending_allocation = None

        signal_history = []
        rebalance_history = []

        peak_equity = float(
            self.initial_capital
        )

        rebalance_count = 0
        first_execution_time = None
        risk_peak_reset_count = 0

        for index, timestamp in enumerate(
            price_panel.index
        ):
            current_prices = (
                self.get_current_prices(
                    price_panel=price_panel,
                    timestamp=timestamp
                )
            )

            portfolio.mark_to_market(
                prices=current_prices,
                timestamp=timestamp
            )

            if pending_allocation is not None:
                rebalance_result = (
                    portfolio.rebalance(
                        target_weights=(
                            pending_allocation[
                                "weights"
                            ]
                        ),
                        prices=current_prices,
                        timestamp=timestamp,
                        source_signal_time=(
                            pending_allocation[
                                "generated_at"
                            ]
                        ),
                        reason="SCHEDULED_REBALANCE"
                    )
                )

                risk_peak_rearmed = (
                    self.should_rearm_risk_peak(
                        allocation=(
                            pending_allocation.get(
                                "allocation",
                                {}
                            )
                        ),
                        portfolio=portfolio
                    )
                )

                if risk_peak_rearmed:
                    peak_equity = float(
                        portfolio.equity
                    )

                    risk_peak_reset_count += 1

                rebalance_result[
                    "risk_peak_rearmed"
                ] = risk_peak_rearmed

                rebalance_history.append(
                    rebalance_result
                )

                rebalance_count += 1

                if (
                    first_execution_time is None
                    and rebalance_result[
                        "orders"
                    ]
                ):
                    first_execution_time = (
                        timestamp
                    )

                pending_allocation = None

            peak_equity = max(
                peak_equity,
                portfolio.equity
            )

            current_drawdown = (
                self.calculate_drawdown(
                    equity=portfolio.equity,
                    peak_equity=peak_equity
                )
            )

            signal_due = (
                index >= first_signal_index
                and (
                    index
                    -
                    first_signal_index
                )
                %
                self.rebalance_frequency
                == 0
            )

            if (
                signal_due
                and not portfolio.bankrupt
            ):
                historical_slices = (
                    self.build_history_slices(
                        prepared_data=(
                            prepared_data
                        ),
                        timestamp=timestamp
                    )
                )

                signals = (
                    self.strategy.generate_signals(
                        market_data=(
                            historical_slices
                        ),
                        asset_classes=(
                            normalized_asset_classes
                        )
                    )
                )

                active_signals = [
                    signal
                    for signal in signals.values()
                    if signal.get(
                        "action"
                    )
                    in {
                        "LONG",
                        "SHORT"
                    }
                ]

                previous_weights = (
                    portfolio
                    .calculate_current_weights(
                        current_prices
                    )
                )

                if (
                    len(active_signals)
                    >=
                    self.minimum_active_assets
                ):
                    allocation = (
                        self.allocator.allocate(
                            signals=signals,
                            previous_weights=(
                                previous_weights
                            ),
                            current_drawdown=(
                                current_drawdown
                            )
                        )
                    )

                else:
                    allocation = (
                        self.create_flat_allocation(
                            signals=signals,
                            previous_weights=(
                                previous_weights
                            ),
                            current_drawdown=(
                                current_drawdown
                            )
                        )
                    )

                pending_allocation = {
                    "generated_at": timestamp,
                    "weights": dict(
                        allocation[
                            "weights"
                        ]
                    ),
                    "allocation": allocation
                }

                signal_history.append(
                    {
                        "timestamp": timestamp,
                        "signals": signals,
                        "allocation": allocation,
                        "active_signal_count": len(
                            active_signals
                        )
                    }
                )

            if index >= first_signal_index:
                portfolio.record_equity(
                    timestamp=timestamp,
                    prices=current_prices
                )

            if portfolio.bankrupt:
                break

        final_timestamp = (
            price_panel.index[-1]
        )

        final_prices = (
            self.get_current_prices(
                price_panel=price_panel,
                timestamp=final_timestamp
            )
        )

        if (
            self.liquidate_at_end
            and not portfolio.is_flat
        ):
            liquidation = (
                portfolio.liquidate(
                    prices=final_prices,
                    timestamp=final_timestamp,
                    reason="FINAL_LIQUIDATION"
                )
            )

            rebalance_history.append(
                liquidation
            )

            portfolio.record_equity(
                timestamp=final_timestamp,
                prices=final_prices
            )

        metrics = (
            self.calculate_performance_metrics(
                equity_curve=(
                    portfolio.equity_curve
                ),
                initial_capital=(
                    portfolio.initial_capital
                ),
                total_costs=(
                    portfolio.total_costs
                ),
                total_commissions=(
                    portfolio.total_commissions
                ),
                total_slippage_cost=(
                    portfolio
                    .total_slippage_cost
                ),
                total_turnover_ratio=(
                    portfolio
                    .total_turnover_ratio
                ),
                rebalance_count=(
                    rebalance_count
                ),
                order_count=len(
                    portfolio.orders
                )
            )
        )

        return {
            "strategy": getattr(
                self.strategy,
                "name",
                self.strategy.__class__.__name__
            ),
            "tickers": sorted(
                prepared_data
            ),
            "asset_classes": (
                normalized_asset_classes
            ),
            "evaluation_start_time": (
                evaluation_start_time
            ),
            "first_execution_time": (
                first_execution_time
            ),
            "evaluation_end_time": (
                final_timestamp
            ),
            "settings": {
                "initial_capital": (
                    self.initial_capital
                ),
                "rebalance_frequency": (
                    self.rebalance_frequency
                ),
                "use_market_costs": (
                    self.use_market_costs
                ),
                "liquidate_at_end": (
                    self.liquidate_at_end
                ),
                "minimum_active_assets": (
                    self.minimum_active_assets
                )
            },
            "metrics": metrics,
            "signals": signal_history,
            "rebalances": (
                rebalance_history
            ),
            "orders": portfolio.orders,
            "equity_curve": (
                portfolio.equity_curve
            ),
            "final_positions": dict(
                portfolio.positions
            ),
            "pending_allocation_at_end": (
                pending_allocation
            ),
            "risk_peak_reset_count": (
                risk_peak_reset_count
            ),
            "bankrupt": portfolio.bankrupt
        }


class StaticAllocationBacktester:
    """
    Benchmark passivo buy-and-hold.

    I pesi vengono stabiliti prima del test,
    eseguiti nella barra successiva alla data
    iniziale e mantenuti fino alla liquidazione.
    """

    def __init__(
        self,
        initial_capital=10000.0,
        use_market_costs=True,
        commission_percent=0.0,
        slippage_percent=0.0
    ):
        if initial_capital <= 0:
            raise ValueError(
                "initial_capital deve essere positivo."
            )

        self.initial_capital = float(
            initial_capital
        )

        self.use_market_costs = bool(
            use_market_costs
        )

        self.commission_percent = float(
            commission_percent
        )

        self.slippage_percent = float(
            slippage_percent
        )

        self.utility = MultiAssetBacktester(
            initial_capital=initial_capital
        )


    @staticmethod
    def normalize_timestamp(
        value
    ):
        timestamp = pd.to_datetime(
            value,
            errors="coerce",
            utc=True
        )

        if pd.isna(timestamp):
            raise ValueError(
                f"Timestamp non valido: {value}"
            )

        return timestamp.tz_convert(
            None
        )


    def run(
        self,
        market_data,
        target_weights,
        trade_start_time,
        name
    ):
        prepared_data = (
            self.utility.prepare_market_data(
                market_data
            )
        )

        price_panel = (
            self.utility.build_price_panel(
                prepared_data
            )
        )

        normalized_start = (
            self.normalize_timestamp(
                trade_start_time
            )
        )

        eligible_indices = [
            index
            for index, timestamp
            in enumerate(
                price_panel.index
            )
            if timestamp >= normalized_start
        ]

        if not eligible_indices:
            raise ValueError(
                "Nessun dato dopo trade_start_time."
            )

        start_index = eligible_indices[0]
        execution_index = start_index + 1

        if execution_index >= len(
            price_panel
        ):
            raise ValueError(
                "Manca la barra successiva "
                "per l'esecuzione."
            )

        portfolio = MultiAssetPortfolio(
            initial_capital=(
                self.initial_capital
            ),
            use_market_costs=(
                self.use_market_costs
            ),
            commission_percent=(
                self.commission_percent
            ),
            slippage_percent=(
                self.slippage_percent
            )
        )

        rebalance_history = []
        first_execution_time = None

        for index, timestamp in enumerate(
            price_panel.index
        ):
            current_prices = (
                self.utility.get_current_prices(
                    price_panel=price_panel,
                    timestamp=timestamp
                )
            )

            portfolio.mark_to_market(
                prices=current_prices,
                timestamp=timestamp
            )

            if index == execution_index:
                available_targets = {
                    ticker: float(weight)
                    for ticker, weight
                    in target_weights.items()
                    if ticker in current_prices
                }

                if not available_targets:
                    raise ValueError(
                        "Nessun target dispone "
                        "di un prezzo eseguibile."
                    )

                result = portfolio.rebalance(
                    target_weights=(
                        available_targets
                    ),
                    prices=current_prices,
                    timestamp=timestamp,
                    source_signal_time=(
                        price_panel.index[
                            start_index
                        ]
                    ),
                    reason=(
                        "STATIC_INITIAL_ALLOCATION"
                    )
                )

                rebalance_history.append(
                    result
                )

                first_execution_time = timestamp

            if index >= start_index:
                portfolio.record_equity(
                    timestamp=timestamp,
                    prices=current_prices
                )

        final_timestamp = (
            price_panel.index[-1]
        )

        final_prices = (
            self.utility.get_current_prices(
                price_panel=price_panel,
                timestamp=final_timestamp
            )
        )

        if not portfolio.is_flat:
            liquidation = (
                portfolio.liquidate(
                    prices=final_prices,
                    timestamp=final_timestamp,
                    reason="FINAL_LIQUIDATION"
                )
            )

            rebalance_history.append(
                liquidation
            )

            portfolio.record_equity(
                timestamp=final_timestamp,
                prices=final_prices
            )

        metrics = (
            self.utility
            .calculate_performance_metrics(
                equity_curve=(
                    portfolio.equity_curve
                ),
                initial_capital=(
                    portfolio.initial_capital
                ),
                total_costs=(
                    portfolio.total_costs
                ),
                total_commissions=(
                    portfolio.total_commissions
                ),
                total_slippage_cost=(
                    portfolio
                    .total_slippage_cost
                ),
                total_turnover_ratio=(
                    portfolio
                    .total_turnover_ratio
                ),
                rebalance_count=1,
                order_count=len(
                    portfolio.orders
                )
            )
        )

        return {
            "strategy": name,
            "evaluation_start_time": (
                price_panel.index[
                    start_index
                ]
            ),
            "first_execution_time": (
                first_execution_time
            ),
            "evaluation_end_time": (
                final_timestamp
            ),
            "target_weights": dict(
                target_weights
            ),
            "metrics": metrics,
            "rebalances": (
                rebalance_history
            ),
            "orders": portfolio.orders,
            "equity_curve": (
                portfolio.equity_curve
            ),
            "final_positions": dict(
                portfolio.positions
            ),
            "bankrupt": portfolio.bankrupt
        }


class MultiAssetResearchBenchmark:
    """
    Confronta TSMOM con benchmark passivi usando:

    - lo stesso universo;
    - gli stessi costi;
    - la stessa data iniziale;
    - la stessa data finale.

    Le frequenze vengono confrontate, non
    ottimizzate sull'intero storico.
    """

    def __init__(
        self,
        strategy,
        allocator,
        initial_capital=10000.0,
        rebalance_frequencies=None,
        use_market_costs=True,
        minimum_active_assets=3
    ):
        if not hasattr(
            strategy,
            "generate_signals"
        ):
            raise TypeError(
                "strategy deve implementare "
                "generate_signals()."
            )

        if not hasattr(
            allocator,
            "allocate"
        ):
            raise TypeError(
                "allocator deve implementare allocate()."
            )

        if rebalance_frequencies is None:
            rebalance_frequencies = [
                5,
                21,
                63
            ]

        normalized_frequencies = sorted(
            {
                int(value)
                for value
                in rebalance_frequencies
            }
        )

        if not normalized_frequencies:
            raise ValueError(
                "Serve almeno una frequenza."
            )

        if any(
            value <= 0
            for value
            in normalized_frequencies
        ):
            raise ValueError(
                "Le frequenze devono essere positive."
            )

        self.strategy = strategy
        self.allocator = allocator

        self.initial_capital = float(
            initial_capital
        )

        self.rebalance_frequencies = (
            normalized_frequencies
        )

        self.use_market_costs = bool(
            use_market_costs
        )

        self.minimum_active_assets = int(
            minimum_active_assets
        )


    @staticmethod
    def calculate_yearly_returns(
        equity_curve,
        initial_capital
    ):
        if not equity_curve:
            return {}

        frame = pd.DataFrame(
            equity_curve
        )

        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"]
        )

        frame = frame.sort_values(
            "timestamp"
        )

        year_end_equity = (
            frame.groupby(
                frame[
                    "timestamp"
                ].dt.year
            )[
                "equity"
            ]
            .last()
        )

        yearly_returns = {}

        previous_equity = float(
            initial_capital
        )

        for year, equity in (
            year_end_equity.items()
        ):
            current_equity = float(
                equity
            )

            yearly_returns[
                int(year)
            ] = (
                (
                    current_equity
                    /
                    previous_equity
                )
                -
                1
            ) * 100

            previous_equity = (
                current_equity
            )

        return yearly_returns


    def determine_common_start(
        self,
        market_data
    ):
        utility = MultiAssetBacktester(
            strategy=copy.deepcopy(
                self.strategy
            ),
            allocator=copy.deepcopy(
                self.allocator
            ),
            initial_capital=(
                self.initial_capital
            )
        )

        prepared = (
            utility.prepare_market_data(
                market_data
            )
        )

        minimum_history = int(
            getattr(
                self.strategy,
                "minimum_history",
                1
            )
        )

        eligible = {}
        excluded = {}

        ready_dates = []

        for ticker, data in (
            prepared.items()
        ):
            if len(data) < minimum_history:
                excluded[ticker] = (
                    f"Solo {len(data)} barre; "
                    f"richieste {minimum_history}."
                )

                continue

            eligible[ticker] = data

            ready_dates.append(
                data.iloc[
                    minimum_history - 1
                ][
                    "date"
                ]
            )

        if not eligible:
            raise ValueError(
                "Nessun asset dispone dello "
                "storico minimo necessario."
            )

        common_start = max(
            ready_dates
        )

        return {
            "trade_start_time": (
                common_start
            ),
            "eligible_data": eligible,
            "excluded_assets": excluded
        }


    @staticmethod
    def calculate_frequency_summary(
        results
    ):
        if not results:
            return {
                "tested_frequencies": 0,
                "positive_frequencies": 0,
                "positive_frequency_percent": 0.0,
                "median_return_percent": 0.0,
                "median_sharpe_ratio": 0.0,
                "worst_return_percent": 0.0,
                "worst_drawdown_percent": 0.0,
                "status": "NO_RESULTS"
            }

        returns = [
            float(
                result[
                    "metrics"
                ][
                    "total_return_percent"
                ]
            )
            for result in results
        ]

        sharpes = [
            float(
                result[
                    "metrics"
                ][
                    "sharpe_ratio"
                ]
            )
            for result in results
        ]

        drawdowns = [
            float(
                result[
                    "metrics"
                ][
                    "maximum_drawdown_percent"
                ]
            )
            for result in results
        ]

        positive_count = sum(
            1
            for value in returns
            if value > 0
        )

        positive_percent = (
            positive_count
            /
            len(results)
            *
            100
        )

        median_sharpe = (
            statistics.median(
                sharpes
            )
        )

        worst_drawdown = max(
            drawdowns
        )

        if (
            positive_count == len(results)
            and median_sharpe >= 0.50
            and worst_drawdown <= 25
        ):
            status = (
                "ROBUST_FREQUENCY_CANDIDATE"
            )

        elif (
            positive_count
            >=
            math.ceil(
                len(results) * 0.67
            )
        ):
            status = (
                "MIXED_FREQUENCY_CANDIDATE"
            )

        else:
            status = (
                "REJECT_FREQUENCY_FRAGILITY"
            )

        return {
            "tested_frequencies": len(
                results
            ),
            "positive_frequencies": (
                positive_count
            ),
            "positive_frequency_percent": (
                positive_percent
            ),
            "median_return_percent": (
                statistics.median(
                    returns
                )
            ),
            "median_sharpe_ratio": (
                median_sharpe
            ),
            "worst_return_percent": min(
                returns
            ),
            "worst_drawdown_percent": (
                worst_drawdown
            ),
            "status": status
        }


    @staticmethod
    def ranking_key(
        result
    ):
        metrics = result[
            "metrics"
        ]

        return (
            float(
                metrics[
                    "sharpe_ratio"
                ]
            ),
            float(
                metrics[
                    "calmar_ratio"
                ]
            ),
            float(
                metrics[
                    "total_return_percent"
                ]
            ),
            -float(
                metrics[
                    "maximum_drawdown_percent"
                ]
            )
        )


    def run(
        self,
        market_data,
        asset_classes
    ):
        register_research_universe_specifications()

        start_information = (
            self.determine_common_start(
                market_data
            )
        )

        trade_start_time = (
            start_information[
                "trade_start_time"
            ]
        )

        eligible_data = (
            start_information[
                "eligible_data"
            ]
        )

        eligible_asset_classes = {
            ticker: asset_classes[
                ticker
            ]
            for ticker in eligible_data
            if ticker in asset_classes
        }

        tsmom_results = []

        for frequency in (
            self.rebalance_frequencies
        ):
            backtester = (
                ResearchMultiAssetBacktester(
                    strategy=copy.deepcopy(
                        self.strategy
                    ),
                    allocator=copy.deepcopy(
                        self.allocator
                    ),
                    initial_capital=(
                        self.initial_capital
                    ),
                    rebalance_frequency=(
                        frequency
                    ),
                    use_market_costs=(
                        self.use_market_costs
                    ),
                    liquidate_at_end=True,
                    minimum_active_assets=(
                        self.minimum_active_assets
                    )
                )
            )

            result = backtester.run(
                market_data=eligible_data,
                asset_classes=(
                    eligible_asset_classes
                ),
                trade_start_time=(
                    trade_start_time
                )
            )

            result[
                "benchmark_name"
            ] = (
                f"TSMOM_{frequency}D"
            )

            result[
                "rebalance_frequency"
            ] = frequency

            result[
                "yearly_returns"
            ] = (
                self.calculate_yearly_returns(
                    equity_curve=(
                        result[
                            "equity_curve"
                        ]
                    ),
                    initial_capital=(
                        self.initial_capital
                    )
                )
            )

            tsmom_results.append(
                result
            )

        static_backtester = (
            StaticAllocationBacktester(
                initial_capital=(
                    self.initial_capital
                ),
                use_market_costs=(
                    self.use_market_costs
                )
            )
        )

        equal_weight = (
            1.0
            /
            len(eligible_data)
        )

        equal_weight_targets = {
            ticker: equal_weight
            for ticker in eligible_data
        }

        equal_weight_result = (
            static_backtester.run(
                market_data=eligible_data,
                target_weights=(
                    equal_weight_targets
                ),
                trade_start_time=(
                    trade_start_time
                ),
                name=(
                    "EQUAL_WEIGHT_BUY_HOLD"
                )
            )
        )

        equal_weight_result[
            "benchmark_name"
        ] = (
            "EQUAL_WEIGHT_BUY_HOLD"
        )

        equal_weight_result[
            "yearly_returns"
        ] = (
            self.calculate_yearly_returns(
                equity_curve=(
                    equal_weight_result[
                        "equity_curve"
                    ]
                ),
                initial_capital=(
                    self.initial_capital
                )
            )
        )

        passive_results = [
            equal_weight_result
        ]

        if (
            "SPY" in eligible_data
            and "TLT" in eligible_data
        ):
            sixty_forty_result = (
                static_backtester.run(
                    market_data={
                        "SPY": eligible_data[
                            "SPY"
                        ],
                        "TLT": eligible_data[
                            "TLT"
                        ]
                    },
                    target_weights={
                        "SPY": 0.60,
                        "TLT": 0.40
                    },
                    trade_start_time=(
                        trade_start_time
                    ),
                    name="BUY_HOLD_60_40"
                )
            )

            sixty_forty_result[
                "benchmark_name"
            ] = "BUY_HOLD_60_40"

            sixty_forty_result[
                "yearly_returns"
            ] = (
                self.calculate_yearly_returns(
                    equity_curve=(
                        sixty_forty_result[
                            "equity_curve"
                        ]
                    ),
                    initial_capital=(
                        self.initial_capital
                    )
                )
            )

            passive_results.append(
                sixty_forty_result
            )

        all_results = (
            tsmom_results
            +
            passive_results
        )

        ranked_results = sorted(
            all_results,
            key=self.ranking_key,
            reverse=True
        )

        return {
            "trade_start_time": (
                trade_start_time
            ),
            "eligible_assets": sorted(
                eligible_data
            ),
            "excluded_assets": (
                start_information[
                    "excluded_assets"
                ]
            ),
            "tsmom_results": (
                tsmom_results
            ),
            "passive_results": (
                passive_results
            ),
            "ranked_results": (
                ranked_results
            ),
            "frequency_summary": (
                self.calculate_frequency_summary(
                    tsmom_results
                )
            )
        }
