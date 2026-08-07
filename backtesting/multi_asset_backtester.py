import math
import statistics

import numpy as np
import pandas as pd

from backtesting.multi_asset_portfolio import (
    MultiAssetPortfolio
)
from backtesting.portfolio_risk_allocator import (
    PortfolioRiskAllocator
)
from market.specifications import (
    MarketSpecificationRegistry
)
from strategies.multi_asset_tsmom import (
    MultiAssetTimeSeriesMomentumStrategy
)


class MultiAssetBacktester:
    """
    Backtester giornaliero multi-asset.

    Sequenza temporale:

    1. viene contabilizzato il P/L delle posizioni
       detenute dalla barra precedente;

    2. vengono eseguiti i target generati alla
       chiusura della barra precedente;

    3. alla chiusura corrente vengono calcolati
       nuovi segnali;

    4. tali segnali potranno essere eseguiti
       soltanto dalla barra successiva.

    Questa sequenza impedisce di calcolare un
    segnale con il close corrente ed eseguirlo
    retroattivamente allo stesso close.
    """

    ASSET_CLASS_ALIASES = {
        "STOCK": "EQUITY",
        "ETF": "EQUITY",
        "CRYPTO": "CRYPTO",
        "FOREX": "FX",
        "FUTURE": "FUTURE"
    }

    def __init__(
        self,
        strategy=None,
        allocator=None,
        initial_capital=10000.0,
        rebalance_frequency=21,
        use_market_costs=True,
        commission_percent=0.0,
        slippage_percent=0.0,
        liquidate_at_end=True,
        minimum_active_assets=1
    ):
        if strategy is None:
            strategy = (
                MultiAssetTimeSeriesMomentumStrategy()
            )

        if allocator is None:
            allocator = (
                PortfolioRiskAllocator()
            )

        if not hasattr(
            strategy,
            "generate_signals"
        ):
            raise TypeError(
                "La strategia deve implementare "
                "generate_signals()."
            )

        if not hasattr(
            allocator,
            "allocate"
        ):
            raise TypeError(
                "L'allocatore deve implementare allocate()."
            )

        if initial_capital <= 0:
            raise ValueError(
                "initial_capital deve essere positivo."
            )

        if rebalance_frequency <= 0:
            raise ValueError(
                "rebalance_frequency deve essere positivo."
            )

        if minimum_active_assets <= 0:
            raise ValueError(
                "minimum_active_assets deve essere positivo."
            )

        self.strategy = strategy
        self.allocator = allocator

        self.initial_capital = float(
            initial_capital
        )

        self.rebalance_frequency = int(
            rebalance_frequency
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

        self.liquidate_at_end = bool(
            liquidate_at_end
        )

        self.minimum_active_assets = int(
            minimum_active_assets
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
    def prepare_single_asset_data(
        data,
        ticker
    ):
        if not isinstance(
            data,
            pd.DataFrame
        ):
            raise TypeError(
                f"I dati di {ticker} devono "
                "essere un DataFrame."
            )

        if data.empty:
            raise ValueError(
                f"I dati di {ticker} sono vuoti."
            )

        normalized = data.copy()

        normalized.columns = [
            str(column).lower().strip()
            for column in normalized.columns
        ]

        if "close" not in normalized.columns:
            raise ValueError(
                f"Manca close per {ticker}."
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

        normalized["date"] = (
            normalized["date"]
            .dt
            .tz_convert(None)
        )

        normalized = normalized.sort_values(
            "date"
        )

        normalized = normalized.drop_duplicates(
            subset=["date"],
            keep="last"
        )

        normalized = normalized.reset_index(
            drop=True
        )

        if normalized.empty:
            raise ValueError(
                f"Nessuna candela valida per {ticker}."
            )

        return normalized


    def prepare_market_data(
        self,
        market_data
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

        prepared = {}

        for ticker, data in (
            market_data.items()
        ):
            normalized_ticker = (
                self.normalize_ticker(
                    ticker
                )
            )

            prepared[
                normalized_ticker
            ] = (
                self.prepare_single_asset_data(
                    data=data,
                    ticker=normalized_ticker
                )
            )

        return prepared


    @staticmethod
    def build_price_panel(
        prepared_data
    ):
        series = []

        for ticker, data in (
            prepared_data.items()
        ):
            price_series = (
                data.set_index(
                    "date"
                )[
                    "close"
                ]
                .astype(float)
                .rename(
                    ticker
                )
            )

            series.append(
                price_series
            )

        panel = pd.concat(
            series,
            axis=1,
            join="outer"
        )

        panel = panel.sort_index()

        panel = panel.ffill()

        panel = panel.dropna(
            how="all"
        )

        if panel.empty:
            raise ValueError(
                "Impossibile costruire il pannello prezzi."
            )

        return panel


    def build_asset_classes(
        self,
        prepared_data,
        asset_classes=None
    ):
        if asset_classes is None:
            asset_classes = {}

        if not isinstance(
            asset_classes,
            dict
        ):
            raise TypeError(
                "asset_classes deve essere un dizionario."
            )

        normalized_overrides = {
            self.normalize_ticker(
                ticker
            ): (
                str(asset_class)
                .upper()
                .strip()
            )
            for ticker, asset_class
            in asset_classes.items()
        }

        result = {}

        for ticker in prepared_data:
            if ticker in normalized_overrides:
                result[
                    ticker
                ] = normalized_overrides[
                    ticker
                ]

                continue

            specification = (
                MarketSpecificationRegistry.get(
                    ticker
                )
            )

            result[
                ticker
            ] = self.ASSET_CLASS_ALIASES.get(
                specification.asset_class,
                specification.asset_class
            )

        return result


    @staticmethod
    def get_current_prices(
        price_panel,
        timestamp
    ):
        row = price_panel.loc[
            timestamp
        ]

        prices = {}

        for ticker, value in row.items():
            if pd.isna(value):
                continue

            normalized_value = float(
                value
            )

            if (
                math.isfinite(
                    normalized_value
                )
                and normalized_value > 0
            ):
                prices[
                    ticker
                ] = normalized_value

        return prices


    @staticmethod
    def build_history_slices(
        prepared_data,
        timestamp
    ):
        slices = {}

        for ticker, data in (
            prepared_data.items()
        ):
            historical_slice = data[
                data["date"]
                <=
                timestamp
            ].copy()

            if historical_slice.empty:
                continue

            slices[
                ticker
            ] = historical_slice

        return slices


    @staticmethod
    def calculate_drawdown(
        equity,
        peak_equity
    ):
        if peak_equity <= 0:
            return 0.0

        return max(
            0.0,
            (
                peak_equity
                -
                equity
            )
            /
            peak_equity
        )


    @staticmethod
    def calculate_performance_metrics(
        equity_curve,
        initial_capital,
        total_costs,
        total_commissions,
        total_slippage_cost,
        total_turnover_ratio,
        rebalance_count,
        order_count
    ):
        if not equity_curve:
            return {
                "initial_capital": (
                    initial_capital
                ),
                "final_capital": (
                    initial_capital
                ),
                "net_profit": 0.0,
                "total_return_percent": 0.0,
                "annualized_return_percent": 0.0,
                "annualized_volatility_percent": 0.0,
                "sharpe_ratio": 0.0,
                "maximum_drawdown_percent": 0.0,
                "calmar_ratio": 0.0,
                "total_costs": total_costs,
                "total_commissions": (
                    total_commissions
                ),
                "total_slippage_cost": (
                    total_slippage_cost
                ),
                "total_turnover_ratio": (
                    total_turnover_ratio
                ),
                "rebalance_count": (
                    rebalance_count
                ),
                "order_count": order_count
            }

        frame = pd.DataFrame(
            equity_curve
        )

        frame = frame.sort_values(
            "timestamp"
        )

        frame = frame.drop_duplicates(
            subset=["timestamp"],
            keep="last"
        )

        equity = frame[
            "equity"
        ].astype(float)

        final_capital = float(
            equity.iloc[-1]
        )

        net_profit = (
            final_capital
            -
            initial_capital
        )

        total_return = (
            final_capital
            /
            initial_capital
            -
            1
        )

        returns = (
            equity
            .pct_change()
            .replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )
            .dropna()
        )

        observation_count = len(
            returns
        )

        if (
            observation_count > 0
            and final_capital > 0
        ):
            annualized_return = (
                (
                    final_capital
                    /
                    initial_capital
                ) ** (
                    252
                    /
                    observation_count
                )
                -
                1
            )

        else:
            annualized_return = 0.0

        if (
            observation_count > 1
            and float(
                returns.std(
                    ddof=1
                )
            ) > 0
        ):
            annualized_volatility = (
                float(
                    returns.std(
                        ddof=1
                    )
                )
                *
                math.sqrt(252)
            )

            sharpe_ratio = (
                float(
                    returns.mean()
                )
                /
                float(
                    returns.std(
                        ddof=1
                    )
                )
                *
                math.sqrt(252)
            )

        else:
            annualized_volatility = 0.0
            sharpe_ratio = 0.0

        running_peak = (
            equity.cummax()
        )

        drawdowns = (
            running_peak
            -
            equity
        ) / running_peak.replace(
            0,
            np.nan
        )

        maximum_drawdown = float(
            drawdowns.fillna(
                0
            ).max()
        )

        calmar_ratio = (
            annualized_return
            /
            maximum_drawdown
            if maximum_drawdown > 0
            else 0.0
        )

        average_gross_exposure = float(
            frame[
                "gross_exposure"
            ].mean()
        )

        average_net_exposure = float(
            frame[
                "net_exposure"
            ].mean()
        )

        maximum_positions = int(
            frame[
                "position_count"
            ].max()
        )

        return {
            "initial_capital": float(
                initial_capital
            ),
            "final_capital": (
                final_capital
            ),
            "net_profit": float(
                net_profit
            ),
            "total_return_percent": float(
                total_return
                *
                100
            ),
            "annualized_return_percent": float(
                annualized_return
                *
                100
            ),
            "annualized_volatility_percent": float(
                annualized_volatility
                *
                100
            ),
            "sharpe_ratio": float(
                sharpe_ratio
            ),
            "maximum_drawdown_percent": float(
                maximum_drawdown
                *
                100
            ),
            "calmar_ratio": float(
                calmar_ratio
            ),
            "total_costs": float(
                total_costs
            ),
            "total_commissions": float(
                total_commissions
            ),
            "total_slippage_cost": float(
                total_slippage_cost
            ),
            "total_turnover_ratio": float(
                total_turnover_ratio
            ),
            "rebalance_count": int(
                rebalance_count
            ),
            "order_count": int(
                order_count
            ),
            "average_gross_exposure": (
                average_gross_exposure
            ),
            "average_net_exposure": (
                average_net_exposure
            ),
            "maximum_simultaneous_positions": (
                maximum_positions
            ),
            "observation_count": (
                observation_count
            )
        }


    def run(
        self,
        market_data,
        asset_classes=None
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

        first_signal_index = (
            minimum_history
            -
            1
        )

        rebalance_count = 0

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

                rebalance_history.append(
                    rebalance_result
                )

                rebalance_count += 1

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

                if (
                    len(active_signals)
                    >=
                    self.minimum_active_assets
                ):
                    previous_weights = (
                        portfolio
                        .calculate_current_weights(
                            current_prices
                        )
                    )

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
                            "allocation": allocation
                        }
                    )

            portfolio.record_equity(
                timestamp=timestamp,
                prices=current_prices
            )

            if portfolio.bankrupt:
                break

        final_timestamp = price_panel.index[
            -1
        ]

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
            "bankrupt": portfolio.bankrupt
        }