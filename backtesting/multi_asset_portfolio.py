import math
from copy import deepcopy

from market.specifications import (
    MarketSpecificationRegistry
)


class MultiAssetPortfolio:
    """
    Portfolio mark-to-market capace di mantenere
    più posizioni LONG e SHORT simultaneamente.

    Il portfolio usa quantità eseguibili:

    - azioni ed ETF: quantità intere;
    - crypto: quantità frazionarie;
    - futures: moltiplicatore contrattuale;
    - forex: step definito dal registry.

    Il P/L viene aggiornato attraverso la variazione
    giornaliera dei prezzi.

    Commissioni, spread e slippage vengono sottratti
    dall'equity al momento del ribilanciamento.
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

        if commission_percent < 0:
            raise ValueError(
                "commission_percent non può essere negativo."
            )

        if slippage_percent < 0:
            raise ValueError(
                "slippage_percent non può essere negativo."
            )

        self.initial_capital = float(
            initial_capital
        )

        self.equity = float(
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

        self.positions = {}
        self.previous_prices = {}

        self.total_commissions = 0.0
        self.total_slippage_cost = 0.0
        self.total_costs = 0.0

        self.total_turnover_notional = 0.0
        self.total_turnover_ratio = 0.0

        self.orders = []
        self.equity_curve = []

        self.realized_daily_pnl = 0.0
        self.bankrupt = False


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
    def validate_price(
        price,
        ticker
    ):
        normalized_price = float(
            price
        )

        if (
            not math.isfinite(
                normalized_price
            )
            or normalized_price <= 0
        ):
            raise ValueError(
                f"Prezzo non valido per {ticker}: "
                f"{price}"
            )

        return normalized_price


    @staticmethod
    def normalize_prices(
        prices
    ):
        if not isinstance(
            prices,
            dict
        ):
            raise TypeError(
                "prices deve essere un dizionario."
            )

        normalized = {}

        for ticker, price in prices.items():
            normalized_ticker = (
                MultiAssetPortfolio
                .normalize_ticker(
                    ticker
                )
            )

            try:
                normalized_price = (
                    MultiAssetPortfolio
                    .validate_price(
                        price,
                        normalized_ticker
                    )
                )

            except (
                TypeError,
                ValueError
            ):
                continue

            normalized[
                normalized_ticker
            ] = normalized_price

        return normalized


    @property
    def position_count(self):
        return sum(
            1
            for quantity in self.positions.values()
            if abs(
                float(quantity)
            ) > 0
        )


    @property
    def is_flat(self):
        return self.position_count == 0


    def mark_to_market(
        self,
        prices,
        timestamp=None
    ):
        """
        Aggiorna l'equity usando le variazioni
        rispetto ai prezzi precedenti.

        Per una posizione SHORT la quantità è
        negativa, quindi una discesa del prezzo
        genera automaticamente un P/L positivo.
        """

        normalized_prices = (
            self.normalize_prices(
                prices
            )
        )

        daily_pnl = 0.0
        ticker_pnl = {}

        for ticker, quantity in (
            self.positions.items()
        ):
            if ticker not in normalized_prices:
                continue

            if ticker not in self.previous_prices:
                continue

            current_price = (
                normalized_prices[
                    ticker
                ]
            )

            previous_price = (
                self.previous_prices[
                    ticker
                ]
            )

            specification = (
                MarketSpecificationRegistry.get(
                    ticker
                )
            )

            pnl = (
                current_price
                -
                previous_price
            ) * (
                float(quantity)
            ) * (
                specification.contract_multiplier
            )

            daily_pnl += pnl

            ticker_pnl[
                ticker
            ] = pnl

        self.equity += daily_pnl
        self.realized_daily_pnl = daily_pnl

        for ticker, price in (
            normalized_prices.items()
        ):
            self.previous_prices[
                ticker
            ] = price

        if self.equity <= 0:
            self.equity = 0.0
            self.bankrupt = True

        return {
            "timestamp": timestamp,
            "daily_pnl": float(
                daily_pnl
            ),
            "ticker_pnl": ticker_pnl,
            "equity": float(
                self.equity
            )
        }


    def calculate_current_weights(
        self,
        prices
    ):
        """
        Calcola i pesi effettivi delle posizioni
        rispetto all'equity corrente.
        """

        normalized_prices = (
            self.normalize_prices(
                prices
            )
        )

        if self.equity <= 0:
            return {
                ticker: 0.0
                for ticker in self.positions
            }

        weights = {}

        for ticker, quantity in (
            self.positions.items()
        ):
            if ticker not in normalized_prices:
                continue

            specification = (
                MarketSpecificationRegistry.get(
                    ticker
                )
            )

            notional = (
                normalized_prices[
                    ticker
                ]
                *
                float(quantity)
                *
                specification.contract_multiplier
            )

            weights[
                ticker
            ] = (
                notional
                /
                self.equity
            )

        return weights


    def calculate_exposures(
        self,
        prices
    ):
        weights = (
            self.calculate_current_weights(
                prices
            )
        )

        gross_exposure = sum(
            abs(
                float(weight)
            )
            for weight in weights.values()
        )

        net_exposure = sum(
            float(weight)
            for weight in weights.values()
        )

        return {
            "weights": weights,
            "gross_exposure": (
                gross_exposure
            ),
            "net_exposure": (
                net_exposure
            ),
            "cash_weight": max(
                0.0,
                1.0
                -
                gross_exposure
            )
        }


    def calculate_target_quantity(
        self,
        ticker,
        target_weight,
        reference_price,
        equity_reference
    ):
        specification = (
            MarketSpecificationRegistry.get(
                ticker
            )
        )

        target_notional = (
            float(target_weight)
            *
            float(equity_reference)
        )

        raw_quantity = (
            target_notional
            /
            (
                float(reference_price)
                *
                specification.contract_multiplier
            )
        )

        direction = (
            1.0
            if raw_quantity >= 0
            else -1.0
        )

        normalized_absolute_quantity = (
            specification.normalize_quantity(
                requested_quantity=abs(
                    raw_quantity
                ),
                price=reference_price
            )
        )

        if normalized_absolute_quantity == 0:
            return 0.0

        return (
            normalized_absolute_quantity
            *
            direction
        )


    def calculate_execution_cost(
        self,
        ticker,
        reference_price,
        trade_quantity
    ):
        specification = (
            MarketSpecificationRegistry.get(
                ticker
            )
        )

        normalized_quantity = abs(
            float(trade_quantity)
        )

        if normalized_quantity == 0:
            return {
                "side": None,
                "fill_price": float(
                    reference_price
                ),
                "commission": 0.0,
                "slippage_cost": 0.0,
                "total_cost": 0.0,
                "trade_notional": 0.0
            }

        side = (
            "BUY"
            if trade_quantity > 0
            else "SELL"
        )

        if self.use_market_costs:
            fill_price = (
                specification
                .calculate_fill_price(
                    reference_price=(
                        reference_price
                    ),
                    side=side
                )
            )

            commission = (
                specification
                .calculate_commission(
                    price=fill_price,
                    quantity=(
                        normalized_quantity
                    )
                )
            )

        else:
            if side == "BUY":
                fill_price = (
                    float(reference_price)
                    *
                    (
                        1
                        +
                        self.slippage_percent
                    )
                )

            else:
                fill_price = (
                    float(reference_price)
                    *
                    (
                        1
                        -
                        self.slippage_percent
                    )
                )

            notional = (
                specification.calculate_notional(
                    price=fill_price,
                    quantity=(
                        normalized_quantity
                    )
                )
            )

            commission = (
                notional
                *
                self.commission_percent
            )

        slippage_cost = (
            abs(
                float(fill_price)
                -
                float(reference_price)
            )
            *
            normalized_quantity
            *
            specification.contract_multiplier
        )

        trade_notional = (
            float(reference_price)
            *
            normalized_quantity
            *
            specification.contract_multiplier
        )

        total_cost = (
            float(commission)
            +
            float(slippage_cost)
        )

        return {
            "side": side,
            "fill_price": float(
                fill_price
            ),
            "commission": float(
                commission
            ),
            "slippage_cost": float(
                slippage_cost
            ),
            "total_cost": float(
                total_cost
            ),
            "trade_notional": float(
                trade_notional
            )
        }


    def rebalance(
        self,
        target_weights,
        prices,
        timestamp,
        source_signal_time=None,
        reason="REBALANCE"
    ):
        """
        Porta le posizioni verso i pesi target.

        I pesi mancanti vengono interpretati
        come target pari a zero.
        """

        if not isinstance(
            target_weights,
            dict
        ):
            raise TypeError(
                "target_weights deve essere "
                "un dizionario."
            )

        if self.bankrupt:
            return {
                "timestamp": timestamp,
                "orders": [],
                "turnover_notional": 0.0,
                "turnover_ratio": 0.0,
                "total_cost": 0.0,
                "equity_before": 0.0,
                "equity_after": 0.0,
                "skipped": {
                    "portfolio": (
                        "Portfolio insolvente."
                    )
                }
            }

        normalized_prices = (
            self.normalize_prices(
                prices
            )
        )

        normalized_targets = {}

        for ticker, weight in (
            target_weights.items()
        ):
            normalized_ticker = (
                self.normalize_ticker(
                    ticker
                )
            )

            normalized_weight = float(
                weight
            )

            if not math.isfinite(
                normalized_weight
            ):
                raise ValueError(
                    f"Peso non finito per "
                    f"{normalized_ticker}."
                )

            normalized_targets[
                normalized_ticker
            ] = normalized_weight

        all_tickers = sorted(
            set(
                self.positions
            )
            |
            set(
                normalized_targets
            )
        )

        equity_before = float(
            self.equity
        )

        desired_quantities = {}
        skipped = {}

        for ticker in all_tickers:
            if ticker not in normalized_prices:
                skipped[
                    ticker
                ] = (
                    "Prezzo non disponibile."
                )

                desired_quantities[
                    ticker
                ] = self.positions.get(
                    ticker,
                    0.0
                )

                continue

            target_weight = (
                normalized_targets.get(
                    ticker,
                    0.0
                )
            )

            desired_quantities[
                ticker
            ] = (
                self.calculate_target_quantity(
                    ticker=ticker,
                    target_weight=target_weight,
                    reference_price=(
                        normalized_prices[
                            ticker
                        ]
                    ),
                    equity_reference=(
                        equity_before
                    )
                )
            )

        orders = []
        total_cost = 0.0
        turnover_notional = 0.0

        for ticker in all_tickers:
            current_quantity = float(
                self.positions.get(
                    ticker,
                    0.0
                )
            )

            target_quantity = float(
                desired_quantities.get(
                    ticker,
                    current_quantity
                )
            )

            trade_quantity = (
                target_quantity
                -
                current_quantity
            )

            if abs(trade_quantity) < 1e-15:
                continue

            reference_price = (
                normalized_prices[
                    ticker
                ]
            )

            cost_data = (
                self.calculate_execution_cost(
                    ticker=ticker,
                    reference_price=(
                        reference_price
                    ),
                    trade_quantity=(
                        trade_quantity
                    )
                )
            )

            if (
                cost_data[
                    "total_cost"
                ]
                >=
                self.equity
            ):
                skipped[
                    ticker
                ] = (
                    "Costi superiori "
                    "all'equity disponibile."
                )

                continue

            self.positions[
                ticker
            ] = target_quantity

            self.previous_prices[
                ticker
            ] = reference_price

            self.equity -= (
                cost_data[
                    "total_cost"
                ]
            )

            self.total_commissions += (
                cost_data[
                    "commission"
                ]
            )

            self.total_slippage_cost += (
                cost_data[
                    "slippage_cost"
                ]
            )

            self.total_costs += (
                cost_data[
                    "total_cost"
                ]
            )

            turnover_notional += (
                cost_data[
                    "trade_notional"
                ]
            )

            total_cost += (
                cost_data[
                    "total_cost"
                ]
            )

            order = {
                "timestamp": timestamp,
                "source_signal_time": (
                    source_signal_time
                ),
                "reason": reason,
                "ticker": ticker,
                "side": (
                    cost_data["side"]
                ),
                "reference_price": (
                    reference_price
                ),
                "fill_price": (
                    cost_data[
                        "fill_price"
                    ]
                ),
                "previous_quantity": (
                    current_quantity
                ),
                "trade_quantity": (
                    trade_quantity
                ),
                "target_quantity": (
                    target_quantity
                ),
                "target_weight": (
                    normalized_targets.get(
                        ticker,
                        0.0
                    )
                ),
                "trade_notional": (
                    cost_data[
                        "trade_notional"
                    ]
                ),
                "commission": (
                    cost_data[
                        "commission"
                    ]
                ),
                "slippage_cost": (
                    cost_data[
                        "slippage_cost"
                    ]
                ),
                "total_cost": (
                    cost_data[
                        "total_cost"
                    ]
                )
            }

            orders.append(
                order
            )

            self.orders.append(
                deepcopy(
                    order
                )
            )

        self.positions = {
            ticker: quantity
            for ticker, quantity
            in self.positions.items()
            if abs(
                float(quantity)
            ) > 1e-15
        }

        turnover_ratio = (
            turnover_notional
            /
            equity_before
            if equity_before > 0
            else 0.0
        )

        self.total_turnover_notional += (
            turnover_notional
        )

        self.total_turnover_ratio += (
            turnover_ratio
        )

        if self.equity <= 0:
            self.equity = 0.0
            self.bankrupt = True

        exposures = (
            self.calculate_exposures(
                normalized_prices
            )
        )

        return {
            "timestamp": timestamp,
            "source_signal_time": (
                source_signal_time
            ),
            "reason": reason,
            "orders": orders,
            "turnover_notional": float(
                turnover_notional
            ),
            "turnover_ratio": float(
                turnover_ratio
            ),
            "total_cost": float(
                total_cost
            ),
            "equity_before": (
                equity_before
            ),
            "equity_after": float(
                self.equity
            ),
            "actual_weights": (
                exposures["weights"]
            ),
            "gross_exposure": (
                exposures[
                    "gross_exposure"
                ]
            ),
            "net_exposure": (
                exposures[
                    "net_exposure"
                ]
            ),
            "skipped": skipped
        }


    def liquidate(
        self,
        prices,
        timestamp,
        reason="FINAL_LIQUIDATION"
    ):
        zero_targets = {
            ticker: 0.0
            for ticker in self.positions
        }

        return self.rebalance(
            target_weights=zero_targets,
            prices=prices,
            timestamp=timestamp,
            source_signal_time=None,
            reason=reason
        )


    def record_equity(
        self,
        timestamp,
        prices
    ):
        exposures = (
            self.calculate_exposures(
                prices
            )
        )

        snapshot = {
            "timestamp": timestamp,
            "equity": float(
                self.equity
            ),
            "daily_pnl": float(
                self.realized_daily_pnl
            ),
            "position_count": (
                self.position_count
            ),
            "gross_exposure": float(
                exposures[
                    "gross_exposure"
                ]
            ),
            "net_exposure": float(
                exposures[
                    "net_exposure"
                ]
            ),
            "cash_weight": float(
                exposures[
                    "cash_weight"
                ]
            )
        }

        self.equity_curve.append(
            snapshot
        )

        return deepcopy(
            snapshot
        )