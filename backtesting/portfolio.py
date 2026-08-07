from copy import deepcopy

from market.specifications import (
    MarketSpecificationRegistry
)


class BacktestPortfolio:
    """
    Portfolio per backtest con specifiche
    economiche dipendenti dal mercato.

    Supporta:

    - azioni ed ETF con quantità intere;
    - crypto con quantità frazionarie;
    - forex con quantità normalizzate;
    - futures con moltiplicatore contrattuale;
    - margine configurabile per strumenti
      con margin_model='external';
    - costi percentuali legacy;
    - costi specifici del mercato.
    """

    def __init__(
        self,
        initial_capital=10000.0,
        max_position_percent=0.10,
        commission_percent=0.001,
        slippage_percent=0.0005,
        use_market_costs=False,
        margin_overrides=None
    ):
        if initial_capital <= 0:
            raise ValueError(
                "initial_capital deve essere maggiore di zero."
            )

        if not 0 < max_position_percent <= 0.10:
            raise ValueError(
                "max_position_percent deve essere "
                "compreso tra 0 e 0.10."
            )

        if commission_percent is not None:
            if commission_percent < 0:
                raise ValueError(
                    "commission_percent non può essere negativo."
                )

        if slippage_percent is not None:
            if slippage_percent < 0:
                raise ValueError(
                    "slippage_percent non può essere negativo."
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

        for ticker, margin_value in (
            margin_overrides.items()
        ):
            normalized_ticker = (
                str(ticker)
                .upper()
                .strip()
            )

            normalized_margin = float(
                margin_value
            )

            if normalized_margin <= 0:
                raise ValueError(
                    "Ogni margine per contratto "
                    "deve essere positivo."
                )

            normalized_margin_overrides[
                normalized_ticker
            ] = normalized_margin

        self.initial_capital = float(
            initial_capital
        )

        self.cash = float(
            initial_capital
        )

        self.max_position_percent = float(
            max_position_percent
        )

        self.commission_percent = (
            float(
                commission_percent
            )
            if commission_percent is not None
            else 0.0
        )

        self.slippage_percent = (
            float(
                slippage_percent
            )
            if slippage_percent is not None
            else 0.0
        )

        self.use_market_costs = bool(
            use_market_costs
        )

        self.margin_overrides = (
            normalized_margin_overrides
        )

        self.open_position = None
        self.closed_trades = []
        self.equity_curve = []

        self.total_commissions = 0.0


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


    def get_specification(
        self,
        ticker
    ):
        """
        Restituisce le specifiche dello strumento.
        """

        normalized_ticker = (
            self.normalize_ticker(
                ticker
            )
        )

        return (
            MarketSpecificationRegistry.get(
                normalized_ticker
            )
        )


    def calculate_available_position_capital(
        self
    ):
        """
        Capitale massimo impegnabile nella
        singola posizione.
        """

        return max(
            0.0,
            self.cash
            *
            self.max_position_percent
        )


    def calculate_fill_price(
        self,
        specification,
        reference_price,
        side
    ):
        """
        Calcola il prezzo di esecuzione.

        In modalità market costs usa spread,
        tick e slippage dello strumento.

        In modalità legacy applica soltanto
        lo slippage percentuale.
        """

        reference_price = float(
            reference_price
        )

        if reference_price <= 0:
            raise ValueError(
                "reference_price deve essere positivo."
            )

        normalized_side = (
            str(side)
            .upper()
            .strip()
        )

        if normalized_side not in {
            "BUY",
            "SELL"
        }:
            raise ValueError(
                "side deve essere BUY o SELL."
            )

        if self.use_market_costs:
            return (
                specification
                .calculate_fill_price(
                    reference_price=(
                        reference_price
                    ),
                    side=normalized_side
                )
            )

        if normalized_side == "BUY":
            return (
                reference_price
                *
                (
                    1
                    +
                    self.slippage_percent
                )
            )

        return (
            reference_price
            *
            (
                1
                -
                self.slippage_percent
            )
        )


    def calculate_commission(
        self,
        specification,
        fill_price,
        quantity
    ):
        """
        Calcola la commissione per un lato.
        """

        if self.use_market_costs:
            return (
                specification
                .calculate_commission(
                    price=fill_price,
                    quantity=quantity
                )
            )

        notional = (
            specification
            .calculate_notional(
                price=fill_price,
                quantity=quantity
            )
        )

        return (
            notional
            *
            self.commission_percent
        )


    def calculate_capital_required(
        self,
        specification,
        ticker,
        price,
        quantity
    ):
        """
        Calcola il capitale o margine
        da riservare per la posizione.
        """

        if (
            specification.margin_model
            == "external"
        ):
            margin_per_contract = (
                self.margin_overrides.get(
                    ticker
                )
            )

            if margin_per_contract is None:
                return None

            return (
                float(quantity)
                *
                margin_per_contract
            )

        return (
            specification
            .calculate_capital_required(
                price=price,
                quantity=quantity
            )
        )


    def calculate_requested_quantity(
        self,
        specification,
        ticker,
        fill_price,
        capital_budget
    ):
        """
        Calcola la quantità teorica in base
        al capitale massimo disponibile.
        """

        if capital_budget <= 0:
            return 0.0

        if (
            specification.margin_model
            == "external"
        ):
            margin_per_contract = (
                self.margin_overrides.get(
                    ticker
                )
            )

            if margin_per_contract is None:
                return 0.0

            return (
                capital_budget
                /
                margin_per_contract
            )

        if (
            specification.margin_model
            == "per_contract"
        ):
            return (
                capital_budget
                /
                float(
                    specification.margin_value
                )
            )

        notional_per_unit = (
            float(fill_price)
            *
            specification.contract_multiplier
        )

        if (
            specification.margin_model
            == "percentage_notional"
        ):
            capital_per_unit = (
                notional_per_unit
                *
                float(
                    specification.margin_value
                )
            )

        else:
            capital_per_unit = (
                notional_per_unit
            )

        if capital_per_unit <= 0:
            return 0.0

        return (
            capital_budget
            /
            capital_per_unit
        )


    def fit_quantity_to_limits(
        self,
        specification,
        ticker,
        fill_price,
        requested_quantity,
        capital_budget
    ):
        """
        Riduce la quantità finché capitale,
        commissione e limiti risultano validi.
        """

        quantity = (
            specification
            .normalize_quantity(
                requested_quantity=(
                    requested_quantity
                ),
                price=fill_price
            )
        )

        while quantity > 0:
            capital_required = (
                self.calculate_capital_required(
                    specification=(
                        specification
                    ),
                    ticker=ticker,
                    price=fill_price,
                    quantity=quantity
                )
            )

            if capital_required is None:
                return 0.0

            commission = (
                self.calculate_commission(
                    specification=(
                        specification
                    ),
                    fill_price=fill_price,
                    quantity=quantity
                )
            )

            within_position_budget = (
                capital_required
                <= capital_budget
            )

            within_available_cash = (
                capital_required
                +
                commission
                <= self.cash
            )

            if (
                within_position_budget
                and within_available_cash
            ):
                return quantity

            quantity = (
                specification
                .normalize_quantity(
                    requested_quantity=(
                        quantity
                        -
                        specification.quantity_step
                    ),
                    price=fill_price
                )
            )

        return 0.0


    def open_trade(
        self,
        signal,
        timestamp,
        asset_type=None
    ):
        """
        Apre una posizione.

        asset_type è mantenuto soltanto per
        compatibilità con il codice precedente.
        Le specifiche vengono ricavate dal ticker.
        """

        if self.open_position is not None:
            return None

        if not isinstance(
            signal,
            dict
        ):
            raise TypeError(
                "signal deve essere un dizionario."
            )

        ticker = self.normalize_ticker(
            signal.get(
                "ticker",
                ""
            )
        )

        direction = (
            str(
                signal.get(
                    "direction",
                    ""
                )
            )
            .upper()
            .strip()
        )

        if direction not in {
            "LONG",
            "SHORT"
        }:
            raise ValueError(
                "direction deve essere LONG o SHORT."
            )

        reference_entry_price = float(
            signal.get(
                "entry_price",
                0
            )
        )

        if reference_entry_price <= 0:
            raise ValueError(
                "entry_price deve essere positivo."
            )

        specification = (
            self.get_specification(
                ticker
            )
        )

        entry_side = (
            "BUY"
            if direction == "LONG"
            else "SELL"
        )

        entry_fill_price = (
            self.calculate_fill_price(
                specification=specification,
                reference_price=(
                    reference_entry_price
                ),
                side=entry_side
            )
        )

        capital_budget = (
            self.calculate_available_position_capital()
        )

        requested_quantity = (
            self.calculate_requested_quantity(
                specification=specification,
                ticker=ticker,
                fill_price=(
                    entry_fill_price
                ),
                capital_budget=(
                    capital_budget
                )
            )
        )

        quantity = (
            self.fit_quantity_to_limits(
                specification=specification,
                ticker=ticker,
                fill_price=(
                    entry_fill_price
                ),
                requested_quantity=(
                    requested_quantity
                ),
                capital_budget=(
                    capital_budget
                )
            )
        )

        if quantity <= 0:
            return None

        capital_required = (
            self.calculate_capital_required(
                specification=specification,
                ticker=ticker,
                price=entry_fill_price,
                quantity=quantity
            )
        )

        if capital_required is None:
            return None

        entry_notional = (
            specification
            .calculate_notional(
                price=entry_fill_price,
                quantity=quantity
            )
        )

        entry_commission = (
            self.calculate_commission(
                specification=specification,
                fill_price=entry_fill_price,
                quantity=quantity
            )
        )

        total_entry_cash = (
            capital_required
            +
            entry_commission
        )

        if total_entry_cash > self.cash:
            return None

        self.cash -= total_entry_cash

        self.total_commissions += (
            entry_commission
        )

        self.open_position = {
            "ticker": ticker,
            "asset_type": (
                specification.asset_class
            ),
            "specification_code": (
                specification.code
            ),
            "direction": direction,
            "quantity": quantity,
            "contract_multiplier": (
                specification
                .contract_multiplier
            ),
            "tick_size": (
                specification.tick_size
            ),
            "tick_value": (
                specification.tick_value
            ),
            "reference_entry_price": (
                reference_entry_price
            ),
            "entry_price": (
                entry_fill_price
            ),
            "entry_notional": (
                entry_notional
            ),
            "position_value": (
                entry_notional
            ),
            "capital_required": (
                capital_required
            ),
            "entry_commission": (
                entry_commission
            ),
            "stop_loss": float(
                signal["stop_loss"]
            ),
            "take_profit": float(
                signal["take_profit"]
            ),
            "confidence": float(
                signal.get(
                    "confidence",
                    0
                )
            ),
            "risk_reward": float(
                signal.get(
                    "risk_reward",
                    0
                )
            ),
            "strategy": signal.get(
                "strategy",
                "unknown"
            ),
            "reasons": list(
                signal.get(
                    "reasons",
                    []
                )
            ),
            "open_time": timestamp
        }

        return deepcopy(
            self.open_position
        )


    def close_trade(
        self,
        exit_price,
        timestamp,
        reason
    ):
        """
        Chiude la posizione e calcola il P/L
        usando il moltiplicatore dello strumento.
        """

        if self.open_position is None:
            return None

        position = self.open_position

        ticker = position[
            "ticker"
        ]

        specification = (
            self.get_specification(
                ticker
            )
        )

        reference_exit_price = float(
            exit_price
        )

        if reference_exit_price <= 0:
            raise ValueError(
                "exit_price deve essere positivo."
            )

        exit_side = (
            "SELL"
            if position["direction"] == "LONG"
            else "BUY"
        )

        exit_fill_price = (
            self.calculate_fill_price(
                specification=specification,
                reference_price=(
                    reference_exit_price
                ),
                side=exit_side
            )
        )

        quantity = float(
            position["quantity"]
        )

        gross_pnl = (
            specification
            .calculate_pnl(
                entry_price=(
                    position[
                        "entry_price"
                    ]
                ),
                exit_price=(
                    exit_fill_price
                ),
                quantity=quantity,
                direction=(
                    position[
                        "direction"
                    ]
                )
            )
        )

        exit_notional = (
            specification
            .calculate_notional(
                price=exit_fill_price,
                quantity=quantity
            )
        )

        exit_commission = (
            self.calculate_commission(
                specification=specification,
                fill_price=exit_fill_price,
                quantity=quantity
            )
        )

        entry_commission = float(
            position[
                "entry_commission"
            ]
        )

        net_pnl = (
            gross_pnl
            -
            entry_commission
            -
            exit_commission
        )

        capital_required = float(
            position[
                "capital_required"
            ]
        )

        self.cash += (
            capital_required
            +
            gross_pnl
            -
            exit_commission
        )

        self.total_commissions += (
            exit_commission
        )

        return_percent = (
            net_pnl
            /
            capital_required
            *
            100
            if capital_required > 0
            else 0.0
        )

        trade = {
            **position,
            "reference_exit_price": (
                reference_exit_price
            ),
            "exit_price": (
                exit_fill_price
            ),
            "exit_notional": (
                exit_notional
            ),
            "exit_commission": (
                exit_commission
            ),
            "gross_pnl": (
                gross_pnl
            ),
            "pnl": net_pnl,
            "return_percent": (
                return_percent
            ),
            "close_time": timestamp,
            "reason": reason
        }

        self.closed_trades.append(
            trade
        )

        self.open_position = None

        return deepcopy(
            trade
        )


    def calculate_equity(
        self,
        current_price
    ):
        """
        Calcola l'equity mark-to-market.
        """

        if self.open_position is None:
            return float(
                self.cash
            )

        position = self.open_position

        specification = (
            self.get_specification(
                position["ticker"]
            )
        )

        current_price = float(
            current_price
        )

        if current_price <= 0:
            raise ValueError(
                "current_price deve essere positivo."
            )

        unrealized_gross_pnl = (
            specification
            .calculate_pnl(
                entry_price=(
                    position[
                        "entry_price"
                    ]
                ),
                exit_price=current_price,
                quantity=(
                    position[
                        "quantity"
                    ]
                ),
                direction=(
                    position[
                        "direction"
                    ]
                )
            )
        )

        estimated_exit_side = (
            "SELL"
            if position["direction"] == "LONG"
            else "BUY"
        )

        estimated_exit_fill = (
            self.calculate_fill_price(
                specification=specification,
                reference_price=(
                    current_price
                ),
                side=estimated_exit_side
            )
        )

        estimated_exit_commission = (
            self.calculate_commission(
                specification=specification,
                fill_price=(
                    estimated_exit_fill
                ),
                quantity=(
                    position[
                        "quantity"
                    ]
                )
            )
        )

        return float(
            self.cash
            +
            position[
                "capital_required"
            ]
            +
            unrealized_gross_pnl
            -
            estimated_exit_commission
        )


    def record_equity(
        self,
        timestamp,
        equity
    ):
        self.equity_curve.append(
            {
                "timestamp": timestamp,
                "equity": float(
                    equity
                )
            }
        )