from dataclasses import dataclass, field, replace
from decimal import Decimal, ROUND_DOWN


@dataclass(frozen=True)
class ExecutionCostProfile:
    """
    Modello configurabile dei costi operativi.

    I valori predefiniti del registry sono ipotesi
    conservative per la ricerca, non tariffe
    garantite di uno specifico broker.
    """

    name: str

    commission_model: str = (
        "percentage_notional"
    )

    commission_value: float = 0.0
    minimum_commission: float = 0.0

    spread_bps: float = 0.0
    slippage_bps: float = 0.0

    spread_ticks: float = 0.0
    slippage_ticks: float = 0.0

    ALLOWED_COMMISSION_MODELS = {
        "percentage_notional",
        "per_unit",
        "per_contract",
        "fixed"
    }

    def __post_init__(self):
        if (
            self.commission_model
            not in
            self.ALLOWED_COMMISSION_MODELS
        ):
            raise ValueError(
                "Modello commissionale non valido: "
                f"{self.commission_model}"
            )

        numeric_values = {
            "commission_value": (
                self.commission_value
            ),
            "minimum_commission": (
                self.minimum_commission
            ),
            "spread_bps": self.spread_bps,
            "slippage_bps": (
                self.slippage_bps
            ),
            "spread_ticks": (
                self.spread_ticks
            ),
            "slippage_ticks": (
                self.slippage_ticks
            )
        }

        for name, value in (
            numeric_values.items()
        ):
            if value < 0:
                raise ValueError(
                    f"{name} non può essere negativo."
                )


    def calculate_commission(
        self,
        notional,
        quantity
    ):
        """
        Calcola la commissione per un singolo lato
        dell'operazione.
        """

        notional = abs(
            float(notional)
        )

        quantity = abs(
            float(quantity)
        )

        if quantity == 0:
            return 0.0

        if (
            self.commission_model
            == "percentage_notional"
        ):
            raw_commission = (
                notional *
                self.commission_value
            )

        elif (
            self.commission_model
            == "per_unit"
        ):
            raw_commission = (
                quantity *
                self.commission_value
            )

        elif (
            self.commission_model
            == "per_contract"
        ):
            raw_commission = (
                quantity *
                self.commission_value
            )

        else:
            raw_commission = (
                self.commission_value
            )

        return max(
            float(raw_commission),
            float(
                self.minimum_commission
            )
        )


    def calculate_price_adjustment(
        self,
        reference_price,
        tick_size
    ):
        """
        Calcola l'aggiustamento avverso del prezzo
        dovuto a metà spread e slippage.

        Lo spread viene diviso tra ingresso e uscita.
        Lo slippage viene applicato interamente
        a ciascun lato.
        """

        reference_price = float(
            reference_price
        )

        tick_size = float(
            tick_size
        )

        if reference_price <= 0:
            raise ValueError(
                "reference_price deve essere positivo."
            )

        if tick_size <= 0:
            raise ValueError(
                "tick_size deve essere positivo."
            )

        bps_adjustment = (
            reference_price
            *
            (
                self.spread_bps / 2
                +
                self.slippage_bps
            )
            /
            10_000
        )

        tick_adjustment = (
            tick_size
            *
            (
                self.spread_ticks / 2
                +
                self.slippage_ticks
            )
        )

        return (
            bps_adjustment
            +
            tick_adjustment
        )


    def calculate_fill_price(
        self,
        reference_price,
        side,
        tick_size
    ):
        """
        BUY riceve un prezzo più alto.
        SELL riceve un prezzo più basso.
        """

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

        adjustment = (
            self.calculate_price_adjustment(
                reference_price=(
                    reference_price
                ),
                tick_size=tick_size
            )
        )

        if normalized_side == "BUY":
            return (
                float(reference_price)
                +
                adjustment
            )

        return (
            float(reference_price)
            -
            adjustment
        )


@dataclass(frozen=True)
class MarketSpecification:
    """
    Specifiche economiche e operative
    di uno strumento finanziario.
    """

    code: str
    asset_class: str
    quote_currency: str

    quantity_step: float
    minimum_quantity: float

    contract_multiplier: float = 1.0
    tick_size: float = 0.01

    minimum_notional: float = 0.0
    supports_fractional: bool = False

    margin_model: str = "notional"
    margin_value: float | None = 1.0

    cost_profile: ExecutionCostProfile = field(
        default_factory=lambda: (
            ExecutionCostProfile(
                name="zero_cost"
            )
        )
    )

    ALLOWED_ASSET_CLASSES = {
        "STOCK",
        "ETF",
        "CRYPTO",
        "FOREX",
        "FUTURE"
    }

    ALLOWED_MARGIN_MODELS = {
        "notional",
        "percentage_notional",
        "per_contract",
        "external"
    }

    def __post_init__(self):
        if not self.code:
            raise ValueError(
                "code non può essere vuoto."
            )

        if (
            self.asset_class
            not in
            self.ALLOWED_ASSET_CLASSES
        ):
            raise ValueError(
                "Asset class non valida: "
                f"{self.asset_class}"
            )

        if self.quantity_step <= 0:
            raise ValueError(
                "quantity_step deve essere positivo."
            )

        if self.minimum_quantity <= 0:
            raise ValueError(
                "minimum_quantity deve essere positiva."
            )

        if self.contract_multiplier <= 0:
            raise ValueError(
                "contract_multiplier deve essere positivo."
            )

        if self.tick_size <= 0:
            raise ValueError(
                "tick_size deve essere positivo."
            )

        if self.minimum_notional < 0:
            raise ValueError(
                "minimum_notional non può essere negativo."
            )

        if (
            self.margin_model
            not in
            self.ALLOWED_MARGIN_MODELS
        ):
            raise ValueError(
                "margin_model non valido: "
                f"{self.margin_model}"
            )

        if (
            self.margin_model != "external"
            and (
                self.margin_value is None
                or self.margin_value <= 0
            )
        ):
            raise ValueError(
                "margin_value deve essere positivo "
                "quando il margine non è external."
            )


    @property
    def tick_value(self):
        """
        Valore monetario di un tick per unità
        o per contratto.
        """

        return (
            self.tick_size
            *
            self.contract_multiplier
        )


    @staticmethod
    def _decimal_places(
        value
    ):
        decimal_value = Decimal(
            str(value)
        ).normalize()

        return max(
            0,
            -decimal_value.as_tuple().exponent
        )


    def normalize_quantity(
        self,
        requested_quantity,
        price=None
    ):
        """
        Arrotonda la quantità verso il basso
        rispettando step, minimo e notional minimo.

        Restituisce zero quando l'ordine
        non è eseguibile.
        """

        requested_quantity = float(
            requested_quantity
        )

        if requested_quantity < 0:
            raise ValueError(
                "requested_quantity non può essere negativa."
            )

        if requested_quantity == 0:
            return 0.0

        quantity_decimal = Decimal(
            str(requested_quantity)
        )

        step_decimal = Decimal(
            str(self.quantity_step)
        )

        step_count = (
            quantity_decimal /
            step_decimal
        ).to_integral_value(
            rounding=ROUND_DOWN
        )

        normalized_decimal = (
            step_count *
            step_decimal
        )

        normalized = float(
            normalized_decimal
        )

        if (
            normalized
            <
            self.minimum_quantity
        ):
            return 0.0

        precision = self._decimal_places(
            self.quantity_step
        )

        normalized = round(
            normalized,
            precision
        )

        if price is not None:
            if float(price) <= 0:
                raise ValueError(
                    "price deve essere positivo."
                )

            notional = self.calculate_notional(
                price=price,
                quantity=normalized
            )

            if (
                notional
                <
                self.minimum_notional
            ):
                return 0.0

        return normalized


    def calculate_notional(
        self,
        price,
        quantity
    ):
        """
        Valore nozionale della posizione.
        """

        price = float(
            price
        )

        quantity = float(
            quantity
        )

        if price <= 0:
            raise ValueError(
                "price deve essere positivo."
            )

        if quantity < 0:
            raise ValueError(
                "quantity non può essere negativa."
            )

        return (
            price
            *
            quantity
            *
            self.contract_multiplier
        )


    def calculate_capital_required(
        self,
        price,
        quantity
    ):
        """
        Calcola il capitale o margine necessario.

        Per i futures con margin_model='external'
        il valore deve essere fornito dal broker
        o da una configurazione aggiornata.
        """

        quantity = float(
            quantity
        )

        if quantity < 0:
            raise ValueError(
                "quantity non può essere negativa."
            )

        notional = self.calculate_notional(
            price=price,
            quantity=quantity
        )

        if self.margin_model == "notional":
            return notional

        if (
            self.margin_model
            == "percentage_notional"
        ):
            return (
                notional
                *
                float(
                    self.margin_value
                )
            )

        if (
            self.margin_model
            == "per_contract"
        ):
            return (
                quantity
                *
                float(
                    self.margin_value
                )
            )

        raise ValueError(
            f"Margine non configurato per {self.code}. "
            "Serve un valore aggiornato del broker."
        )


    def calculate_pnl(
        self,
        entry_price,
        exit_price,
        quantity,
        direction
    ):
        """
        Calcola il P/L lordo rispettando
        il moltiplicatore contrattuale.
        """

        entry_price = float(
            entry_price
        )

        exit_price = float(
            exit_price
        )

        quantity = float(
            quantity
        )

        normalized_direction = (
            str(direction)
            .upper()
            .strip()
        )

        if entry_price <= 0:
            raise ValueError(
                "entry_price deve essere positivo."
            )

        if exit_price <= 0:
            raise ValueError(
                "exit_price deve essere positivo."
            )

        if quantity < 0:
            raise ValueError(
                "quantity non può essere negativa."
            )

        if normalized_direction == "LONG":
            direction_multiplier = 1

        elif normalized_direction == "SHORT":
            direction_multiplier = -1

        else:
            raise ValueError(
                "direction deve essere LONG o SHORT."
            )

        return (
            (
                exit_price
                -
                entry_price
            )
            *
            quantity
            *
            self.contract_multiplier
            *
            direction_multiplier
        )


    def calculate_commission(
        self,
        price,
        quantity
    ):
        notional = self.calculate_notional(
            price=price,
            quantity=quantity
        )

        return (
            self.cost_profile
            .calculate_commission(
                notional=notional,
                quantity=quantity
            )
        )


    def calculate_fill_price(
        self,
        reference_price,
        side
    ):
        return (
            self.cost_profile
            .calculate_fill_price(
                reference_price=(
                    reference_price
                ),
                side=side,
                tick_size=self.tick_size
            )
        )


class MarketSpecificationRegistry:
    """
    Registry centralizzato delle specifiche.

    I futures sconosciuti vengono rifiutati:
    non è sicuro assumere moltiplicatore 1
    per un contratto non configurato.
    """

    STOCK_COSTS = ExecutionCostProfile(
        name="stock_research_default",
        commission_model=(
            "percentage_notional"
        ),
        commission_value=0.0005,
        minimum_commission=1.0,
        spread_bps=1.0,
        slippage_bps=2.0
    )

    ETF_COSTS = ExecutionCostProfile(
        name="etf_research_default",
        commission_model=(
            "percentage_notional"
        ),
        commission_value=0.0005,
        minimum_commission=1.0,
        spread_bps=0.5,
        slippage_bps=1.0
    )

    CRYPTO_COSTS = ExecutionCostProfile(
        name="crypto_research_default",
        commission_model=(
            "percentage_notional"
        ),
        commission_value=0.001,
        minimum_commission=0.0,
        spread_bps=5.0,
        slippage_bps=5.0
    )

    FOREX_COSTS = ExecutionCostProfile(
        name="forex_research_default",
        commission_model=(
            "percentage_notional"
        ),
        commission_value=0.00002,
        minimum_commission=0.0,
        spread_bps=1.0,
        slippage_bps=0.5
    )

    FUTURES_COSTS = ExecutionCostProfile(
        name="futures_research_default",
        commission_model="per_contract",
        commission_value=2.50,
        minimum_commission=0.0,
        spread_ticks=1.0,
        slippage_ticks=1.0
    )

    ETF_TICKERS = {
        "SPY",
        "QQQ",
        "VTI",
        "IWM",
        "DIA",
        "TLT",
        "GLD"
    }

    EXACT_SPECIFICATIONS = {
        "GC=F": MarketSpecification(
            code="GC=F",
            asset_class="FUTURE",
            quote_currency="USD",
            quantity_step=1.0,
            minimum_quantity=1.0,
            contract_multiplier=100.0,
            tick_size=0.10,
            supports_fractional=False,
            margin_model="external",
            margin_value=None,
            cost_profile=FUTURES_COSTS
        ),

        "MGC=F": MarketSpecification(
            code="MGC=F",
            asset_class="FUTURE",
            quote_currency="USD",
            quantity_step=1.0,
            minimum_quantity=1.0,
            contract_multiplier=10.0,
            tick_size=0.10,
            supports_fractional=False,
            margin_model="external",
            margin_value=None,
            cost_profile=FUTURES_COSTS
        ),

        "CL=F": MarketSpecification(
            code="CL=F",
            asset_class="FUTURE",
            quote_currency="USD",
            quantity_step=1.0,
            minimum_quantity=1.0,
            contract_multiplier=1000.0,
            tick_size=0.01,
            supports_fractional=False,
            margin_model="external",
            margin_value=None,
            cost_profile=FUTURES_COSTS
        ),

        "MCL=F": MarketSpecification(
            code="MCL=F",
            asset_class="FUTURE",
            quote_currency="USD",
            quantity_step=1.0,
            minimum_quantity=1.0,
            contract_multiplier=100.0,
            tick_size=0.01,
            supports_fractional=False,
            margin_model="external",
            margin_value=None,
            cost_profile=FUTURES_COSTS
        )
    }

    STOCK_TEMPLATE = MarketSpecification(
        code="STOCK_TEMPLATE",
        asset_class="STOCK",
        quote_currency="USD",
        quantity_step=1.0,
        minimum_quantity=1.0,
        contract_multiplier=1.0,
        tick_size=0.01,
        supports_fractional=False,
        margin_model="notional",
        margin_value=1.0,
        cost_profile=STOCK_COSTS
    )

    ETF_TEMPLATE = MarketSpecification(
        code="ETF_TEMPLATE",
        asset_class="ETF",
        quote_currency="USD",
        quantity_step=1.0,
        minimum_quantity=1.0,
        contract_multiplier=1.0,
        tick_size=0.01,
        supports_fractional=False,
        margin_model="notional",
        margin_value=1.0,
        cost_profile=ETF_COSTS
    )

    CRYPTO_TEMPLATE = MarketSpecification(
        code="CRYPTO_TEMPLATE",
        asset_class="CRYPTO",
        quote_currency="USD",
        quantity_step=0.00000001,
        minimum_quantity=0.00000001,
        contract_multiplier=1.0,
        tick_size=0.01,
        minimum_notional=1.0,
        supports_fractional=True,
        margin_model="notional",
        margin_value=1.0,
        cost_profile=CRYPTO_COSTS
    )

    FOREX_TEMPLATE = MarketSpecification(
        code="FOREX_TEMPLATE",
        asset_class="FOREX",
        quote_currency="USD",
        quantity_step=1000.0,
        minimum_quantity=1000.0,
        contract_multiplier=1.0,
        tick_size=0.0001,
        minimum_notional=1000.0,
        supports_fractional=False,
        margin_model="notional",
        margin_value=1.0,
        cost_profile=FOREX_COSTS
    )


    @classmethod
    def register(
        cls,
        ticker,
        specification
    ):
        if not isinstance(
            specification,
            MarketSpecification
        ):
            raise TypeError(
                "specification deve essere "
                "MarketSpecification."
            )

        normalized_ticker = (
            str(ticker)
            .upper()
            .strip()
        )

        if not normalized_ticker:
            raise ValueError(
                "ticker non può essere vuoto."
            )

        cls.EXACT_SPECIFICATIONS[
            normalized_ticker
        ] = specification


    @classmethod
    def get(
        cls,
        ticker
    ):
        normalized_ticker = (
            str(ticker)
            .upper()
            .strip()
        )

        if not normalized_ticker:
            raise ValueError(
                "ticker non può essere vuoto."
            )

        if (
            normalized_ticker
            in cls.EXACT_SPECIFICATIONS
        ):
            return cls.EXACT_SPECIFICATIONS[
                normalized_ticker
            ]

        if normalized_ticker.endswith(
            "=F"
        ):
            raise KeyError(
                "Future non configurato: "
                f"{normalized_ticker}. "
                "Specificare moltiplicatore, tick "
                "e modello di margine."
            )

        if normalized_ticker.endswith(
            "-USD"
        ):
            return replace(
                cls.CRYPTO_TEMPLATE,
                code=normalized_ticker
            )

        if normalized_ticker.endswith(
            "=X"
        ):
            tick_size = (
                0.01
                if "JPY" in normalized_ticker
                else 0.0001
            )

            return replace(
                cls.FOREX_TEMPLATE,
                code=normalized_ticker,
                tick_size=tick_size
            )

        if (
            normalized_ticker
            in cls.ETF_TICKERS
        ):
            return replace(
                cls.ETF_TEMPLATE,
                code=normalized_ticker
            )

        return replace(
            cls.STOCK_TEMPLATE,
            code=normalized_ticker
        )