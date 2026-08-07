import pandas as pd

from analysis.indicators import (
    calculate_returns,
    calculate_sma,
    calculate_volatility,
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands
)
from analysis.analyzer import analyze_asset

from backtesting.performance import PerformanceAnalyzer
from backtesting.portfolio import BacktestPortfolio

from decision.market_decision_engine import MarketDecisionEngine
from market.specifications import MarketSpecificationRegistry
from signals.signal_generator import SignalGenerator
from validator.signal_validator import SignalValidator


class Backtester:
    """
    Motore di backtesting storico market-aware.

    Supporta:

    - strategia legacy;
    - strategie intercambiabili;
    - warm-up senza operazioni;
    - LONG e SHORT;
    - stop loss e take profit;
    - durata massima delle posizioni;
    - cooldown;
    - limite giornaliero dei trade;
    - costi percentuali legacy;
    - costi specifici per mercato;
    - quantità, margini e moltiplicatori
      dipendenti dallo strumento.
    """

    def __init__(
        self,
        initial_capital=10000.0,
        max_position_percent=0.10,
        minimum_history=30,
        commission_percent=0.001,
        slippage_percent=0.0005,
        max_holding_bars=78,
        cooldown_bars=12,
        max_trades_per_day=3,
        strategy=None,
        use_market_costs=False,
        margin_overrides=None
    ):
        if initial_capital <= 0:
            raise ValueError(
                "Il capitale iniziale deve essere maggiore di zero."
            )

        if not 0 < max_position_percent <= 0.10:
            raise ValueError(
                "L'esposizione massima deve essere "
                "compresa tra 0 e 0.10."
            )

        if minimum_history < 26:
            raise ValueError(
                "minimum_history deve essere almeno 26."
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

        if max_holding_bars <= 0:
            raise ValueError(
                "max_holding_bars deve essere maggiore di zero."
            )

        if cooldown_bars < 0:
            raise ValueError(
                "cooldown_bars non può essere negativo."
            )

        if max_trades_per_day <= 0:
            raise ValueError(
                "max_trades_per_day deve essere maggiore di zero."
            )

        if (
            strategy is not None
            and not hasattr(
                strategy,
                "generate_decision"
            )
        ):
            raise TypeError(
                "La strategia deve implementare generate_decision()."
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

        self.initial_capital = float(
            initial_capital
        )

        self.max_position_percent = float(
            max_position_percent
        )

        self.minimum_history = int(
            minimum_history
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

        self.max_holding_bars = int(
            max_holding_bars
        )

        self.cooldown_bars = int(
            cooldown_bars
        )

        self.max_trades_per_day = int(
            max_trades_per_day
        )

        self.strategy = strategy

        self.use_market_costs = bool(
            use_market_costs
        )

        self.margin_overrides = dict(
            margin_overrides
        )

        self.decision_engine = (
            MarketDecisionEngine()
        )

        self.signal_generator = (
            SignalGenerator()
        )

        self.signal_validator = (
            SignalValidator(
                min_confidence=60,
                min_risk_reward=2
            )
        )

        self.performance_analyzer = (
            PerformanceAnalyzer()
        )


    @property
    def strategy_name(self):
        if self.strategy is None:
            return "legacy_momentum"

        return getattr(
            self.strategy,
            "name",
            self.strategy.__class__.__name__
        )


    @property
    def cost_model_name(self):
        if self.use_market_costs:
            return "market_specification"

        if (
            self.commission_percent == 0
            and self.slippage_percent == 0
        ):
            return "zero_costs"

        return "legacy_percentage"


    @staticmethod
    def normalize_timestamp(
        value
    ):
        if value is None:
            return None

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
    def validate_data(
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
                "Il DataFrame storico è vuoto."
            )

        normalized = data.copy()

        normalized.columns = [
            str(column).lower().strip()
            for column in normalized.columns
        ]

        required_columns = {
            "open",
            "high",
            "low",
            "close",
            "volume"
        }

        missing_columns = (
            required_columns
            -
            set(normalized.columns)
        )

        if missing_columns:
            raise ValueError(
                "Colonne mancanti: "
                f"{sorted(missing_columns)}"
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

        normalized = normalized.dropna(
            subset=["date"]
        )

        normalized["date"] = (
            normalized["date"]
            .dt
            .tz_convert(None)
        )

        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]

        for column in numeric_columns:
            normalized[column] = pd.to_numeric(
                normalized[column],
                errors="coerce"
            )

        normalized = normalized.dropna(
            subset=[
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume"
            ]
        )

        normalized = normalized.sort_values(
            "date"
        ).reset_index(
            drop=True
        )

        if normalized.empty:
            raise ValueError(
                "Nessuna candela valida."
            )

        return normalized


    @staticmethod
    def prepare_legacy_data(
        data
    ):
        prepared = data.copy()

        prepared = calculate_returns(
            prepared
        )

        prepared = calculate_sma(
            prepared
        )

        prepared = calculate_volatility(
            prepared
        )

        prepared = calculate_ema(
            prepared
        )

        prepared = calculate_rsi(
            prepared
        )

        prepared = calculate_macd(
            prepared
        )

        prepared = calculate_bollinger_bands(
            prepared
        )

        return prepared


    @staticmethod
    def calculate_atr(
        data,
        period=14
    ):
        previous_close = data[
            "close"
        ].shift(1)

        high_low = (
            data["high"]
            -
            data["low"]
        )

        high_previous_close = (
            data["high"]
            -
            previous_close
        ).abs()

        low_previous_close = (
            data["low"]
            -
            previous_close
        ).abs()

        true_range = pd.concat(
            [
                high_low,
                high_previous_close,
                low_previous_close
            ],
            axis=1
        ).max(
            axis=1
        )

        return true_range.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period
        ).mean()


    def create_legacy_decision(
        self,
        historical_slice
    ):
        analysis = analyze_asset(
            historical_slice
        )

        decision = self.decision_engine.decide(
            {
                "score": analysis["score"],
                "risk": analysis["risk"],
                "signals": list(
                    analysis["reasons"]
                )
            }
        )

        decision["strategy"] = (
            "legacy_momentum"
        )

        return decision


    def create_strategy_decision(
        self,
        ticker,
        historical_slice
    ):
        return self.strategy.generate_decision(
            data=historical_slice,
            ticker=ticker
        )


    def create_validated_signal(
        self,
        ticker,
        historical_slice
    ):
        if self.strategy is None:
            decision = (
                self.create_legacy_decision(
                    historical_slice
                )
            )

            latest = historical_slice.iloc[
                -1
            ]

            atr = latest.get(
                "ATR",
                float("nan")
            )

        else:
            decision = (
                self.create_strategy_decision(
                    ticker=ticker,
                    historical_slice=(
                        historical_slice
                    )
                )
            )

            atr = decision.get(
                "atr",
                float("nan")
            )

        action = decision.get(
            "action",
            "WAIT"
        )

        if action not in {
            "BUY",
            "SELL"
        }:
            return None

        if (
            pd.isna(atr)
            or float(atr) <= 0
        ):
            return None

        latest = historical_slice.iloc[
            -1
        ]

        confidence = float(
            decision.get(
                "confidence",
                0
            )
        )

        signal = self.signal_generator.generate(
            ticker=ticker,
            price=float(
                latest["close"]
            ),
            atr=float(atr),
            decision={
                "action": action,
                "confidence": confidence
            }
        )

        if signal is None:
            return None

        signal["strategy"] = decision.get(
            "strategy",
            self.strategy_name
        )

        signal["reasons"] = list(
            decision.get(
                "reasons",
                []
            )
        )

        validation = (
            self.signal_validator.validate(
                signal
            )
        )

        if not validation["approved"]:
            return None

        return validation["signal"]


    @staticmethod
    def determine_exit(
        position,
        candle
    ):
        high = float(
            candle["high"]
        )

        low = float(
            candle["low"]
        )

        direction = position[
            "direction"
        ]

        stop_loss = float(
            position["stop_loss"]
        )

        take_profit = float(
            position["take_profit"]
        )

        if direction == "LONG":
            if low <= stop_loss:
                return {
                    "price": stop_loss,
                    "reason": "STOP_LOSS"
                }

            if high >= take_profit:
                return {
                    "price": take_profit,
                    "reason": "TAKE_PROFIT"
                }

        elif direction == "SHORT":
            if high >= stop_loss:
                return {
                    "price": stop_loss,
                    "reason": "STOP_LOSS"
                }

            if low <= take_profit:
                return {
                    "price": take_profit,
                    "reason": "TAKE_PROFIT"
                }

        return None


    def prepare_backtest_data(
        self,
        clean_data
    ):
        if self.strategy is None:
            prepared_data = (
                self.prepare_legacy_data(
                    clean_data
                )
            )

            prepared_data["ATR"] = (
                self.calculate_atr(
                    prepared_data
                )
            )

            prepared_data = (
                prepared_data.dropna(
                    subset=[
                        "SMA_20",
                        "Volatility",
                        "RSI",
                        "MACD",
                        "MACD_signal",
                        "BB_upper",
                        "BB_lower",
                        "ATR"
                    ]
                )
                .reset_index(
                    drop=True
                )
            )

            return prepared_data

        prepared_data = clean_data.copy()

        prepared_data["ATR"] = (
            self.calculate_atr(
                prepared_data
            )
        )

        prepared_data = prepared_data.dropna(
            subset=["ATR"]
        ).reset_index(
            drop=True
        )

        return prepared_data


    def run(
        self,
        ticker,
        data,
        trade_start_time=None
    ):
        """
        Esegue il backtest.

        trade_start_time permette di usare le
        candele precedenti soltanto come warm-up.
        """

        normalized_ticker = (
            str(ticker)
            .upper()
            .strip()
        )

        specification = (
            MarketSpecificationRegistry.get(
                normalized_ticker
            )
        )

        clean_data = self.validate_data(
            data
        )

        prepared_data = (
            self.prepare_backtest_data(
                clean_data
            )
        )

        if len(prepared_data) < (
            self.minimum_history
        ):
            raise ValueError(
                "Storico insufficiente."
            )

        normalized_trade_start = (
            self.normalize_timestamp(
                trade_start_time
            )
            if trade_start_time is not None
            else None
        )

        portfolio = BacktestPortfolio(
            initial_capital=(
                self.initial_capital
            ),
            max_position_percent=(
                self.max_position_percent
            ),
            commission_percent=(
                self.commission_percent
            ),
            slippage_percent=(
                self.slippage_percent
            ),
            use_market_costs=(
                self.use_market_costs
            ),
            margin_overrides=(
                self.margin_overrides
            )
        )

        cooldown_remaining = 0
        position_open_index = None
        current_day = None
        trades_today = 0

        for index in range(
            self.minimum_history,
            len(prepared_data)
        ):
            current_candle = (
                prepared_data.iloc[index]
            )

            timestamp = self.normalize_timestamp(
                current_candle["date"]
            )

            candle_day = timestamp.date()

            if candle_day != current_day:
                current_day = candle_day
                trades_today = 0

            if cooldown_remaining > 0:
                cooldown_remaining -= 1

            if portfolio.open_position is not None:
                exit_data = self.determine_exit(
                    portfolio.open_position,
                    current_candle
                )

                holding_bars = (
                    index
                    -
                    position_open_index
                )

                if (
                    exit_data is None
                    and holding_bars >=
                    self.max_holding_bars
                ):
                    exit_data = {
                        "price": float(
                            current_candle[
                                "close"
                            ]
                        ),
                        "reason": (
                            "MAX_HOLDING_TIME"
                        )
                    }

                if exit_data is not None:
                    portfolio.close_trade(
                        exit_price=(
                            exit_data["price"]
                        ),
                        timestamp=timestamp,
                        reason=(
                            exit_data["reason"]
                        )
                    )

                    position_open_index = None

                    cooldown_remaining = (
                        self.cooldown_bars
                    )

            trading_enabled = (
                normalized_trade_start is None
                or timestamp >=
                normalized_trade_start
            )

            can_open_trade = (
                trading_enabled
                and portfolio.open_position is None
                and cooldown_remaining == 0
                and trades_today
                <
                self.max_trades_per_day
            )

            if can_open_trade:
                historical_slice = (
                    prepared_data.iloc[
                        :index + 1
                    ].copy()
                )

                signal = (
                    self.create_validated_signal(
                        ticker=normalized_ticker,
                        historical_slice=(
                            historical_slice
                        )
                    )
                )

                if signal is not None:
                    position = (
                        portfolio.open_trade(
                            signal=signal,
                            timestamp=timestamp
                        )
                    )

                    if position is not None:
                        position_open_index = (
                            index
                        )

                        trades_today += 1

            current_price = float(
                current_candle["close"]
            )

            current_equity = (
                portfolio.calculate_equity(
                    current_price=current_price
                )
            )

            portfolio.record_equity(
                timestamp=timestamp,
                equity=current_equity
            )

        if portfolio.open_position is not None:
            final_candle = (
                prepared_data.iloc[-1]
            )

            portfolio.close_trade(
                exit_price=float(
                    final_candle["close"]
                ),
                timestamp=self.normalize_timestamp(
                    final_candle["date"]
                ),
                reason="END_OF_DATA"
            )

        final_capital = float(
            portfolio.cash
        )

        metrics = (
            self.performance_analyzer.calculate(
                initial_capital=(
                    portfolio.initial_capital
                ),
                final_capital=final_capital,
                trades=(
                    portfolio.closed_trades
                ),
                equity_curve=(
                    portfolio.equity_curve
                )
            )
        )

        metrics["total_commissions"] = float(
            portfolio.total_commissions
        )

        metrics["cost_model"] = (
            self.cost_model_name
        )

        return {
            "ticker": normalized_ticker,
            "asset_type": (
                specification.asset_class
            ),
            "strategy": self.strategy_name,
            "cost_model": (
                self.cost_model_name
            ),
            "metrics": metrics,
            "trades": (
                portfolio.closed_trades
            ),
            "equity_curve": (
                portfolio.equity_curve
            ),
            "market_specification": {
                "code": specification.code,
                "asset_class": (
                    specification.asset_class
                ),
                "contract_multiplier": (
                    specification.contract_multiplier
                ),
                "quantity_step": (
                    specification.quantity_step
                ),
                "minimum_quantity": (
                    specification.minimum_quantity
                ),
                "tick_size": (
                    specification.tick_size
                ),
                "tick_value": (
                    specification.tick_value
                ),
                "margin_model": (
                    specification.margin_model
                )
            }
        }