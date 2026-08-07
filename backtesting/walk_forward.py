import copy

import pandas as pd

from backtesting.backtester import Backtester
from backtesting.performance import PerformanceAnalyzer


class WalkForwardValidator:
    """
    Validazione fuori campione tramite
    finestre temporali consecutive.

    La finestra TRAIN viene usata soltanto
    per lo storico e gli indicatori.

    Le operazioni iniziano esclusivamente
    nella finestra TEST.
    """

    def __init__(
        self,
        train_bars=1000,
        test_bars=500,
        step_bars=None,
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
        if train_bars <= 0:
            raise ValueError(
                "train_bars deve essere maggiore di zero."
            )

        if test_bars <= 0:
            raise ValueError(
                "test_bars deve essere maggiore di zero."
            )

        if step_bars is None:
            step_bars = test_bars

        if step_bars <= 0:
            raise ValueError(
                "step_bars deve essere maggiore di zero."
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

        self.train_bars = int(
            train_bars
        )

        self.test_bars = int(
            test_bars
        )

        self.step_bars = int(
            step_bars
        )

        self.initial_capital = float(
            initial_capital
        )

        self.strategy = strategy

        self.use_market_costs = bool(
            use_market_costs
        )

        self.margin_overrides = dict(
            margin_overrides
        )

        self.backtester_settings = {
            "initial_capital": (
                initial_capital
            ),
            "max_position_percent": (
                max_position_percent
            ),
            "minimum_history": (
                minimum_history
            ),
            "commission_percent": (
                commission_percent
            ),
            "slippage_percent": (
                slippage_percent
            ),
            "max_holding_bars": (
                max_holding_bars
            ),
            "cooldown_bars": (
                cooldown_bars
            ),
            "max_trades_per_day": (
                max_trades_per_day
            ),
            "use_market_costs": (
                self.use_market_costs
            ),
            "margin_overrides": (
                self.margin_overrides
            )
        }

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

        commission = float(
            self.backtester_settings[
                "commission_percent"
            ]
        )

        slippage = float(
            self.backtester_settings[
                "slippage_percent"
            ]
        )

        if (
            commission == 0
            and slippage == 0
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
            return None

        return timestamp.tz_convert(
            None
        )


    @classmethod
    def validate_data(
        cls,
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

        normalized = normalized.sort_values(
            "date"
        ).reset_index(
            drop=True
        )

        if normalized.empty:
            raise ValueError(
                "Nessun timestamp valido."
            )

        return normalized


    def create_windows(
        self,
        data
    ):
        windows = []

        total_bars = len(data)
        start_index = 0
        fold_number = 1

        while True:
            train_start = start_index

            train_end = (
                train_start
                +
                self.train_bars
            )

            test_start = train_end

            test_end = (
                test_start
                +
                self.test_bars
            )

            if test_end > total_bars:
                break

            windows.append(
                {
                    "fold": fold_number,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end
                }
            )

            fold_number += 1
            start_index += self.step_bars

        return windows


    @classmethod
    def filter_test_trades(
        cls,
        trades,
        test_start_time,
        test_end_time
    ):
        normalized_start = (
            cls.normalize_timestamp(
                test_start_time
            )
        )

        normalized_end = (
            cls.normalize_timestamp(
                test_end_time
            )
        )

        if (
            normalized_start is None
            or normalized_end is None
        ):
            return []

        filtered = []

        for trade in trades:
            open_time = cls.normalize_timestamp(
                trade.get("open_time")
            )

            if open_time is None:
                continue

            if (
                normalized_start
                <= open_time
                <= normalized_end
            ):
                normalized_trade = dict(
                    trade
                )

                normalized_trade[
                    "open_time"
                ] = open_time

                normalized_trade[
                    "close_time"
                ] = cls.normalize_timestamp(
                    trade.get("close_time")
                )

                filtered.append(
                    normalized_trade
                )

        return filtered


    def build_trade_equity_curve(
        self,
        trades,
        test_start_time,
        test_end_time
    ):
        equity = self.initial_capital

        normalized_start = (
            self.normalize_timestamp(
                test_start_time
            )
        )

        normalized_end = (
            self.normalize_timestamp(
                test_end_time
            )
        )

        curve = [
            {
                "timestamp": normalized_start,
                "equity": equity
            }
        ]

        for trade in trades:
            equity += float(
                trade["pnl"]
            )

            curve.append(
                {
                    "timestamp": (
                        self.normalize_timestamp(
                            trade.get(
                                "close_time"
                            )
                        )
                    ),
                    "equity": equity
                }
            )

        curve.append(
            {
                "timestamp": normalized_end,
                "equity": equity
            }
        )

        return curve


    def evaluate_fold(
        self,
        ticker,
        fold_data,
        test_start_time,
        test_end_time,
        fold_number
    ):
        fold_strategy = (
            copy.deepcopy(
                self.strategy
            )
            if self.strategy is not None
            else None
        )

        backtester = Backtester(
            strategy=fold_strategy,
            **self.backtester_settings
        )

        normalized_test_start = (
            self.normalize_timestamp(
                test_start_time
            )
        )

        normalized_test_end = (
            self.normalize_timestamp(
                test_end_time
            )
        )

        result = backtester.run(
            ticker=ticker,
            data=fold_data,
            trade_start_time=(
                normalized_test_start
            )
        )

        test_trades = self.filter_test_trades(
            trades=result["trades"],
            test_start_time=(
                normalized_test_start
            ),
            test_end_time=(
                normalized_test_end
            )
        )

        test_net_profit = sum(
            float(trade["pnl"])
            for trade in test_trades
        )

        final_capital = (
            self.initial_capital
            +
            test_net_profit
        )

        test_equity_curve = (
            self.build_trade_equity_curve(
                trades=test_trades,
                test_start_time=(
                    normalized_test_start
                ),
                test_end_time=(
                    normalized_test_end
                )
            )
        )

        metrics = (
            self.performance_analyzer.calculate(
                initial_capital=(
                    self.initial_capital
                ),
                final_capital=final_capital,
                trades=test_trades,
                equity_curve=(
                    test_equity_curve
                )
            )
        )

        metrics["total_commissions"] = sum(
            float(
                trade.get(
                    "entry_commission",
                    0
                )
            )
            +
            float(
                trade.get(
                    "exit_commission",
                    0
                )
            )
            for trade in test_trades
        )

        metrics["cost_model"] = (
            self.cost_model_name
        )

        return {
            "fold": fold_number,
            "strategy": self.strategy_name,
            "cost_model": (
                self.cost_model_name
            ),
            "test_start": (
                normalized_test_start
            ),
            "test_end": (
                normalized_test_end
            ),
            "metrics": metrics,
            "trades": test_trades
        }


    def calculate_summary(
        self,
        folds
    ):
        if not folds:
            return {
                "total_folds": 0,
                "profitable_folds": 0,
                "losing_folds": 0,
                "profitable_fold_percent": 0.0,
                "total_trades": 0,
                "total_net_profit": 0.0,
                "total_commissions": 0.0,
                "average_fold_return_percent": 0.0,
                "median_fold_return_percent": 0.0,
                "worst_fold_return_percent": 0.0,
                "best_fold_return_percent": 0.0,
                "walk_forward_efficiency": 0.0,
                "is_robust": False
            }

        returns = [
            fold["metrics"][
                "total_return_percent"
            ]
            for fold in folds
        ]

        net_profits = [
            fold["metrics"][
                "net_profit"
            ]
            for fold in folds
        ]

        total_commissions = sum(
            float(
                fold["metrics"].get(
                    "total_commissions",
                    0
                )
            )
            for fold in folds
        )

        profitable_folds = sum(
            1
            for value in returns
            if value > 0
        )

        losing_folds = sum(
            1
            for value in returns
            if value < 0
        )

        total_trades = sum(
            fold["metrics"][
                "total_trades"
            ]
            for fold in folds
        )

        total_net_profit = sum(
            net_profits
        )

        average_return = (
            sum(returns)
            /
            len(returns)
        )

        sorted_returns = sorted(
            returns
        )

        middle = (
            len(sorted_returns)
            //
            2
        )

        if len(sorted_returns) % 2 == 0:
            median_return = (
                sorted_returns[
                    middle - 1
                ]
                +
                sorted_returns[
                    middle
                ]
            ) / 2

        else:
            median_return = (
                sorted_returns[middle]
            )

        profitable_fold_percent = (
            profitable_folds
            /
            len(folds)
            *
            100
        )

        positive_profit = sum(
            value
            for value in net_profits
            if value > 0
        )

        negative_profit = abs(
            sum(
                value
                for value in net_profits
                if value < 0
            )
        )

        if negative_profit > 0:
            walk_forward_efficiency = (
                positive_profit
                /
                negative_profit
            )

        elif positive_profit > 0:
            walk_forward_efficiency = (
                float("inf")
            )

        else:
            walk_forward_efficiency = 0.0

        is_robust = (
            len(folds) >= 3
            and profitable_fold_percent >= 60
            and total_net_profit > 0
            and total_trades >= 30
            and walk_forward_efficiency > 1
        )

        return {
            "total_folds": len(folds),
            "profitable_folds": (
                profitable_folds
            ),
            "losing_folds": (
                losing_folds
            ),
            "profitable_fold_percent": (
                profitable_fold_percent
            ),
            "total_trades": total_trades,
            "total_net_profit": (
                total_net_profit
            ),
            "total_commissions": (
                total_commissions
            ),
            "average_fold_return_percent": (
                average_return
            ),
            "median_fold_return_percent": (
                median_return
            ),
            "worst_fold_return_percent": min(
                returns
            ),
            "best_fold_return_percent": max(
                returns
            ),
            "walk_forward_efficiency": (
                walk_forward_efficiency
            ),
            "is_robust": is_robust
        }


    def run(
        self,
        ticker,
        data
    ):
        normalized = self.validate_data(
            data
        )

        windows = self.create_windows(
            normalized
        )

        if not windows:
            raise ValueError(
                "Storico insufficiente per creare "
                "finestre walk-forward."
            )

        folds = []

        for window in windows:
            fold_data = normalized.iloc[
                window["train_start"]:
                window["test_end"]
            ].copy()

            test_start_time = normalized.iloc[
                window["test_start"]
            ]["date"]

            test_end_time = normalized.iloc[
                window["test_end"] - 1
            ]["date"]

            fold_result = self.evaluate_fold(
                ticker=ticker,
                fold_data=fold_data,
                test_start_time=test_start_time,
                test_end_time=test_end_time,
                fold_number=window["fold"]
            )

            folds.append(
                fold_result
            )

        summary = self.calculate_summary(
            folds
        )

        return {
            "ticker": ticker,
            "strategy": self.strategy_name,
            "cost_model": (
                self.cost_model_name
            ),
            "settings": {
                "train_bars": self.train_bars,
                "test_bars": self.test_bars,
                "step_bars": self.step_bars,
                "use_market_costs": (
                    self.use_market_costs
                ),
                "margin_overrides": dict(
                    self.margin_overrides
                )
            },
            "folds": folds,
            "summary": summary
        }