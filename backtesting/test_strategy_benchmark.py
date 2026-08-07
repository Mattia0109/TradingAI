import pandas as pd

from backtesting.strategy_benchmark import (
    StrategyBenchmarkRunner
)


class NamedStrategy:
    def __init__(
        self,
        name
    ):
        self.name = name

    def generate_decision(
        self,
        data,
        ticker
    ):
        return {
            "ticker": ticker,
            "strategy": self.name,
            "action": "WAIT",
            "confidence": 0,
            "atr": 1,
            "reasons": []
        }


class FakeDataLoader:
    def __init__(
        self,
        rows=500
    ):
        self.rows = rows
        self.calls = 0

    def __call__(
        self,
        ticker,
        period,
        interval
    ):
        self.calls += 1

        dates = pd.date_range(
            start="2026-01-01",
            periods=self.rows,
            freq="5min"
        )

        return pd.DataFrame(
            {
                "date": dates,
                "open": [100.0] * self.rows,
                "high": [101.0] * self.rows,
                "low": [99.0] * self.rows,
                "close": [100.0] * self.rows,
                "volume": [1000] * self.rows
            }
        )


class FakeWalkForwardValidator:
    created_settings = []

    def __init__(
        self,
        strategy=None,
        use_market_costs=False,
        margin_overrides=None,
        **kwargs
    ):
        self.strategy = strategy

        self.use_market_costs = bool(
            use_market_costs
        )

        self.margin_overrides = dict(
            margin_overrides or {}
        )

        self.__class__.created_settings.append(
            {
                "use_market_costs": (
                    self.use_market_costs
                ),
                "margin_overrides": dict(
                    self.margin_overrides
                )
            }
        )


    def run(
        self,
        ticker,
        data
    ):
        strategy_name = (
            "legacy"
            if self.strategy is None
            else self.strategy.name
        )

        if strategy_name == "good":
            net_profit = (
                60.0
                if self.use_market_costs
                else 100.0
            )

            summary = {
                "total_folds": 5,
                "profitable_folds": 4,
                "losing_folds": 1,
                "profitable_fold_percent": 80.0,
                "total_trades": 50,
                "total_net_profit": net_profit,
                "total_commissions": (
                    20.0
                    if self.use_market_costs
                    else 0.0
                ),
                "average_fold_return_percent": 0.2,
                "median_fold_return_percent": 0.15,
                "worst_fold_return_percent": -0.1,
                "best_fold_return_percent": 0.5,
                "walk_forward_efficiency": 2.0,
                "is_robust": True
            }

        else:
            summary = {
                "total_folds": 5,
                "profitable_folds": 1,
                "losing_folds": 4,
                "profitable_fold_percent": 20.0,
                "total_trades": 50,
                "total_net_profit": -100.0,
                "total_commissions": (
                    20.0
                    if self.use_market_costs
                    else 0.0
                ),
                "average_fold_return_percent": -0.2,
                "median_fold_return_percent": -0.15,
                "worst_fold_return_percent": -0.5,
                "best_fold_return_percent": 0.1,
                "walk_forward_efficiency": 0.5,
                "is_robust": False
            }

        return {
            "ticker": ticker,
            "summary": summary,
            "folds": []
        }


def create_market_case(
    ticker="AAPL"
):
    return {
        "ticker": ticker,
        "period": "60d",
        "interval": "5m",
        "train_bars": 200,
        "test_bars": 100,
        "step_bars": 100
    }


def test_data_downloaded_once_per_case():

    loader = FakeDataLoader()

    runner = StrategyBenchmarkRunner(
        strategies={
            "good": NamedStrategy(
                "good"
            ),
            "bad": NamedStrategy(
                "bad"
            )
        },
        data_loader=loader,
        validator_class=(
            FakeWalkForwardValidator
        )
    )

    runner.run(
        market_cases=[
            create_market_case()
        ]
    )

    assert loader.calls == 1


def test_good_strategy_advances_to_cost_test():

    runner = StrategyBenchmarkRunner(
        strategies={
            "good": NamedStrategy(
                "good"
            )
        },
        data_loader=FakeDataLoader(),
        cost_mode="gross",
        validator_class=(
            FakeWalkForwardValidator
        )
    )

    result = runner.run(
        market_cases=[
            create_market_case()
        ]
    )

    candidate = result[
        "candidates"
    ][0]

    assert candidate[
        "classification"
    ] == "GROSS_EDGE_CANDIDATE"

    assert candidate[
        "advance_to_cost_test"
    ] is True

    assert candidate[
        "advance_to_robustness_test"
    ] is False


def test_market_mode_detects_net_candidate():

    FakeWalkForwardValidator.created_settings.clear()

    runner = StrategyBenchmarkRunner(
        strategies={
            "good": NamedStrategy(
                "good"
            )
        },
        data_loader=FakeDataLoader(),
        cost_mode="market",
        validator_class=(
            FakeWalkForwardValidator
        )
    )

    result = runner.run(
        market_cases=[
            create_market_case()
        ]
    )

    candidate = result[
        "candidates"
    ][0]

    assert candidate[
        "classification"
    ] == "NET_EDGE_CANDIDATE"

    assert candidate[
        "advance_to_cost_test"
    ] is False

    assert candidate[
        "advance_to_robustness_test"
    ] is True

    assert candidate[
        "total_commissions"
    ] == 20

    assert (
        FakeWalkForwardValidator
        .created_settings[0][
            "use_market_costs"
        ]
        is True
    )


def test_bad_market_strategy_is_rejected():

    runner = StrategyBenchmarkRunner(
        strategies={
            "bad": NamedStrategy(
                "bad"
            )
        },
        data_loader=FakeDataLoader(),
        cost_mode="market",
        validator_class=(
            FakeWalkForwardValidator
        )
    )

    result = runner.run(
        market_cases=[
            create_market_case()
        ]
    )

    test_result = result[
        "results"
    ][0]

    assert test_result[
        "classification"
    ] == "REJECT_NO_NET_EDGE"

    assert result[
        "candidates"
    ] == []


def test_insufficient_data_is_skipped():

    runner = StrategyBenchmarkRunner(
        strategies={
            "good": NamedStrategy(
                "good"
            )
        },
        data_loader=FakeDataLoader(
            rows=250
        ),
        validator_class=(
            FakeWalkForwardValidator
        )
    )

    result = runner.run(
        market_cases=[
            create_market_case()
        ]
    )

    assert result[
        "results"
    ][0][
        "status"
    ] == "SKIPPED"

    assert result[
        "results"
    ][0][
        "classification"
    ] == "SKIPPED"


def test_future_without_margin_is_skipped():

    loader = FakeDataLoader()

    runner = StrategyBenchmarkRunner(
        strategies={
            "good": NamedStrategy(
                "good"
            )
        },
        data_loader=loader,
        validator_class=(
            FakeWalkForwardValidator
        )
    )

    result = runner.run(
        market_cases=[
            create_market_case(
                ticker="CL=F"
            )
        ]
    )

    assert loader.calls == 0

    assert result[
        "results"
    ][0][
        "status"
    ] == "SKIPPED"

    assert (
        "Margine esterno"
        in result[
            "results"
        ][0][
            "error"
        ]
    )


def test_future_with_margin_is_executed():

    FakeWalkForwardValidator.created_settings.clear()

    loader = FakeDataLoader()

    runner = StrategyBenchmarkRunner(
        strategies={
            "good": NamedStrategy(
                "good"
            )
        },
        data_loader=loader,
        margin_overrides={
            "CL=F": 5000
        },
        validator_class=(
            FakeWalkForwardValidator
        )
    )

    result = runner.run(
        market_cases=[
            create_market_case(
                ticker="CL=F"
            )
        ]
    )

    assert loader.calls == 1

    assert result[
        "results"
    ][0][
        "status"
    ] == "COMPLETED"

    assert (
        FakeWalkForwardValidator
        .created_settings[0][
            "margin_overrides"
        ][
            "CL=F"
        ]
        == 5000
    )


def test_invalid_cost_mode_is_rejected():

    raised_error = False

    try:
        StrategyBenchmarkRunner(
            strategies={
                "good": NamedStrategy(
                    "good"
                )
            },
            data_loader=FakeDataLoader(),
            cost_mode="invalid"
        )

    except ValueError:
        raised_error = True

    assert raised_error is True