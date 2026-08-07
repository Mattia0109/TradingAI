import pandas as pd

from backtesting.walk_forward import (
    WalkForwardValidator
)


def create_walk_forward_data(
    rows=500
):
    dates = pd.date_range(
        start="2025-01-01",
        periods=rows,
        freq="h"
    )

    close_prices = []

    price = 100.0

    for index in range(rows):
        cycle = index % 100

        if cycle < 60:
            price += 0.25
        else:
            price -= 0.15

        close_prices.append(
            round(
                price,
                4
            )
        )

    return pd.DataFrame(
        {
            "date": dates,
            "open": [
                value - 0.05
                for value in close_prices
            ],
            "high": [
                value + 0.30
                for value in close_prices
            ],
            "low": [
                value - 0.30
                for value in close_prices
            ],
            "close": close_prices,
            "volume": [
                1_000_000
            ] * rows
        }
    )


class AlwaysBuyStrategy:
    name = "always_buy_test"

    def generate_decision(
        self,
        data,
        ticker
    ):
        return {
            "ticker": ticker,
            "strategy": self.name,
            "action": "BUY",
            "confidence": 80,
            "atr": 1.0,
            "reasons": [
                "Segnale deterministico"
            ]
        }


def test_create_windows():

    validator = WalkForwardValidator(
        train_bars=200,
        test_bars=100,
        step_bars=100
    )

    data = create_walk_forward_data(
        rows=500
    )

    windows = validator.create_windows(
        data
    )

    assert len(windows) == 3

    assert windows[0][
        "train_start"
    ] == 0

    assert windows[0][
        "train_end"
    ] == 200

    assert windows[0][
        "test_start"
    ] == 200

    assert windows[0][
        "test_end"
    ] == 300


def test_walk_forward_returns_report():

    validator = WalkForwardValidator(
        train_bars=200,
        test_bars=100,
        step_bars=100,
        initial_capital=10000,
        minimum_history=30,
        commission_percent=0,
        slippage_percent=0,
        max_holding_bars=24,
        cooldown_bars=2,
        max_trades_per_day=3
    )

    data = create_walk_forward_data(
        rows=500
    )

    result = validator.run(
        ticker="TEST",
        data=data
    )

    assert result["ticker"] == "TEST"

    assert result["strategy"] == (
        "legacy_momentum"
    )

    assert len(
        result["folds"]
    ) == 3

    assert result["summary"][
        "total_folds"
    ] == 3

    assert (
        result["summary"][
            "total_trades"
        ]
        >= 0
    )

    assert "is_robust" in result[
        "summary"
    ]


def test_custom_strategy_starts_only_in_test():

    validator = WalkForwardValidator(
        train_bars=200,
        test_bars=100,
        step_bars=100,
        initial_capital=10000,
        minimum_history=30,
        commission_percent=0,
        slippage_percent=0,
        max_holding_bars=12,
        cooldown_bars=1,
        max_trades_per_day=3,
        strategy=AlwaysBuyStrategy()
    )

    data = create_walk_forward_data(
        rows=500
    )

    result = validator.run(
        ticker="TEST",
        data=data
    )

    assert result["strategy"] == (
        "always_buy_test"
    )

    assert (
        result["summary"]["total_trades"]
        > 0
    )

    for fold in result["folds"]:
        assert fold["strategy"] == (
            "always_buy_test"
        )

        for trade in fold["trades"]:
            assert (
                pd.Timestamp(
                    trade["open_time"]
                )
                >= pd.Timestamp(
                    fold["test_start"]
                )
            )


def test_insufficient_data_rejected():

    validator = WalkForwardValidator(
        train_bars=200,
        test_bars=100
    )

    data = create_walk_forward_data(
        rows=250
    )

    raised_error = False

    try:
        validator.run(
            ticker="TEST",
            data=data
        )

    except ValueError:
        raised_error = True

    assert raised_error is True


def test_robust_summary_conditions():

    validator = WalkForwardValidator()

    folds = [
        {
            "metrics": {
                "total_return_percent": 1.0,
                "net_profit": 100,
                "total_trades": 12
            }
        },
        {
            "metrics": {
                "total_return_percent": 0.5,
                "net_profit": 50,
                "total_trades": 12
            }
        },
        {
            "metrics": {
                "total_return_percent": 0.8,
                "net_profit": 80,
                "total_trades": 12
            }
        }
    ]

    summary = validator.calculate_summary(
        folds
    )

    assert summary[
        "profitable_fold_percent"
    ] == 100

    assert summary[
        "total_trades"
    ] == 36

    assert summary[
        "is_robust"
    ] is True