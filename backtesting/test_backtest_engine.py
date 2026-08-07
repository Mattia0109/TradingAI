import pandas as pd

from backtesting.backtester import Backtester
from strategies.trend_following import (
    TrendFollowingStrategy
)


def create_historical_data(
    rows=240
):
    dates = pd.date_range(
        start="2025-01-01",
        periods=rows,
        freq="h"
    )

    close_prices = []

    price = 100.0

    for index in range(rows):
        if index % 30 < 20:
            price += 1.0
        else:
            price -= 0.6

        close_prices.append(
            round(
                price,
                2
            )
        )

    return pd.DataFrame(
        {
            "date": dates,
            "open": [
                value - 0.3
                for value in close_prices
            ],
            "high": [
                value + 2.0
                for value in close_prices
            ],
            "low": [
                value - 2.0
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
            "atr": 2.0,
            "reasons": [
                "Segnale test"
            ]
        }


def test_backtester_returns_report():

    data = create_historical_data()

    backtester = Backtester(
        initial_capital=10000,
        max_position_percent=0.10,
        minimum_history=30
    )

    result = backtester.run(
        ticker="TEST",
        data=data
    )

    assert result["ticker"] == "TEST"
    assert result["strategy"] == (
        "legacy_momentum"
    )

    assert "metrics" in result
    assert "trades" in result
    assert "equity_curve" in result

    assert (
        result["metrics"][
            "initial_capital"
        ]
        == 10000
    )

    assert (
        result["metrics"][
            "final_capital"
        ]
        > 0
    )


def test_custom_strategy_is_used():

    data = create_historical_data()

    backtester = Backtester(
        initial_capital=10000,
        max_position_percent=0.10,
        minimum_history=30,
        commission_percent=0,
        slippage_percent=0,
        max_holding_bars=5,
        cooldown_bars=1,
        max_trades_per_day=3,
        strategy=AlwaysBuyStrategy()
    )

    result = backtester.run(
        ticker="TEST",
        data=data
    )

    assert result["strategy"] == (
        "always_buy_test"
    )

    assert (
        result["metrics"][
            "total_trades"
        ]
        > 0
    )

    for trade in result["trades"]:
        assert trade["strategy"] == (
            "always_buy_test"
        )


def test_trend_following_strategy_integrates():

    data = create_historical_data(
        rows=300
    )

    strategy = TrendFollowingStrategy(
        minimum_adx=10,
        minimum_ema_distance_percent=0.0001,
        minimum_slope_percent=0.00001
    )

    backtester = Backtester(
        initial_capital=10000,
        minimum_history=60,
        commission_percent=0,
        slippage_percent=0,
        strategy=strategy
    )

    result = backtester.run(
        ticker="TEST",
        data=data
    )

    assert result["strategy"] == (
        "trend_following"
    )

    assert "metrics" in result
    assert "trades" in result


def test_trade_start_time_blocks_warmup_trades():

    data = create_historical_data(
        rows=240
    )

    trade_start_time = data.iloc[
        180
    ]["date"]

    backtester = Backtester(
        initial_capital=10000,
        minimum_history=30,
        commission_percent=0,
        slippage_percent=0,
        max_holding_bars=5,
        cooldown_bars=1,
        strategy=AlwaysBuyStrategy()
    )

    result = backtester.run(
        ticker="TEST",
        data=data,
        trade_start_time=trade_start_time
    )

    assert (
        result["metrics"]["total_trades"]
        > 0
    )

    for trade in result["trades"]:
        assert (
            pd.Timestamp(
                trade["open_time"]
            )
            >= pd.Timestamp(
                trade_start_time
            )
        )


def test_position_limit_is_ten_percent():

    backtester = Backtester(
        initial_capital=10000,
        max_position_percent=0.10
    )

    assert (
        backtester.max_position_percent
        == 0.10
    )


def test_position_above_ten_percent_rejected():

    raised_error = False

    try:
        Backtester(
            initial_capital=10000,
            max_position_percent=0.15
        )

    except ValueError:
        raised_error = True

    assert raised_error is True


def test_invalid_strategy_rejected():

    raised_error = False

    try:
        Backtester(
            strategy=object()
        )

    except TypeError:
        raised_error = True

    assert raised_error is True


def test_invalid_data_rejected():

    backtester = Backtester()

    invalid_data = pd.DataFrame(
        {
            "close": [
                100,
                101
            ]
        }
    )

    raised_error = False

    try:
        backtester.run(
            ticker="TEST",
            data=invalid_data
        )

    except ValueError:
        raised_error = True

    assert raised_error is True