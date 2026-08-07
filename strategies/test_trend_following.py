import math

import pandas as pd

from strategies.trend_following import (
    TrendFollowingStrategy
)


def create_trending_data(
    direction="UP",
    rows=220
):
    dates = pd.date_range(
        start="2026-01-01",
        periods=rows,
        freq="5min"
    )

    close_prices = []

    base_price = 100.0

    for index in range(rows):
        wave = math.sin(
            index / 4
        ) * 0.18

        if direction == "UP":
            trend = index * 0.10

        elif direction == "DOWN":
            trend = -index * 0.10

        else:
            trend = 0.0

        close_price = (
            base_price +
            trend +
            wave
        )

        close_prices.append(
            round(
                close_price,
                4
            )
        )

    return pd.DataFrame(
        {
            "date": dates,
            "open": [
                value - 0.08
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


def test_uptrend_generates_buy():

    strategy = TrendFollowingStrategy(
        minimum_adx=15,
        minimum_ema_distance_percent=0.0001,
        minimum_slope_percent=0.00001
    )

    data = create_trending_data(
        direction="UP"
    )

    result = strategy.generate_decision(
        data=data,
        ticker="TEST"
    )

    assert result["action"] == "BUY"
    assert result["strategy"] == (
        "trend_following"
    )

    assert result["confidence"] >= 50
    assert result["atr"] > 0


def test_downtrend_generates_sell():

    strategy = TrendFollowingStrategy(
        minimum_adx=15,
        minimum_ema_distance_percent=0.0001,
        minimum_slope_percent=0.00001
    )

    data = create_trending_data(
        direction="DOWN"
    )

    result = strategy.generate_decision(
        data=data,
        ticker="TEST"
    )

    assert result["action"] == "SELL"
    assert result["confidence"] >= 50
    assert result["atr"] > 0


def test_flat_market_generates_wait():

    strategy = TrendFollowingStrategy(
        minimum_adx=20,
        minimum_ema_distance_percent=0.001,
        minimum_slope_percent=0.0005
    )

    data = create_trending_data(
        direction="FLAT"
    )

    result = strategy.generate_decision(
        data=data,
        ticker="TEST"
    )

    assert result["action"] == "WAIT"


def test_insufficient_history_waits():

    strategy = TrendFollowingStrategy()

    data = create_trending_data(
        direction="UP",
        rows=20
    )

    result = strategy.generate_decision(
        data=data,
        ticker="TEST"
    )

    assert result["action"] == "WAIT"


def test_invalid_parameters_rejected():

    raised_error = False

    try:
        TrendFollowingStrategy(
            fast_ema_period=50,
            slow_ema_period=20
        )

    except ValueError:
        raised_error = True

    assert raised_error is True