import numpy as np
import pandas as pd
import pytest

from strategies.multi_asset_tsmom import (
    MultiAssetTimeSeriesMomentumStrategy
)


def create_trending_data(
    direction=1,
    rows=340
):
    dates = pd.date_range(
        start="2020-01-01",
        periods=rows,
        freq="1D"
    )

    index = np.arange(
        rows
    )

    logarithmic_trend = (
        direction
        *
        0.0015
        *
        index
    )

    oscillation = (
        0.01
        *
        np.sin(
            index / 7
        )
    )

    close_prices = (
        100
        *
        np.exp(
            logarithmic_trend
            +
            oscillation
        )
    )

    return pd.DataFrame(
        {
            "date": dates,
            "close": close_prices
        }
    )


def test_uptrend_generates_long_signal():

    strategy = (
        MultiAssetTimeSeriesMomentumStrategy()
    )

    result = strategy.generate_signal(
        data=create_trending_data(
            direction=1
        ),
        ticker="SPY",
        asset_class="EQUITY"
    )

    assert result["action"] == "LONG"
    assert result["signal"] > 0
    assert result["strength"] > 0
    assert result["history_ready"] is True


def test_downtrend_generates_short_signal():

    strategy = (
        MultiAssetTimeSeriesMomentumStrategy()
    )

    result = strategy.generate_signal(
        data=create_trending_data(
            direction=-1
        ),
        ticker="SPY",
        asset_class="EQUITY"
    )

    assert result["action"] == "SHORT"
    assert result["signal"] < 0
    assert result["strength"] > 0


def test_flat_prices_generate_flat_signal():

    dates = pd.date_range(
        start="2020-01-01",
        periods=340,
        freq="1D"
    )

    data = pd.DataFrame(
        {
            "date": dates,
            "close": [100.0] * 340
        }
    )

    strategy = (
        MultiAssetTimeSeriesMomentumStrategy()
    )

    result = strategy.generate_signal(
        data=data,
        ticker="SPY"
    )

    assert result["action"] == "FLAT"
    assert result["signal"] == 0
    assert (
        "Volatilità"
        in result["reasons"][0]
    )


def test_insufficient_history_is_flat():

    strategy = (
        MultiAssetTimeSeriesMomentumStrategy()
    )

    result = strategy.generate_signal(
        data=create_trending_data(
            direction=1,
            rows=100
        ),
        ticker="SPY"
    )

    assert result["action"] == "FLAT"
    assert result["history_ready"] is False
    assert result["signal"] == 0


def test_signal_contains_all_components():

    strategy = (
        MultiAssetTimeSeriesMomentumStrategy()
    )

    result = strategy.generate_signal(
        data=create_trending_data(),
        ticker="SPY"
    )

    assert set(
        result["components"]
    ) == {
        "21",
        "63",
        "126",
        "252"
    }

    contribution_sum = sum(
        component[
            "weighted_contribution"
        ]
        for component in result[
            "components"
        ].values()
    )

    assert contribution_sum == pytest.approx(
        result["raw_signal"]
    )


def test_multiple_assets_are_processed():

    strategy = (
        MultiAssetTimeSeriesMomentumStrategy()
    )

    results = strategy.generate_signals(
        market_data={
            "SPY": create_trending_data(
                direction=1
            ),
            "TLT": create_trending_data(
                direction=-1
            )
        },
        asset_classes={
            "SPY": "EQUITY",
            "TLT": "BOND"
        }
    )

    assert set(results) == {
        "SPY",
        "TLT"
    }

    assert results["SPY"][
        "action"
    ] == "LONG"

    assert results["TLT"][
        "action"
    ] == "SHORT"

    assert results["TLT"][
        "asset_class"
    ] == "BOND"


def test_invalid_lookback_weights_are_rejected():

    with pytest.raises(
        ValueError
    ):
        MultiAssetTimeSeriesMomentumStrategy(
            lookback_weights={
                21: 1.0,
                63: -0.5
            }
        )