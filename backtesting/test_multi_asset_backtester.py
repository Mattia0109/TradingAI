import numpy as np
import pandas as pd

from backtesting.multi_asset_backtester import (
    MultiAssetBacktester
)
from backtesting.portfolio_risk_allocator import (
    PortfolioRiskAllocator
)
from strategies.multi_asset_tsmom import (
    MultiAssetTimeSeriesMomentumStrategy
)


def create_trending_asset(
    direction,
    rows=390,
    start_price=100.0
):
    dates = pd.date_range(
        start="2020-01-01",
        periods=rows,
        freq="1D"
    )

    index = np.arange(
        rows
    )

    trend = (
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
            index / 9
        )
    )

    close = (
        start_price
        *
        np.exp(
            trend
            +
            oscillation
        )
    )

    return pd.DataFrame(
        {
            "date": dates,
            "close": close
        }
    )


def build_strategy():
    return (
        MultiAssetTimeSeriesMomentumStrategy(
            no_trade_threshold=0.05
        )
    )


def build_allocator():
    return PortfolioRiskAllocator(
        target_portfolio_volatility=1.0,
        max_gross_exposure=0.80,
        max_asset_weight=0.40,
        max_asset_class_weight=0.50,
        minimum_trade_weight=0,
        turnover_buffer=0
    )


def create_market_data():
    return {
        "SPY": create_trending_asset(
            direction=1,
            start_price=100
        ),
        "TLT": create_trending_asset(
            direction=-1,
            start_price=100
        )
    }


def create_asset_classes():
    return {
        "SPY": "EQUITY",
        "TLT": "BOND"
    }


def test_backtester_holds_multiple_positions():

    backtester = MultiAssetBacktester(
        strategy=build_strategy(),
        allocator=build_allocator(),
        initial_capital=10000,
        rebalance_frequency=21,
        use_market_costs=False
    )

    result = backtester.run(
        market_data=create_market_data(),
        asset_classes=create_asset_classes()
    )

    assert (
        result["metrics"][
            "maximum_simultaneous_positions"
        ]
        >= 2
    )

    assert (
        result["metrics"][
            "order_count"
        ]
        > 0
    )

    assert result[
        "final_positions"
    ] == {}


def test_perfect_trends_generate_profit():

    backtester = MultiAssetBacktester(
        strategy=build_strategy(),
        allocator=build_allocator(),
        initial_capital=10000,
        rebalance_frequency=21,
        use_market_costs=False
    )

    result = backtester.run(
        market_data=create_market_data(),
        asset_classes=create_asset_classes()
    )

    assert (
        result["metrics"][
            "net_profit"
        ]
        > 0
    )

    assert (
        result["metrics"][
            "total_return_percent"
        ]
        > 0
    )

    assert result[
        "bankrupt"
    ] is False


def test_signals_execute_only_on_later_bar():

    backtester = MultiAssetBacktester(
        strategy=build_strategy(),
        allocator=build_allocator(),
        initial_capital=10000,
        rebalance_frequency=21,
        use_market_costs=False
    )

    result = backtester.run(
        market_data=create_market_data(),
        asset_classes=create_asset_classes()
    )

    scheduled_orders = [
        order
        for order in result[
            "orders"
        ]
        if order[
            "reason"
        ] == "SCHEDULED_REBALANCE"
    ]

    assert scheduled_orders

    for order in scheduled_orders:
        assert (
            order[
                "source_signal_time"
            ]
            <
            order[
                "timestamp"
            ]
        )


def test_market_costs_reduce_final_capital():

    zero_cost_backtester = (
        MultiAssetBacktester(
            strategy=build_strategy(),
            allocator=build_allocator(),
            initial_capital=10000,
            rebalance_frequency=21,
            use_market_costs=False
        )
    )

    market_cost_backtester = (
        MultiAssetBacktester(
            strategy=build_strategy(),
            allocator=build_allocator(),
            initial_capital=10000,
            rebalance_frequency=21,
            use_market_costs=True
        )
    )

    zero_cost_result = (
        zero_cost_backtester.run(
            market_data=create_market_data(),
            asset_classes=create_asset_classes()
        )
    )

    market_cost_result = (
        market_cost_backtester.run(
            market_data=create_market_data(),
            asset_classes=create_asset_classes()
        )
    )

    assert (
        market_cost_result[
            "metrics"
        ][
            "total_costs"
        ]
        > 0
    )

    assert (
        market_cost_result[
            "metrics"
        ][
            "final_capital"
        ]
        <
        zero_cost_result[
            "metrics"
        ][
            "final_capital"
        ]
    )


def test_risk_limits_are_respected():

    backtester = MultiAssetBacktester(
        strategy=build_strategy(),
        allocator=build_allocator(),
        initial_capital=10000,
        rebalance_frequency=21,
        use_market_costs=False
    )

    result = backtester.run(
        market_data=create_market_data(),
        asset_classes=create_asset_classes()
    )

    for snapshot in result[
        "equity_curve"
    ]:
        assert (
            snapshot[
                "gross_exposure"
            ]
            <= 0.85
        )


def test_equity_curve_is_created():

    backtester = MultiAssetBacktester(
        strategy=build_strategy(),
        allocator=build_allocator(),
        initial_capital=10000,
        rebalance_frequency=21,
        use_market_costs=False
    )

    result = backtester.run(
        market_data=create_market_data(),
        asset_classes=create_asset_classes()
    )

    assert len(
        result[
            "equity_curve"
        ]
    ) >= 390

    assert (
        result[
            "equity_curve"
        ][0][
            "equity"
        ]
        == 10000
    )

    assert (
        result[
            "metrics"
        ][
            "observation_count"
        ]
        > 0
    )