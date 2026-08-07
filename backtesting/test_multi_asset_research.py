import numpy as np
import pandas as pd
import pytest

from backtesting.multi_asset_research import (
    MultiAssetResearchBenchmark,
    ResearchMultiAssetBacktester,
    StaticAllocationBacktester
)
from backtesting.portfolio_risk_allocator import (
    PortfolioRiskAllocator
)
from strategies.multi_asset_tsmom import (
    MultiAssetTimeSeriesMomentumStrategy
)


def create_asset(
    direction=1,
    rows=500,
    start="2018-01-01",
    start_price=100.0
):
    dates = pd.date_range(
        start=start,
        periods=rows,
        freq="1D"
    )

    index = np.arange(
        rows
    )

    trend = (
        direction
        *
        0.0012
        *
        index
    )

    oscillation = (
        0.01
        *
        np.sin(
            index / 8
        )
    )

    prices = (
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
            "close": prices
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
        max_gross_exposure=0.90,
        max_asset_weight=0.40,
        max_asset_class_weight=0.60,
        minimum_trade_weight=0,
        turnover_buffer=0
    )


def create_market_data():
    return {
        "SPY": create_asset(
            direction=1,
            start_price=100
        ),
        "TLT": create_asset(
            direction=-1,
            start_price=100
        ),
        "BTC-USD": create_asset(
            direction=1,
            start_price=1000
        )
    }


def create_asset_classes():
    return {
        "SPY": "EQUITY",
        "TLT": "BOND",
        "BTC-USD": "CRYPTO"
    }


def test_research_backtester_respects_trade_start():

    data = create_market_data()

    start_time = data[
        "SPY"
    ].iloc[
        320
    ][
        "date"
    ]

    backtester = (
        ResearchMultiAssetBacktester(
            strategy=build_strategy(),
            allocator=build_allocator(),
            initial_capital=10000,
            rebalance_frequency=21,
            use_market_costs=False,
            minimum_active_assets=2
        )
    )

    result = backtester.run(
        market_data=data,
        asset_classes=(
            create_asset_classes()
        ),
        trade_start_time=start_time
    )

    assert (
        result[
            "evaluation_start_time"
        ]
        >=
        start_time
    )

    assert result[
        "orders"
    ]

    for order in result[
        "orders"
    ]:
        if (
            order["reason"]
            ==
            "SCHEDULED_REBALANCE"
        ):
            assert (
                order[
                    "source_signal_time"
                ]
                >=
                start_time
            )

            assert (
                order[
                    "timestamp"
                ]
                >
                order[
                    "source_signal_time"
                ]
            )


def test_research_equity_curve_starts_at_evaluation():

    data = create_market_data()

    start_time = data[
        "SPY"
    ].iloc[
        320
    ][
        "date"
    ]

    backtester = (
        ResearchMultiAssetBacktester(
            strategy=build_strategy(),
            allocator=build_allocator(),
            initial_capital=10000,
            rebalance_frequency=21,
            use_market_costs=False,
            minimum_active_assets=2
        )
    )

    result = backtester.run(
        market_data=data,
        asset_classes=(
            create_asset_classes()
        ),
        trade_start_time=start_time
    )

    first_snapshot = result[
        "equity_curve"
    ][0]

    assert (
        first_snapshot[
            "timestamp"
        ]
        >=
        start_time
    )

    assert first_snapshot[
        "equity"
    ] == pytest.approx(
        10000
    )


def test_static_buy_and_hold_generates_profit():

    backtester = StaticAllocationBacktester(
        initial_capital=10000,
        use_market_costs=False
    )

    data = {
        "SPY": create_asset(
            direction=1
        )
    }

    start_time = data[
        "SPY"
    ].iloc[
        300
    ][
        "date"
    ]

    result = backtester.run(
        market_data=data,
        target_weights={
            "SPY": 1.0
        },
        trade_start_time=start_time,
        name="SPY_BUY_HOLD"
    )

    assert result[
        "metrics"
    ][
        "net_profit"
    ] > 0

    assert result[
        "first_execution_time"
    ] > start_time

    assert result[
        "final_positions"
    ] == {}


def test_common_start_requires_history_for_all_assets():

    strategy = build_strategy()

    benchmark = MultiAssetResearchBenchmark(
        strategy=strategy,
        allocator=build_allocator(),
        rebalance_frequencies=[
            21
        ],
        use_market_costs=False
    )

    market_data = {
        "SPY": create_asset(
            rows=500,
            start="2018-01-01"
        ),
        "TLT": create_asset(
            rows=500,
            start="2019-01-01"
        )
    }

    information = (
        benchmark.determine_common_start(
            market_data
        )
    )

    expected = (
        market_data[
            "TLT"
        ].iloc[
            strategy.minimum_history
            -
            1
        ][
            "date"
        ]
    )

    assert (
        information[
            "trade_start_time"
        ]
        ==
        expected
    )


def test_complete_benchmark_runs():

    benchmark = MultiAssetResearchBenchmark(
        strategy=build_strategy(),
        allocator=build_allocator(),
        initial_capital=10000,
        rebalance_frequencies=[
            5,
            21,
            63
        ],
        use_market_costs=False,
        minimum_active_assets=2
    )

    result = benchmark.run(
        market_data=create_market_data(),
        asset_classes=(
            create_asset_classes()
        )
    )

    assert len(
        result[
            "tsmom_results"
        ]
    ) == 3

    assert len(
        result[
            "passive_results"
        ]
    ) == 2

    names = {
        item[
            "benchmark_name"
        ]
        for item in result[
            "ranked_results"
        ]
    }

    assert names == {
        "TSMOM_5D",
        "TSMOM_21D",
        "TSMOM_63D",
        "EQUAL_WEIGHT_BUY_HOLD",
        "BUY_HOLD_60_40"
    }


def test_yearly_returns_are_generated():

    benchmark = MultiAssetResearchBenchmark(
        strategy=build_strategy(),
        allocator=build_allocator(),
        rebalance_frequencies=[
            21
        ],
        use_market_costs=False,
        minimum_active_assets=2
    )

    result = benchmark.run(
        market_data=create_market_data(),
        asset_classes=(
            create_asset_classes()
        )
    )

    strategy_result = result[
        "tsmom_results"
    ][0]

    assert strategy_result[
        "yearly_returns"
    ]

    assert all(
        isinstance(
            year,
            int
        )
        for year in strategy_result[
            "yearly_returns"
        ]
    )