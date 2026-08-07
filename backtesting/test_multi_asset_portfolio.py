import pandas as pd
import pytest

from backtesting.multi_asset_portfolio import (
    MultiAssetPortfolio
)


def test_long_position_generates_positive_pnl():

    portfolio = MultiAssetPortfolio(
        initial_capital=10000,
        use_market_costs=False
    )

    portfolio.rebalance(
        target_weights={
            "SPY": 0.50
        },
        prices={
            "SPY": 100
        },
        timestamp=pd.Timestamp(
            "2025-01-01"
        )
    )

    assert portfolio.positions[
        "SPY"
    ] == 50

    result = portfolio.mark_to_market(
        prices={
            "SPY": 110
        },
        timestamp=pd.Timestamp(
            "2025-01-02"
        )
    )

    assert result[
        "daily_pnl"
    ] == pytest.approx(
        500
    )

    assert portfolio.equity == pytest.approx(
        10500
    )


def test_short_position_generates_positive_pnl():

    portfolio = MultiAssetPortfolio(
        initial_capital=10000,
        use_market_costs=False
    )

    portfolio.rebalance(
        target_weights={
            "SPY": -0.50
        },
        prices={
            "SPY": 100
        },
        timestamp=pd.Timestamp(
            "2025-01-01"
        )
    )

    assert portfolio.positions[
        "SPY"
    ] == -50

    result = portfolio.mark_to_market(
        prices={
            "SPY": 90
        },
        timestamp=pd.Timestamp(
            "2025-01-02"
        )
    )

    assert result[
        "daily_pnl"
    ] == pytest.approx(
        500
    )

    assert portfolio.equity == pytest.approx(
        10500
    )


def test_market_costs_reduce_equity():

    portfolio = MultiAssetPortfolio(
        initial_capital=10000,
        use_market_costs=True
    )

    result = portfolio.rebalance(
        target_weights={
            "AAPL": 0.10
        },
        prices={
            "AAPL": 100
        },
        timestamp=pd.Timestamp(
            "2025-01-01"
        )
    )

    assert result[
        "total_cost"
    ] > 0

    assert portfolio.total_costs > 0
    assert portfolio.total_commissions >= 1
    assert portfolio.equity < 10000


def test_rebalance_closes_missing_target():

    portfolio = MultiAssetPortfolio(
        initial_capital=10000,
        use_market_costs=False
    )

    portfolio.rebalance(
        target_weights={
            "SPY": 0.40,
            "TLT": -0.30
        },
        prices={
            "SPY": 100,
            "TLT": 100
        },
        timestamp=pd.Timestamp(
            "2025-01-01"
        )
    )

    assert portfolio.position_count == 2

    portfolio.rebalance(
        target_weights={
            "SPY": 0.20
        },
        prices={
            "SPY": 100,
            "TLT": 100
        },
        timestamp=pd.Timestamp(
            "2025-01-02"
        )
    )

    assert portfolio.positions[
        "SPY"
    ] == 20

    assert "TLT" not in portfolio.positions
    assert portfolio.position_count == 1


def test_crypto_quantity_is_fractional():

    portfolio = MultiAssetPortfolio(
        initial_capital=10000,
        use_market_costs=False
    )

    portfolio.rebalance(
        target_weights={
            "BTC-USD": 0.10
        },
        prices={
            "BTC-USD": 50000
        },
        timestamp=pd.Timestamp(
            "2025-01-01"
        )
    )

    assert portfolio.positions[
        "BTC-USD"
    ] == pytest.approx(
        0.02
    )


def test_liquidation_closes_all_positions():

    portfolio = MultiAssetPortfolio(
        initial_capital=10000,
        use_market_costs=False
    )

    portfolio.rebalance(
        target_weights={
            "SPY": 0.40,
            "TLT": -0.30
        },
        prices={
            "SPY": 100,
            "TLT": 100
        },
        timestamp=pd.Timestamp(
            "2025-01-01"
        )
    )

    liquidation = portfolio.liquidate(
        prices={
            "SPY": 105,
            "TLT": 95
        },
        timestamp=pd.Timestamp(
            "2025-01-02"
        )
    )

    assert portfolio.is_flat is True
    assert portfolio.position_count == 0
    assert len(
        liquidation["orders"]
    ) == 2


def test_uneconomic_resize_is_batched_but_close_executes():

    portfolio = MultiAssetPortfolio(
        initial_capital=10000,
        use_market_costs=True,
        maximum_resize_cost_ratio=0.0025
    )

    portfolio.rebalance(
        target_weights={
            "AAPL": 0.10
        },
        prices={
            "AAPL": 100
        },
        timestamp=pd.Timestamp(
            "2025-01-01"
        )
    )

    small_resize = portfolio.rebalance(
        target_weights={
            "AAPL": 0.12
        },
        prices={
            "AAPL": 100
        },
        timestamp=pd.Timestamp(
            "2025-01-02"
        )
    )

    assert small_resize[
        "orders"
    ] == []

    assert small_resize[
        "uneconomic_resize_skips"
    ] == 1

    assert portfolio.positions[
        "AAPL"
    ] == 10

    accumulated_resize = portfolio.rebalance(
        target_weights={
            "AAPL": 0.20
        },
        prices={
            "AAPL": 100
        },
        timestamp=pd.Timestamp(
            "2025-01-03"
        )
    )

    assert len(
        accumulated_resize[
            "orders"
        ]
    ) == 1

    risk_reduction = portfolio.rebalance(
        target_weights={
            "AAPL": 0.18
        },
        prices={
            "AAPL": 100
        },
        timestamp=pd.Timestamp(
            "2025-01-04"
        )
    )

    assert len(
        risk_reduction[
            "orders"
        ]
    ) == 1

    reduction_order = risk_reduction[
        "orders"
    ][0]

    assert (
        reduction_order[
            "total_cost"
        ]
        /
        reduction_order[
            "trade_notional"
        ]
        >
        0.0025
    )

    liquidation = portfolio.rebalance(
        target_weights={
            "AAPL": 0.0
        },
        prices={
            "AAPL": 100
        },
        timestamp=pd.Timestamp(
            "2025-01-05"
        )
    )

    assert len(
        liquidation[
            "orders"
        ]
    ) == 1

    assert portfolio.is_flat is True


def test_invalid_maximum_resize_cost_ratio_is_rejected():

    with pytest.raises(
        ValueError
    ):
        MultiAssetPortfolio(
            maximum_resize_cost_ratio=0.0
        )
