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