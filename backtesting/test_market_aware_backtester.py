import pandas as pd

from backtesting.backtester import Backtester
from backtesting.walk_forward import WalkForwardValidator


class AlwaysBuyStrategy:
    name = "always_buy_market_test"

    def __init__(
        self,
        atr=0.50
    ):
        self.atr = float(
            atr
        )

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
            "atr": self.atr,
            "reasons": [
                "Segnale deterministico"
            ]
        }


def create_historical_data(
    rows=260,
    start_price=100.0,
    price_step=0.10
):
    dates = pd.date_range(
        start="2026-01-01",
        periods=rows,
        freq="5min"
    )

    close_prices = [
        start_price
        +
        index
        *
        price_step
        for index in range(rows)
    ]

    return pd.DataFrame(
        {
            "date": dates,
            "open": [
                price - 0.10
                for price in close_prices
            ],
            "high": [
                price + 0.40
                for price in close_prices
            ],
            "low": [
                price - 0.40
                for price in close_prices
            ],
            "close": close_prices,
            "volume": [
                1_000_000
            ] * rows
        }
    )


def test_market_costs_are_applied_by_backtester():

    data = create_historical_data()

    zero_cost_backtester = Backtester(
        initial_capital=10000,
        minimum_history=30,
        commission_percent=0,
        slippage_percent=0,
        max_holding_bars=2,
        cooldown_bars=1,
        max_trades_per_day=20,
        strategy=AlwaysBuyStrategy(),
        use_market_costs=False
    )

    market_cost_backtester = Backtester(
        initial_capital=10000,
        minimum_history=30,
        max_holding_bars=2,
        cooldown_bars=1,
        max_trades_per_day=20,
        strategy=AlwaysBuyStrategy(),
        use_market_costs=True
    )

    zero_result = zero_cost_backtester.run(
        ticker="AAPL",
        data=data
    )

    market_result = (
        market_cost_backtester.run(
            ticker="AAPL",
            data=data
        )
    )

    assert (
        zero_result["cost_model"]
        == "zero_costs"
    )

    assert (
        market_result["cost_model"]
        == "market_specification"
    )

    assert (
        zero_result["metrics"][
            "total_commissions"
        ]
        == 0
    )

    assert (
        market_result["metrics"][
            "total_commissions"
        ]
        > 0
    )

    assert (
        market_result["metrics"][
            "final_capital"
        ]
        <
        zero_result["metrics"][
            "final_capital"
        ]
    )


def test_future_without_margin_override_has_no_trades():

    data = create_historical_data(
        start_price=70,
        price_step=0.02
    )

    backtester = Backtester(
        initial_capital=100000,
        minimum_history=30,
        commission_percent=0,
        slippage_percent=0,
        max_holding_bars=2,
        cooldown_bars=1,
        max_trades_per_day=20,
        strategy=AlwaysBuyStrategy(
            atr=0.05
        )
    )

    result = backtester.run(
        ticker="CL=F",
        data=data
    )

    assert result["asset_type"] == "FUTURE"

    assert (
        result["metrics"][
            "total_trades"
        ]
        == 0
    )


def test_future_with_margin_override_uses_multiplier():

    data = create_historical_data(
        start_price=70,
        price_step=0.05
    )

    backtester = Backtester(
        initial_capital=100000,
        max_position_percent=0.10,
        minimum_history=30,
        commission_percent=0,
        slippage_percent=0,
        max_holding_bars=2,
        cooldown_bars=1,
        max_trades_per_day=20,
        strategy=AlwaysBuyStrategy(
            atr=0.05
        ),
        margin_overrides={
            "CL=F": 5000
        }
    )

    result = backtester.run(
        ticker="CL=F",
        data=data
    )

    assert (
        result["metrics"][
            "total_trades"
        ]
        > 0
    )

    assert (
        result[
            "market_specification"
        ][
            "contract_multiplier"
        ]
        == 1000
    )

    first_trade = result[
        "trades"
    ][0]

    assert (
        first_trade[
            "contract_multiplier"
        ]
        == 1000
    )

    assert first_trade[
        "quantity"
    ] == 2


def test_walk_forward_propagates_market_costs():

    data = create_historical_data(
        rows=300
    )

    validator = WalkForwardValidator(
        train_bars=100,
        test_bars=50,
        step_bars=50,
        initial_capital=10000,
        minimum_history=30,
        max_holding_bars=2,
        cooldown_bars=1,
        max_trades_per_day=20,
        strategy=AlwaysBuyStrategy(),
        use_market_costs=True
    )

    result = validator.run(
        ticker="AAPL",
        data=data
    )

    assert (
        result["cost_model"]
        == "market_specification"
    )

    assert (
        result["summary"][
            "total_trades"
        ]
        > 0
    )

    assert (
        result["summary"][
            "total_commissions"
        ]
        > 0
    )

    for fold in result["folds"]:
        assert (
            fold["cost_model"]
            == "market_specification"
        )

        for trade in fold["trades"]:
            assert (
                trade[
                    "entry_commission"
                ]
                >= 1
            )