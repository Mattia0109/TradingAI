from datetime import datetime

from backtesting.portfolio import (
    BacktestPortfolio
)


def build_signal(
    ticker,
    direction,
    entry_price,
    stop_loss,
    take_profit
):
    return {
        "ticker": ticker,
        "direction": direction,
        "confidence": 80,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk_reward": 2,
        "strategy": "market_spec_test"
    }


def test_stock_uses_integer_quantity():

    portfolio = BacktestPortfolio(
        initial_capital=10000,
        max_position_percent=0.10,
        commission_percent=0,
        slippage_percent=0
    )

    position = portfolio.open_trade(
        signal=build_signal(
            ticker="AAPL",
            direction="LONG",
            entry_price=150,
            stop_loss=145,
            take_profit=160
        ),
        timestamp=datetime.now()
    )

    assert position is not None
    assert position["quantity"] == 6
    assert position["asset_type"] == "STOCK"
    assert position["contract_multiplier"] == 1


def test_crypto_uses_fractional_quantity():

    portfolio = BacktestPortfolio(
        initial_capital=10000,
        max_position_percent=0.10,
        commission_percent=0,
        slippage_percent=0
    )

    position = portfolio.open_trade(
        signal=build_signal(
            ticker="BTC-USD",
            direction="LONG",
            entry_price=50000,
            stop_loss=48000,
            take_profit=54000
        ),
        timestamp=datetime.now()
    )

    assert position is not None
    assert position["quantity"] == 0.02
    assert position["position_value"] == 1000
    assert position["asset_type"] == "CRYPTO"


def test_future_requires_margin_override():

    portfolio = BacktestPortfolio(
        initial_capital=100000,
        max_position_percent=0.10,
        commission_percent=0,
        slippage_percent=0
    )

    position = portfolio.open_trade(
        signal=build_signal(
            ticker="CL=F",
            direction="LONG",
            entry_price=70,
            stop_loss=69,
            take_profit=72
        ),
        timestamp=datetime.now()
    )

    assert position is None


def test_future_uses_contract_multiplier():

    portfolio = BacktestPortfolio(
        initial_capital=100000,
        max_position_percent=0.10,
        commission_percent=0,
        slippage_percent=0,
        margin_overrides={
            "CL=F": 5000
        }
    )

    position = portfolio.open_trade(
        signal=build_signal(
            ticker="CL=F",
            direction="LONG",
            entry_price=70,
            stop_loss=69,
            take_profit=72
        ),
        timestamp=datetime.now()
    )

    assert position is not None
    assert position["quantity"] == 2
    assert position["contract_multiplier"] == 1000

    trade = portfolio.close_trade(
        exit_price=70.10,
        timestamp=datetime.now(),
        reason="TEST_EXIT"
    )

    assert round(
        trade["gross_pnl"],
        2
    ) == 200

    assert round(
        trade["pnl"],
        2
    ) == 200

    assert round(
        portfolio.cash,
        2
    ) == 100200


def test_market_cost_profile_is_applied():

    portfolio = BacktestPortfolio(
        initial_capital=10000,
        max_position_percent=0.10,
        use_market_costs=True
    )

    position = portfolio.open_trade(
        signal=build_signal(
            ticker="AAPL",
            direction="LONG",
            entry_price=100,
            stop_loss=95,
            take_profit=110
        ),
        timestamp=datetime.now()
    )

    assert position is not None
    assert position["entry_price"] > 100
    assert position["entry_commission"] >= 1
    assert portfolio.total_commissions >= 1


def test_short_trade_uses_market_specification():

    portfolio = BacktestPortfolio(
        initial_capital=10000,
        max_position_percent=0.10,
        commission_percent=0,
        slippage_percent=0
    )

    position = portfolio.open_trade(
        signal=build_signal(
            ticker="AAPL",
            direction="SHORT",
            entry_price=100,
            stop_loss=105,
            take_profit=90
        ),
        timestamp=datetime.now()
    )

    assert position is not None
    assert position["quantity"] == 10

    trade = portfolio.close_trade(
        exit_price=90,
        timestamp=datetime.now(),
        reason="TAKE_PROFIT"
    )

    assert trade["gross_pnl"] == 100
    assert trade["pnl"] == 100
    assert portfolio.cash == 10100