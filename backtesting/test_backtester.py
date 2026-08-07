from datetime import datetime, timedelta

from backtesting.performance import PerformanceAnalyzer
from backtesting.portfolio import BacktestPortfolio


def test_long_trade_profit_without_costs():

    portfolio = BacktestPortfolio(
        initial_capital=10000,
        max_position_percent=0.10,
        commission_percent=0,
        slippage_percent=0
    )

    signal = {
        "ticker": "AAPL",
        "direction": "LONG",
        "confidence": 80,
        "entry_price": 100,
        "stop_loss": 95,
        "take_profit": 110,
        "risk_reward": 2,
        "strategy": "test"
    }

    position = portfolio.open_trade(
        signal=signal,
        timestamp=datetime.now(),
        asset_type="STOCK"
    )

    assert position is not None
    assert position["quantity"] == 10

    trade = portfolio.close_trade(
        exit_price=110,
        timestamp=datetime.now(),
        reason="TAKE_PROFIT"
    )

    assert trade["pnl"] == 100
    assert portfolio.cash == 10100


def test_short_trade_profit_without_costs():

    portfolio = BacktestPortfolio(
        initial_capital=10000,
        commission_percent=0,
        slippage_percent=0
    )

    signal = {
        "ticker": "AAPL",
        "direction": "SHORT",
        "confidence": 80,
        "entry_price": 100,
        "stop_loss": 105,
        "take_profit": 90,
        "risk_reward": 2,
        "strategy": "test"
    }

    portfolio.open_trade(
        signal=signal,
        timestamp=datetime.now(),
        asset_type="STOCK"
    )

    trade = portfolio.close_trade(
        exit_price=90,
        timestamp=datetime.now(),
        reason="TAKE_PROFIT"
    )

    assert trade["pnl"] == 100
    assert portfolio.cash == 10100


def test_crypto_fractional_quantity():

    portfolio = BacktestPortfolio(
        initial_capital=10000,
        max_position_percent=0.10,
        commission_percent=0,
        slippage_percent=0
    )

    signal = {
        "ticker": "BTC-USD",
        "direction": "LONG",
        "confidence": 80,
        "entry_price": 50000,
        "stop_loss": 48000,
        "take_profit": 54000,
        "risk_reward": 2,
        "strategy": "test"
    }

    position = portfolio.open_trade(
        signal=signal,
        timestamp=datetime.now(),
        asset_type="CRYPTO"
    )

    assert position is not None
    assert position["quantity"] == 0.02
    assert position["position_value"] == 1000


def test_costs_reduce_profit():

    portfolio = BacktestPortfolio(
        initial_capital=10000,
        max_position_percent=0.10,
        commission_percent=0.001,
        slippage_percent=0.001
    )

    signal = {
        "ticker": "AAPL",
        "direction": "LONG",
        "confidence": 80,
        "entry_price": 100,
        "stop_loss": 95,
        "take_profit": 110,
        "risk_reward": 2,
        "strategy": "test"
    }

    portfolio.open_trade(
        signal=signal,
        timestamp=datetime.now(),
        asset_type="STOCK"
    )

    trade = portfolio.close_trade(
        exit_price=110,
        timestamp=datetime.now(),
        reason="TAKE_PROFIT"
    )

    assert trade["pnl"] < 100
    assert trade["entry_commission"] > 0
    assert trade["exit_commission"] > 0


def test_performance_metrics():

    analyzer = PerformanceAnalyzer()

    start_time = datetime(
        2026,
        1,
        1,
        10,
        0
    )

    trades = [
        {
            "pnl": 100,
            "return_percent": 2,
            "open_time": start_time,
            "close_time": (
                start_time +
                timedelta(hours=1)
            )
        },
        {
            "pnl": -50,
            "return_percent": -1,
            "open_time": (
                start_time +
                timedelta(hours=2)
            ),
            "close_time": (
                start_time +
                timedelta(hours=3)
            )
        },
        {
            "pnl": 200,
            "return_percent": 4,
            "open_time": (
                start_time +
                timedelta(hours=4)
            ),
            "close_time": (
                start_time +
                timedelta(hours=5)
            )
        }
    ]

    equity_curve = [
        {
            "timestamp": start_time,
            "equity": 10000
        },
        {
            "timestamp": (
                start_time +
                timedelta(hours=1)
            ),
            "equity": 10100
        },
        {
            "timestamp": (
                start_time +
                timedelta(hours=3)
            ),
            "equity": 10050
        },
        {
            "timestamp": (
                start_time +
                timedelta(hours=6)
            ),
            "equity": 10250
        }
    ]

    result = analyzer.calculate(
        initial_capital=10000,
        final_capital=10250,
        trades=trades,
        equity_curve=equity_curve
    )

    assert result["net_profit"] == 250
    assert result["total_trades"] == 3

    assert round(
        result["win_rate_percent"],
        2
    ) == 66.67

    assert result["profit_factor"] == 6

    assert result["average_win"] == 150
    assert result["average_loss"] == 50
    assert result["payoff_ratio"] == 3

    assert round(
        result["expectancy"],
        2
    ) == 83.33

    assert result[
        "max_consecutive_losses"
    ] == 1

    assert result[
        "max_consecutive_wins"
    ] == 1

    assert result[
        "statistical_warning"
    ] is True


def test_consecutive_losses():

    analyzer = PerformanceAnalyzer()

    trades = [
        {"pnl": -10},
        {"pnl": -20},
        {"pnl": -30},
        {"pnl": 50},
        {"pnl": -5}
    ]

    result = (
        analyzer.calculate_max_consecutive_losses(
            trades
        )
    )

    assert result == 3