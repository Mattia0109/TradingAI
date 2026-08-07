import pytest

from backtesting.portfolio_risk_allocator import (
    PortfolioRiskAllocator
)


def build_signal(
    ticker,
    action,
    signal,
    volatility,
    asset_class
):
    return {
        "ticker": ticker,
        "action": action,
        "signal": signal,
        "annualized_volatility": volatility,
        "asset_class": asset_class
    }


def test_long_and_short_weights_are_preserved():

    allocator = PortfolioRiskAllocator(
        target_portfolio_volatility=1.0,
        max_gross_exposure=1.0,
        max_asset_weight=0.50,
        max_asset_class_weight=1.0,
        minimum_trade_weight=0
    )

    result = allocator.allocate(
        signals=[
            build_signal(
                "SPY",
                "LONG",
                1.0,
                0.20,
                "EQUITY"
            ),
            build_signal(
                "TLT",
                "SHORT",
                -1.0,
                0.20,
                "BOND"
            )
        ]
    )

    assert result[
        "weights"
    ][
        "SPY"
    ] > 0

    assert result[
        "weights"
    ][
        "TLT"
    ] < 0

    assert result[
        "gross_exposure"
    ] == pytest.approx(
        1.0
    )

    assert result[
        "net_exposure"
    ] == pytest.approx(
        0.0
    )


def test_single_asset_weight_is_capped():

    allocator = PortfolioRiskAllocator(
        target_portfolio_volatility=1.0,
        max_gross_exposure=1.0,
        max_asset_weight=0.10,
        max_asset_class_weight=1.0,
        minimum_trade_weight=0
    )

    result = allocator.allocate(
        signals=[
            build_signal(
                "SPY",
                "LONG",
                1.0,
                0.20,
                "EQUITY"
            )
        ]
    )

    assert result[
        "weights"
    ][
        "SPY"
    ] == pytest.approx(
        0.10
    )

    assert result[
        "gross_exposure"
    ] == pytest.approx(
        0.10
    )


def test_asset_class_exposure_is_capped():

    allocator = PortfolioRiskAllocator(
        target_portfolio_volatility=1.0,
        max_gross_exposure=1.0,
        max_asset_weight=0.20,
        max_asset_class_weight=0.30,
        minimum_trade_weight=0
    )

    signals = [
        build_signal(
            ticker=f"EQ{index}",
            action="LONG",
            signal=1.0,
            volatility=0.20,
            asset_class="EQUITY"
        )
        for index in range(
            5
        )
    ]

    result = allocator.allocate(
        signals=signals
    )

    assert (
        result[
            "asset_class_exposures"
        ][
            "EQUITY"
        ]
        <= 0.30 + 1e-12
    )


def test_high_volatility_reduces_exposure():

    allocator = PortfolioRiskAllocator(
        target_portfolio_volatility=0.10,
        max_gross_exposure=1.0,
        max_asset_weight=0.50,
        max_asset_class_weight=1.0,
        minimum_trade_weight=0
    )

    signals = [
        build_signal(
            ticker=f"A{index}",
            action="LONG",
            signal=1.0,
            volatility=1.0,
            asset_class=f"CLASS_{index}"
        )
        for index in range(
            4
        )
    ]

    result = allocator.allocate(
        signals=signals
    )

    assert (
        result[
            "estimated_volatility"
        ]
        <= 0.10 + 1e-12
    )

    assert result[
        "gross_exposure"
    ] < 1.0


def test_turnover_buffer_preserves_small_change():

    allocator = PortfolioRiskAllocator(
        target_portfolio_volatility=1.0,
        max_gross_exposure=1.0,
        max_asset_weight=0.10,
        max_asset_class_weight=1.0,
        minimum_trade_weight=0,
        turnover_buffer=0.01
    )

    result = allocator.allocate(
        signals=[
            build_signal(
                "SPY",
                "LONG",
                1.0,
                0.20,
                "EQUITY"
            )
        ],
        previous_weights={
            "SPY": 0.095
        }
    )

    assert result[
        "weights"
    ][
        "SPY"
    ] == pytest.approx(
        0.095
    )

    assert result[
        "turnover"
    ] == pytest.approx(
        0.0
    )


def test_drawdown_kill_switch_closes_positions():

    allocator = PortfolioRiskAllocator(
        drawdown_kill_switch=0.20
    )

    result = allocator.allocate(
        signals=[
            build_signal(
                "SPY",
                "LONG",
                1.0,
                0.20,
                "EQUITY"
            )
        ],
        previous_weights={
            "SPY": 0.10
        },
        current_drawdown=0.20
    )

    assert result[
        "kill_switch_active"
    ] is True

    assert result[
        "weights"
    ][
        "SPY"
    ] == 0

    assert result[
        "gross_exposure"
    ] == 0

    assert result[
        "turnover"
    ] == pytest.approx(
        0.10
    )


def test_severe_drawdown_preserves_recovery_exposure_until_kill_switch():

    allocator = PortfolioRiskAllocator(
        drawdown_kill_switch=0.20
    )

    multiplier = allocator.calculate_drawdown_multiplier(
        0.199999
    )

    assert multiplier >= 0.25

    result = allocator.allocate(
        signals=[
            build_signal(
                "SPY",
                "LONG",
                1.0,
                0.20,
                "EQUITY"
            )
        ],
        current_drawdown=0.199999
    )

    assert result[
        "kill_switch_active"
    ] is False

    assert result[
        "gross_exposure"
    ] > 0

    assert allocator.calculate_drawdown_multiplier(
        0.20
    ) == 0


def test_flat_and_invalid_signals_are_rejected():

    allocator = PortfolioRiskAllocator()

    result = allocator.allocate(
        signals=[
            build_signal(
                "SPY",
                "FLAT",
                0.0,
                0.20,
                "EQUITY"
            ),
            build_signal(
                "TLT",
                "LONG",
                1.0,
                0.0,
                "BOND"
            )
        ]
    )

    assert result[
        "gross_exposure"
    ] == 0

    assert "SPY" in result[
        "rejected"
    ]

    assert "TLT" in result[
        "rejected"
    ]
