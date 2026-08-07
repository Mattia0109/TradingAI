import pytest

from market.specifications import (
    ExecutionCostProfile,
    MarketSpecification,
    MarketSpecificationRegistry
)


def test_stock_quantity_is_integer():

    specification = (
        MarketSpecificationRegistry.get(
            "AAPL"
        )
    )

    quantity = (
        specification.normalize_quantity(
            requested_quantity=9.9,
            price=200
        )
    )

    assert quantity == 9
    assert (
        specification.asset_class
        == "STOCK"
    )


def test_crypto_supports_fractional_quantity():

    specification = (
        MarketSpecificationRegistry.get(
            "BTC-USD"
        )
    )

    quantity = (
        specification.normalize_quantity(
            requested_quantity=(
                0.0123456789
            ),
            price=50000
        )
    )

    assert quantity == 0.01234567
    assert (
        specification.supports_fractional
        is True
    )


def test_forex_tick_size_depends_on_pair():

    eurusd = (
        MarketSpecificationRegistry.get(
            "EURUSD=X"
        )
    )

    usdjpy = (
        MarketSpecificationRegistry.get(
            "USDJPY=X"
        )
    )

    assert eurusd.tick_size == 0.0001
    assert usdjpy.tick_size == 0.01


def test_gold_future_tick_value():

    specification = (
        MarketSpecificationRegistry.get(
            "GC=F"
        )
    )

    assert (
        specification.contract_multiplier
        == 100
    )

    assert specification.tick_size == 0.10
    assert specification.tick_value == 10


def test_crude_oil_future_tick_value():

    specification = (
        MarketSpecificationRegistry.get(
            "CL=F"
        )
    )

    assert (
        specification.contract_multiplier
        == 1000
    )

    assert specification.tick_size == 0.01
    assert specification.tick_value == 10


def test_future_pnl_uses_multiplier():

    specification = (
        MarketSpecificationRegistry.get(
            "CL=F"
        )
    )

    long_pnl = specification.calculate_pnl(
        entry_price=70.00,
        exit_price=70.10,
        quantity=1,
        direction="LONG"
    )

    short_pnl = specification.calculate_pnl(
        entry_price=70.10,
        exit_price=70.00,
        quantity=1,
        direction="SHORT"
    )

    assert round(long_pnl, 2) == 100
    assert round(short_pnl, 2) == 100


def test_micro_contracts_have_smaller_multiplier():

    gold = (
        MarketSpecificationRegistry.get(
            "MGC=F"
        )
    )

    oil = (
        MarketSpecificationRegistry.get(
            "MCL=F"
        )
    )

    assert gold.contract_multiplier == 10
    assert gold.tick_value == 1

    assert oil.contract_multiplier == 100
    assert oil.tick_value == 1


def test_unknown_future_is_rejected():

    with pytest.raises(
        KeyError
    ):
        MarketSpecificationRegistry.get(
            "UNKNOWN=F"
        )


def test_cost_profile_changes_fill_and_commission():

    costs = ExecutionCostProfile(
        name="test_costs",
        commission_model=(
            "percentage_notional"
        ),
        commission_value=0.001,
        minimum_commission=1.0,
        spread_bps=2.0,
        slippage_bps=1.0
    )

    specification = MarketSpecification(
        code="TEST",
        asset_class="STOCK",
        quote_currency="USD",
        quantity_step=1,
        minimum_quantity=1,
        contract_multiplier=1,
        tick_size=0.01,
        cost_profile=costs
    )

    commission = (
        specification.calculate_commission(
            price=100,
            quantity=20
        )
    )

    buy_fill = (
        specification.calculate_fill_price(
            reference_price=100,
            side="BUY"
        )
    )

    sell_fill = (
        specification.calculate_fill_price(
            reference_price=100,
            side="SELL"
        )
    )

    assert commission == 2
    assert buy_fill > 100
    assert sell_fill < 100


def test_external_margin_requires_configuration():

    specification = (
        MarketSpecificationRegistry.get(
            "GC=F"
        )
    )

    with pytest.raises(
        ValueError
    ):
        specification.calculate_capital_required(
            price=4000,
            quantity=1
        )