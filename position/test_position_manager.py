from position.position_manager import PositionManager



def test_position_size():


    manager = PositionManager()


    quantity = manager.calculate_position_size(

        capital=10000,

        entry_price=100,

        stop_loss=95

    )


    assert quantity == 40




def test_position_plan():


    manager = PositionManager()


    plan = manager.create_position_plan(

        ticker="NVDA",

        direction="LONG",

        capital=10000,

        entry_price=100,

        stop_loss=95,

        take_profit=115

    )


    assert plan["ticker"] == "NVDA"

    assert plan["quantity"] == 40