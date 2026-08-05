from execution.execution_engine import ExecutionEngine



def test_execution_engine():


    engine = ExecutionEngine()


    result = engine.execute_trade(

        ticker="NVDA",

        direction="LONG",

        capital=10000,

        entry_price=100,

        stop_loss=95,

        take_profit=115

    )


    position = result["position"]


    assert position["ticker"] == "NVDA"

    assert position["quantity"] == 40