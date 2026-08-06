from engine.trading_pipeline import TradingPipeline



def test_trading_pipeline():

    pipeline = TradingPipeline()

    result = pipeline.generate_signal(
        "AAPL"
    )

    assert result is not None