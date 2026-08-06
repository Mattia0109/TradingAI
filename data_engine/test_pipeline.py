from data_engine.pipeline import MarketDataPipeline



def test_pipeline_market_data():


    pipeline = MarketDataPipeline()


    data = pipeline.get_market_data(

        "AAPL"

    )


    assert data is not None


    assert data["ticker"] == "AAPL"


    assert data["close"] > 0


    assert data["volume"] >= 0