from data_engine.market_data_engine import MarketDataEngine



def test_market_data_snapshot():


    engine = MarketDataEngine()


    data = engine.get_snapshot(

        "BTC-USD",

        60000

    )


    assert data["ticker"] == "BTC-USD"

    assert data["price"] == 60000

    assert data["timestamp"] is not None



def test_cache():


    engine = MarketDataEngine()


    engine.get_snapshot(

        "AAPL",

        200

    )


    saved = engine.get_cached_data(

        "AAPL"

    )


    assert saved["price"] == 200