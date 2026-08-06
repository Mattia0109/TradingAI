from data_engine.providers.yahoo_provider import YahooProvider



def test_yahoo_history():


    provider = YahooProvider()


    data = provider.get_history(

        "AAPL",

        period="5d"

    )


    assert len(data) > 0



def test_latest_price():


    provider = YahooProvider()


    price = provider.get_latest_price(

        "AAPL"

    )


    assert price > 0