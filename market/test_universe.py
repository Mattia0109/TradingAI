from market.universe import MarketUniverse



def test_market_universe():


    universe = MarketUniverse()


    assets = universe.get_all_assets()


    assert "AAPL" in assets

    assert "BTC-USD" in assets

    assert "EURUSD=X" in assets

    assert "GC=F" in assets