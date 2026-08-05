from scanner.market_scanner import MarketScanner



def test_scanner_creation():


    scanner = MarketScanner()


    tickers = scanner.get_tickers()


    assert "AAPL" in tickers

    assert "BTC-USD" in tickers

    assert "EURUSD" in tickers

    assert "GC=F" in tickers



def test_scanner_scan():


    scanner = MarketScanner()


    result = scanner.scan()


    assert result["count"] > 0

    assert len(result["assets"]) > 0