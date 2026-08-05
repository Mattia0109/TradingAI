from scanner.market_scanner import MarketScanner



def test_scanner_creation():

    scanner = MarketScanner()

    tickers = scanner.get_tickers()

    print(tickers)


    assert len(tickers) > 0