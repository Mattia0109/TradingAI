from engine.trading_loop import TradingLoop


def test_trading_loop():
    bot = TradingLoop()
    candidate = {
        "ticker": "AAA",
        "signal": {"direction": "LONG", "confidence": 80},
        "score": 80,
    }
    bot.scanner.get_tickers = lambda: ["AAA", "BBB"]
    bot.pipeline.generate_signal = (
        lambda ticker: candidate if ticker == "AAA" else None
    )
    bot.ranker.rank = lambda opportunities: opportunities

    result = bot.run_once()

    assert result == [candidate]
