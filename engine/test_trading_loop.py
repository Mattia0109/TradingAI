from engine.trading_loop import TradingLoop



def test_trading_loop():

    bot = TradingLoop()


    result = bot.run_once()


    assert result is not None