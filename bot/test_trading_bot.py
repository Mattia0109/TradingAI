from bot.trading_bot import TradingBot



def test_bot_creation():


    bot = TradingBot()


    assert bot.running is False


    bot.start()


    assert bot.running is True


    bot.stop()


    assert bot.running is False