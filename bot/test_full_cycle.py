from bot.trading_bot import TradingBot



class FakeDecisionEngine:


    def evaluate(self, asset):

        return {

            "decision": "WAIT"

        }



class FakeRanker:


    def rank(self, opportunities):

        return opportunities




def test_bot_cycle():


    bot = TradingBot(

        decision_engine=FakeDecisionEngine(),

        ranker=FakeRanker()

    )


    bot.start()


    result = bot.run_cycle()


    assert result is None