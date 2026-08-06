from bot.trading_bot import TradingBot
from decision.market_decision_engine import MarketDecisionEngine
from ranking.opportunity_ranker import OpportunityRanker



def test_bot_creation():

    decision_engine = MarketDecisionEngine()

    ranker = OpportunityRanker()


    bot = TradingBot(
        decision_engine,
        ranker
    )


    assert bot.running is False


    bot.start()

    assert bot.running is True


    bot.stop()

    assert bot.running is False