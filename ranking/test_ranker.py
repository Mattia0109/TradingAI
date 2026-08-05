from datetime import datetime

from core.models import TradeSignal

from ranking.opportunity_ranker import OpportunityRanker



def test_ranker():


    signal1 = TradeSignal(

        ticker="NVDA",

        timestamp=datetime.now(),

        direction="LONG",

        confidence=90,

        entry_price=180,

        stop_loss=175,

        take_profit=195,

        risk_reward=3,

        timeframe="5m",

        strategy="momentum"

    )


    signal2 = TradeSignal(

        ticker="AAPL",

        timestamp=datetime.now(),

        direction="LONG",

        confidence=60,

        entry_price=200,

        stop_loss=198,

        take_profit=204,

        risk_reward=1.5,

        timeframe="5m",

        strategy="momentum"

    )


    results = [

        {
            "approved": True,
            "signal": signal1
        },

        {
            "approved": True,
            "signal": signal2
        }

    ]


    ranker = OpportunityRanker()


    ranked = ranker.rank(results)


    assert ranked[0]["ticker"] == "NVDA"