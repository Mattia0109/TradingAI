from decision.market_decision_engine import MarketDecisionEngine



def test_buy_decision():


    engine = MarketDecisionEngine()


    result = engine.decide({

        "score": 60,

        "risk": "NORMAL",

        "signals": [

            "Trend positivo",

            "MACD positivo"

        ]

    })


    assert result["action"] == "BUY"

    assert result["confidence"] == 60



def test_wait_decision():


    engine = MarketDecisionEngine()


    result = engine.decide({

        "score": 10,

        "risk": "NORMAL",

        "signals": []

    })


    assert result["action"] == "WAIT"



def test_high_risk_penalty():


    engine = MarketDecisionEngine()


    result = engine.decide({

        "score": 60,

        "risk": "HIGH",

        "signals": []

    })


    assert result["confidence"] == 40