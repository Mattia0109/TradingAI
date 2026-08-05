from datetime import datetime

from core.models import MarketSnapshot
from core.decision_engine import DecisionEngine



def test_decision_engine_long():


    snapshot = MarketSnapshot(

        ticker="NVDA",

        timeframe="5m",

        timestamp=datetime.now(),

        price=180,

        volume=50000000,

        rsi=55,

        macd=3,

        macd_signal=1,

        sma_20=170,

        volatility=0.20
    )


    engine = DecisionEngine()


    signal = engine.analyze(snapshot)


    print(signal)


    assert signal.direction == "LONG"

    assert signal.confidence > 50

    assert signal.take_profit > signal.entry_price