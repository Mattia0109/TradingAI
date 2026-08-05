from datetime import datetime

from core.models import TradeSignal

from core.risk_manager import RiskManager



def test_risk_manager_accepts_good_trade():


    signal = TradeSignal(

        ticker="NVDA",

        timestamp=datetime.now(),

        direction="LONG",

        confidence=85,

        entry_price=180,

        stop_loss=178,

        take_profit=185,

        risk_reward=2.5,

        timeframe="5m",

        strategy="momentum"

    )


    manager = RiskManager()


    result = manager.validate(signal)


    assert result["approved"] == True



def test_risk_manager_rejects_bad_trade():


    signal = TradeSignal(

        ticker="TSLA",

        timestamp=datetime.now(),

        direction="LONG",

        confidence=40,

        entry_price=300,

        stop_loss=299,

        take_profit=301,

        risk_reward=1,

        timeframe="5m",

        strategy="test"

    )


    manager = RiskManager()


    result = manager.validate(signal)


    assert result["approved"] == False