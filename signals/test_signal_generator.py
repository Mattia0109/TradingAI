from signals.signal_generator import SignalGenerator



def test_long_signal():


    generator = SignalGenerator()


    signal = generator.generate(

        ticker="NVDA",

        price=180,

        atr=3,

        decision={

            "action": "BUY",

            "confidence": 80

        }

    )


    assert signal["direction"] == "LONG"

    assert signal["entry_price"] == 180

    assert signal["stop_loss"] == 174

    assert signal["risk_reward"] == 3



def test_short_signal():


    generator = SignalGenerator()


    signal = generator.generate(

        ticker="AAPL",

        price=200,

        atr=2,

        decision={

            "action": "SELL",

            "confidence": 70

        }

    )


    assert signal["direction"] == "SHORT"

    assert signal["stop_loss"] == 204



def test_wait_signal():


    generator = SignalGenerator()


    signal = generator.generate(

        ticker="MSFT",

        price=400,

        atr=5,

        decision={

            "action": "WAIT",

            "confidence": 30

        }

    )


    assert signal is None