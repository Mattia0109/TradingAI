from validator.signal_validator import SignalValidator



def valid_long():

    return {

        "ticker": "NVDA",

        "direction": "LONG",

        "confidence": 85,

        "entry_price": 180,

        "stop_loss": 174,

        "take_profit": 198,

        "risk_reward": 3

    }



def test_valid_signal():


    validator = SignalValidator()


    result = validator.validate(
        valid_long()
    )


    assert result["approved"] is True



def test_low_confidence():


    validator = SignalValidator()


    signal = valid_long()

    signal["confidence"] = 30


    result = validator.validate(
        signal
    )


    assert result["approved"] is False



def test_bad_risk_reward():


    validator = SignalValidator()


    signal = valid_long()

    signal["risk_reward"] = 1


    result = validator.validate(
        signal
    )


    assert result["approved"] is False