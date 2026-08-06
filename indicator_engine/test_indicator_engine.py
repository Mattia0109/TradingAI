import pandas as pd

from indicator_engine.indicator_engine import IndicatorEngine



def create_test_data():

    return pd.DataFrame({

        "Open": [
            100,101,102,103,104,
            105,106,107,108,109,
            110,111,112,113,114,
            115,116,117,118,119,
            120,121,122,123,124,
            125,126,127,128,129
        ],

        "High": [
            101,102,103,104,105,
            106,107,108,109,110,
            111,112,113,114,115,
            116,117,118,119,120,
            121,122,123,124,125,
            126,127,128,129,130
        ],

        "Low": [
            99,100,101,102,103,
            104,105,106,107,108,
            109,110,111,112,113,
            114,115,116,117,118,
            119,120,121,122,123,
            124,125,126,127,128
        ],

        "Close": [
            100,101,102,103,104,
            105,106,107,108,109,
            110,111,112,113,114,
            115,116,117,118,119,
            120,121,122,123,124,
            125,126,127,128,129
        ],

        "Volume": [
            1000000
        ] * 30

    })



def test_ema():

    engine = IndicatorEngine()

    df = create_test_data()


    result = engine.ema(
        df["Close"],
        10
    )


    assert len(result) == 30

    assert result.iloc[-1] > result.iloc[0]



def test_rsi():

    engine = IndicatorEngine()

    df = create_test_data()


    result = engine.rsi(
        df["Close"]
    )


    assert len(result) == 30



def test_macd():

    engine = IndicatorEngine()

    df = create_test_data()


    result = engine.macd(
        df["Close"]
    )


    assert len(result) == 30



def test_atr():

    engine = IndicatorEngine()

    df = create_test_data()


    result = engine.atr(
        df
    )


    assert len(result) == 30