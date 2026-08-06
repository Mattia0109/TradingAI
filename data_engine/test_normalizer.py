import pandas as pd

from data_engine.normalizer import DataNormalizer



def test_normalize_candle():


    candle = pd.Series({

        "Open": 100,

        "High": 105,

        "Low": 98,

        "Close": 103,

        "Volume": 1000000

    })


    normalizer = DataNormalizer()


    result = normalizer.normalize_candle(

        "AAPL",

        candle

    )


    assert result["ticker"] == "AAPL"

    assert result["close"] == 103

    assert result["volume"] == 1000000