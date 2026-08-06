from indicator_engine.analyzer import IndicatorAnalyzer



def test_bullish_analysis():


    analyzer = IndicatorAnalyzer()


    result = analyzer.analyze({

        "EMA20": 210,

        "EMA50": 200,

        "MACD": 3,

        "RSI": 55,

        "ATR": 2,

        "price": 200

    })


    assert result["trend"] == "BULLISH"

    assert result["score"] > 0



def test_bearish_analysis():


    analyzer = IndicatorAnalyzer()


    result = analyzer.analyze({

        "EMA20": 190,

        "EMA50": 200,

        "MACD": -3,

        "RSI": 50,

        "ATR": 2,

        "price": 200

    })


    assert result["trend"] == "BEARISH"

    assert result["score"] < 0