import pandas as pd


def calculate_returns(data):
    """
    Calcola il rendimento percentuale giornaliero
    """

    data["Returns"] = data["close"].pct_change()

    return data



def calculate_sma(data, period=20):
    """
    Calcola la Simple Moving Average
    """

    data[f"SMA_{period}"] = (
        data["close"]
        .rolling(window=period)
        .mean()
    )

    return data



def calculate_volatility(data, period=20):
    """
    Calcola la volatilità annualizzata
    """

    data["Volatility"] = (
        data["Returns"]
        .rolling(window=period)
        .std()
        *
        (252 ** 0.5)
    )

    return data