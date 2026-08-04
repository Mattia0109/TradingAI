import pandas as pd


def calculate_returns(data):
    """
    Calcola il rendimento giornaliero
    """

    data["Returns"] = data["close"].pct_change()

    return data



def calculate_sma(data, period=20):
    """
    Calcola la media mobile semplice
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



def calculate_ema(data, period=12):
    """
    Calcola la media mobile esponenziale
    """

    data[f"EMA_{period}"] = (
        data["close"]
        .ewm(span=period, adjust=False)
        .mean()
    )

    return data



def calculate_rsi(data, period=14):
    """
    Calcola RSI
    """

    delta = data["close"].diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)


    avg_gain = (
        gain
        .rolling(window=period)
        .mean()
    )

    avg_loss = (
        loss
        .rolling(window=period)
        .mean()
    )


    rs = avg_gain / avg_loss

    data["RSI"] = 100 - (
        100 / (1 + rs)
    )

    return data



def calculate_macd(data):
    """
    Calcola MACD e signal line
    """

    ema12 = (
        data["close"]
        .ewm(
            span=12,
            adjust=False
        )
        .mean()
    )

    ema26 = (
        data["close"]
        .ewm(
            span=26,
            adjust=False
        )
        .mean()
    )


    data["MACD"] = ema12 - ema26


    data["MACD_signal"] = (
        data["MACD"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    return data



def calculate_bollinger_bands(data, period=20):
    """
    Calcola Bollinger Bands
    """

    middle = (
        data["close"]
        .rolling(window=period)
        .mean()
    )

    std = (
        data["close"]
        .rolling(window=period)
        .std()
    )


    data["BB_middle"] = middle

    data["BB_upper"] = (
        middle + (std * 2)
    )

    data["BB_lower"] = (
        middle - (std * 2)
    )


    return data