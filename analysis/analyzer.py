import pandas as pd


def analyze_asset(data):

    latest = data.iloc[-1]

    score = 50

    reasons = []


    # =====================
    # TREND
    # =====================

    if latest["close"] > latest["SMA_20"]:

        score += 15
        reasons.append("Trend positivo")

    else:

        score -= 15
        reasons.append("Trend negativo")


    # =====================
    # RSI MOMENTUM
    # =====================

    rsi = latest["RSI"]

    if rsi < 30:

        score += 10
        reasons.append("RSI ipervenduto")

    elif rsi > 70:

        score -= 10
        reasons.append("RSI ipercomprato")

    else:

        score += 5
        reasons.append("RSI neutrale")


    # =====================
    # MACD
    # =====================

    if latest["MACD"] > latest["MACD_signal"]:

        score += 15
        reasons.append("MACD positivo")

    else:

        score -= 10
        reasons.append("MACD negativo")


    # =====================
    # BOLLINGER
    # =====================

    if latest["close"] < latest["BB_lower"]:

        score += 10
        reasons.append("Prezzo sotto banda inferiore")

    elif latest["close"] > latest["BB_upper"]:

        score -= 10
        reasons.append("Prezzo sopra banda superiore")

    else:

        score += 5
        reasons.append("Prezzo nel range")


    # =====================
    # RISCHIO
    # =====================

    volatility = latest["Volatility"]


    if volatility < 0.20:

        score += 10
        risk = "BASSO"

    elif volatility < 0.35:

        score += 5
        risk = "MEDIO"

    else:

        score -= 10
        risk = "ALTO"


    # Limiti score

    score = max(0, min(100, score))


    # Decisione

    if score >= 75:

        decision = "BUY"

    elif score >= 55:

        decision = "WATCHLIST BUY"

    elif score >= 40:

        decision = "WAIT"

    else:

        decision = "AVOID"


    if latest["close"] > latest["SMA_20"]:

        trend = "POSITIVO"

    else:

        trend = "NEGATIVO"


    return {

        "trend": trend,
        "risk": risk,
        "score": score,
        "decision": decision,
        "reasons": reasons

    }