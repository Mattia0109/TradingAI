def analyze_asset(data):
    """
    Analizza un asset usando indicatori tecnici
    """

    latest = data.iloc[-1]

    score = 50

    analysis = {}


    # Analisi trend tramite SMA20
    if latest["close"] > latest["SMA_20"]:
        trend = "POSITIVO"
        score += 15
    else:
        trend = "NEGATIVO"
        score -= 15


    # Analisi volatilità
    volatility = latest["Volatility"]

    if volatility < 0.20:
        risk = "BASSO"
        score += 10

    elif volatility < 0.40:
        risk = "MEDIO"

    else:
        risk = "ALTO"
        score -= 10


    # Limitiamo lo score
    score = max(0, min(100, score))


    # Decisione finale
    if score >= 70:
        decision = "WATCHLIST BUY"

    elif score <= 40:
        decision = "AVOID"

    else:
        decision = "WAIT"


    analysis["trend"] = trend
    analysis["risk"] = risk
    analysis["score"] = score
    analysis["decision"] = decision


    return analysis