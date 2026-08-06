class IndicatorAnalyzer:
    """
    Interpreta gli indicatori tecnici
    e crea una valutazione del mercato.
    """



    def analyze(
        self,
        indicators
    ):

        signals = []

        score = 0



        # TREND EMA

        if indicators["EMA20"] > indicators["EMA50"]:

            score += 25

            signals.append(
                "EMA20 sopra EMA50: trend positivo"
            )

        else:

            score -= 25

            signals.append(
                "EMA20 sotto EMA50: trend negativo"
            )



        # MACD

        if indicators["MACD"] > 0:

            score += 25

            signals.append(
                "MACD positivo"
            )

        else:

            score -= 25

            signals.append(
                "MACD negativo"
            )



        # RSI

        rsi = indicators["RSI"]


        if rsi < 30:

            score += 15

            signals.append(
                "RSI ipervenduto"
            )


        elif rsi > 70:

            score -= 10

            signals.append(
                "RSI elevato"
            )


        else:

            score += 5

            signals.append(
                "RSI neutrale"
            )



        # VOLATILITA'

        atr = indicators["ATR"]


        if atr > indicators["price"] * 0.03:

            risk = "HIGH"

        else:

            risk = "NORMAL"



        # CLASSIFICAZIONE

        if score >= 40:

            trend = "BULLISH"

        elif score <= -20:

            trend = "BEARISH"

        else:

            trend = "NEUTRAL"



        return {

            "score": score,

            "trend": trend,

            "risk": risk,

            "signals": signals

        }