from datetime import datetime

from core.models import MarketSnapshot, TradeSignal


class DecisionEngine:
    """
    Primo motore decisionale del bot.

    Trasforma i dati di mercato in un segnale operativo.
    """

    def analyze(self, market: MarketSnapshot) -> TradeSignal:

        score = 0
        reasons = []


        # Analisi trend tramite SMA
        if market.sma_20 is not None:

            if market.price > market.sma_20:
                score += 20
                reasons.append("Prezzo sopra SMA20")

            else:
                score -= 20
                reasons.append("Prezzo sotto SMA20")


        # Analisi RSI
        if market.rsi is not None:

            if 40 <= market.rsi <= 65:
                score += 15
                reasons.append("RSI favorevole")

            elif market.rsi > 70:
                score -= 10
                reasons.append("RSI ipercomprato")

            elif market.rsi < 30:
                score += 5
                reasons.append("RSI ipervenduto")


        # Analisi MACD
        if (
            market.macd is not None
            and market.macd_signal is not None
        ):

            if market.macd > market.macd_signal:
                score += 20
                reasons.append("MACD positivo")

            else:
                score -= 20
                reasons.append("MACD negativo")


        # Volatilità
        if market.volatility is not None:

            if market.volatility < 0.30:
                score += 10
                reasons.append("Volatilità controllata")

            else:
                score -= 10
                reasons.append("Volatilità alta")


        # Decisione

        if score >= 50:

            direction = "LONG"

        elif score <= -50:

            direction = "SHORT"

        else:

            direction = "HOLD"


        confidence = min(abs(score), 95)


        entry = market.price


        if direction == "LONG":

            stop_loss = entry * 0.99
            take_profit = entry * 1.02


        elif direction == "SHORT":

            stop_loss = entry * 1.01
            take_profit = entry * 0.98


        else:

            stop_loss = entry
            take_profit = entry


        risk = abs(entry - stop_loss)

        reward = abs(take_profit - entry)


        risk_reward = (
            reward / risk
            if risk > 0
            else 0
        )


        return TradeSignal(

            ticker=market.ticker,

            timestamp=datetime.now(),

            direction=direction,

            confidence=confidence,

            entry_price=entry,

            stop_loss=stop_loss,

            take_profit=take_profit,

            risk_reward=risk_reward,

            timeframe=market.timeframe,

            strategy="indicator_based",

            reasons=reasons

        )