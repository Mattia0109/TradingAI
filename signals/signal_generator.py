from datetime import datetime


class SignalGenerator:
    """
    Genera segnali operativi
    partendo dalla decisione del mercato.
    """

    def generate(
        self,
        ticker,
        price,
        atr,
        decision
    ):

        action = decision["action"]

        confidence = decision["confidence"]


        if action == "BUY":

            direction = "LONG"

            entry = price

            stop_loss = price - (atr * 2)

            take_profit = price + (
                (price - stop_loss) * 3
            )


        elif action == "SELL":

            direction = "SHORT"

            entry = price

            stop_loss = price + (atr * 2)

            take_profit = price - (
                (stop_loss - price) * 3
            )


        else:

            return None



        risk = abs(
            entry - stop_loss
        )


        reward = abs(
            take_profit - entry
        )


        risk_reward = reward / risk


        return {

            "ticker": ticker,

            "timestamp": datetime.now(),

            "direction": direction,

            "confidence": confidence,

            "entry_price": entry,

            "stop_loss": stop_loss,

            "take_profit": take_profit,

            "risk_reward": round(
                risk_reward,
                2
            ),

            "strategy": "momentum"

        }