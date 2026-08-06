class SignalValidator:
    """
    Controlla se un trade signal
    rispetta i requisiti minimi.
    """

    def __init__(
        self,
        min_confidence=50,
        min_risk_reward=2
    ):

        self.min_confidence = min_confidence

        self.min_risk_reward = min_risk_reward



    def validate(
        self,
        signal
    ):

        reasons = []


        # Confidence

        if signal["confidence"] < self.min_confidence:

            reasons.append(
                "Confidence troppo bassa"
            )


        # Risk reward

        if signal["risk_reward"] < self.min_risk_reward:

            reasons.append(
                "Risk reward insufficiente"
            )


        # Prezzi

        if signal["direction"] == "LONG":

            if signal["take_profit"] <= signal["entry_price"]:

                reasons.append(
                    "Take profit non valido"
                )


            if signal["stop_loss"] >= signal["entry_price"]:

                reasons.append(
                    "Stop loss non valido"
                )


        elif signal["direction"] == "SHORT":

            if signal["take_profit"] >= signal["entry_price"]:

                reasons.append(
                    "Take profit non valido"
                )


            if signal["stop_loss"] <= signal["entry_price"]:

                reasons.append(
                    "Stop loss non valido"
                )


        approved = len(reasons) == 0



        return {

            "approved": approved,

            "signal": signal,

            "reasons": reasons

        }