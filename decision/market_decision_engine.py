class MarketDecisionEngine:
    """
    Trasforma l'analisi degli indicatori
    in una decisione di trading.
    """



    def decide(
        self,
        analysis
    ):

        score = analysis["score"]

        risk = analysis["risk"]

        reasons = analysis["signals"]



        confidence = abs(score)



        # rischio elevato riduce fiducia

        if risk == "HIGH":

            confidence -= 20

            reasons.append(
                "Rischio elevato"
            )



        if confidence < 0:

            confidence = 0



        # decisione

        if score >= 40:

            action = "BUY"


        elif score <= -40:

            action = "SELL"


        else:

            action = "WAIT"



        return {

            "action": action,

            "confidence": confidence,

            "risk": risk,

            "reasons": reasons

        }