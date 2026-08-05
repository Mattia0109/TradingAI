from core.models import TradeSignal


class RiskManager:
    """
    Controlla se un segnale può essere eseguito.
    """

    def __init__(
        self,
        min_confidence=60,
        min_risk_reward=1.5,
        max_leverage=2
    ):

        self.min_confidence = min_confidence
        self.min_risk_reward = min_risk_reward
        self.max_leverage = max_leverage



    def validate(self, signal: TradeSignal):

        reasons = []


        # Confidence

        if signal.confidence < self.min_confidence:

            reasons.append(
                "Confidence troppo bassa"
            )


        # Risk Reward

        if signal.risk_reward < self.min_risk_reward:

            reasons.append(
                "Risk/Reward insufficiente"
            )


        # Direzione

        if signal.direction == "HOLD":

            reasons.append(
                "Nessun segnale operativo"
            )


        approved = len(reasons) == 0


        signal.approved = approved


        return {

            "approved": approved,

            "reasons": reasons,

            "signal": signal

        }