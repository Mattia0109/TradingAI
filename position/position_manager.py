from config.settings import MAX_RISK_PER_TRADE


class PositionManager:
    """
    Gestisce dimensionamento posizione
    e livelli di rischio.
    """

    def __init__(self):

        self.max_risk = MAX_RISK_PER_TRADE



    def calculate_position_size(
        self,
        capital,
        entry_price,
        stop_loss
    ):

        """
        Calcola quantità acquistabile
        rispettando il rischio massimo.
        """

        risk_amount = capital * self.max_risk


        risk_per_share = abs(
            entry_price - stop_loss
        )


        if risk_per_share == 0:

            return 0


        quantity = int(
            risk_amount / risk_per_share
        )


        return quantity



    def create_position_plan(
        self,
        ticker,
        direction,
        capital,
        entry_price,
        stop_loss,
        take_profit
    ):


        quantity = self.calculate_position_size(

            capital,

            entry_price,

            stop_loss

        )


        return {

            "ticker": ticker,

            "direction": direction,

            "quantity": quantity,

            "entry_price": entry_price,

            "stop_loss": stop_loss,

            "take_profit": take_profit

        }