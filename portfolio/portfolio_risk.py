MAX_TOTAL_EXPOSURE = 0.50
MAX_OPEN_POSITIONS = 5



class PortfolioRiskManager:
    """
    Gestisce il rischio complessivo del portafoglio.
    """

    def __init__(
        self,
        max_exposure=MAX_TOTAL_EXPOSURE,
        max_positions=MAX_OPEN_POSITIONS
    ):

        self.max_exposure = max_exposure

        self.max_positions = max_positions



    def calculate_exposure(
        self,
        capital,
        positions
    ):

        invested = sum(
            position["value"]
            for position in positions
        )

        return invested / capital



    def can_open_position(
        self,
        capital,
        positions,
        new_position_value
    ):

        current_exposure = self.calculate_exposure(
            capital,
            positions
        )


        new_exposure = (
            current_exposure +
            new_position_value / capital
        )


        if len(positions) >= self.max_positions:

            return {

                "approved": False,

                "reason": "Maximum open positions reached"

            }



        if new_exposure > self.max_exposure:

            return {

                "approved": False,

                "reason": "Maximum portfolio exposure reached"

            }



        return {

            "approved": True,

            "reason": "Risk limits respected"

        }