from datetime import datetime

from config.settings import INITIAL_CAPITAL


class BrokerSimulator:
    """
    Simulatore di broker per paper trading.

    Gestisce:
    - capitale
    - posizioni aperte
    - storico trade
    """

    def __init__(self):

        self.balance = INITIAL_CAPITAL

        self.positions = []

        self.history = []


    def open_position(
        self,
        ticker,
        direction,
        quantity,
        price
    ):

        position = {

            "ticker": ticker,

            "direction": direction,

            "quantity": quantity,

            "entry_price": price,

            "open_time": datetime.now()

        }


        self.positions.append(position)

        return position



    def close_position(
        self,
        position,
        exit_price
    ):

        if position["direction"] == "LONG":

            pnl = (
                exit_price -
                position["entry_price"]
            ) * position["quantity"]


        else:

            pnl = (
                position["entry_price"] -
                exit_price
            ) * position["quantity"]


        self.balance += pnl


        trade = {

            "ticker": position["ticker"],

            "direction": position["direction"],

            "pnl": pnl,

            "exit_price": exit_price,

            "close_time": datetime.now()

        }


        self.history.append(trade)


        self.positions.remove(position)


        return trade