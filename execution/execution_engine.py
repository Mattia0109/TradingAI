from paper.broker_simulator import BrokerSimulator
from position.position_manager import PositionManager


class ExecutionEngine:
    """
    Gestisce l'esecuzione degli ordini.

    Collega:
    - Position Manager
    - Paper Broker
    """


    def __init__(self):

        self.broker = BrokerSimulator()

        self.position_manager = PositionManager()



    def execute_trade(
        self,
        ticker,
        direction,
        capital,
        entry_price,
        stop_loss,
        take_profit
    ):


        plan = self.position_manager.create_position_plan(

            ticker=ticker,

            direction=direction,

            capital=capital,

            entry_price=entry_price,

            stop_loss=stop_loss,

            take_profit=take_profit

        )


        position = self.broker.open_position(

            ticker=plan["ticker"],

            direction=plan["direction"],

            quantity=plan["quantity"],

            price=plan["entry_price"]

        )


        return {

            "position": position,

            "plan": plan

        }