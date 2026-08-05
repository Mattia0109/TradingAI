class StrategyRunner:
    """
    Esegue il ciclo strategico del bot.

    Riceve asset analizzati
    e decide quali operazioni eseguire.
    """


    def __init__(
        self,
        decision_engine,
        ranker,
        execution_engine
    ):

        self.decision_engine = decision_engine

        self.ranker = ranker

        self.execution_engine = execution_engine



    def evaluate_market(
        self,
        market_data
    ):

        opportunities = []


        for asset in market_data:


            analysis = self.decision_engine.evaluate(
                asset
            )


            if analysis["decision"] == "BUY":

                opportunities.append(

                    analysis

                )


        ranked = self.ranker.rank(

            opportunities

        )


        return ranked



    def execute_best_trade(
        self,
        opportunities
    ):


        if not opportunities:

            return None



        best = opportunities[0]


        signal = best["signal"]



        trade = self.execution_engine.execute_trade(

            ticker=signal.ticker,

            direction=signal.direction,

            capital=10000,

            entry_price=signal.entry_price,

            stop_loss=signal.stop_loss,

            take_profit=signal.take_profit

        )


        return trade