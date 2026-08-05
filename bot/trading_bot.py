from scanner.market_scanner import MarketScanner
from execution.execution_engine import ExecutionEngine
from bot.strategy_runner import StrategyRunner



class TradingBot:
    """
    Controller principale del trading bot.
    """


    def __init__(
        self,
        decision_engine,
        ranker
    ):


        self.scanner = MarketScanner()

        self.execution = ExecutionEngine()


        self.strategy_runner = StrategyRunner(

            decision_engine,

            ranker,

            self.execution

        )


        self.running = False



    def start(self):

        self.running = True



    def stop(self):

        self.running = False



    def run_cycle(self):


        assets = self.scanner.get_tickers()


        opportunities = self.strategy_runner.evaluate_market(

            assets

        )


        trade = self.strategy_runner.execute_best_trade(

            opportunities

        )


        return trade