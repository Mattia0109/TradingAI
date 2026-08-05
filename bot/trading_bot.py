from scanner.market_scanner import MarketScanner
from execution.execution_engine import ExecutionEngine


class TradingBot:
    """
    Controller principale del sistema trading.

    Coordina:
    - scanner mercato
    - analisi
    - decisione
    - esecuzione
    """


    def __init__(self):

        self.scanner = MarketScanner()

        self.execution = ExecutionEngine()

        self.running = False



    def start(self):

        self.running = True


    def stop(self):

        self.running = False



    def scan_market(self):

        return self.scanner.get_tickers()