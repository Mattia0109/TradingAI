from scanner.market_scanner import MarketScanner
from ranking.opportunity_ranker import OpportunityRanker
from engine.trading_pipeline import TradingPipeline



class TradingLoop:
    """
    Ciclo principale del trading bot.

    Coordina:
    - Market Scanner
    - Trading Pipeline
    - Opportunity Ranking
    """



    def __init__(self):

        self.scanner = MarketScanner()

        self.pipeline = TradingPipeline()

        self.ranker = OpportunityRanker()



    def run_once(self):

        print("=" * 50)
        print("TRADING AI AUTONOMOUS SCAN")
        print("=" * 50)



        print("\nScansione mercato...")



        assets = self.scanner.get_tickers()



        opportunities = []



        for ticker in assets:

            result = self.pipeline.generate_signal(
                ticker
            )


            if result is not None:

                opportunities.append(
                    result
                )



        print(
            f"Segnali trovati: {len(opportunities)}"
        )



        ranked = self.ranker.rank(
            opportunities
        )



        print("\nMigliori opportunità:")



        for index, trade in enumerate(

            ranked[:5],

            start=1

        ):

            signal = trade["signal"]


            print(
                f"""
{index}) {trade["ticker"]}
Direction: {signal["direction"]}
Confidence: {signal["confidence"]}
Score: {trade["score"]}
"""
            )



        return ranked



    def start(self):

        return self.run_once()