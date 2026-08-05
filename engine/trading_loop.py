from scanner.market_scanner import MarketScanner
from ranking.opportunity_ranker import OpportunityRanker


class TradingLoop:
    """
    Ciclo principale del trading bot.

    Coordina:
    - Market Scanner
    - Opportunity Ranking
    """

    def __init__(self):

        self.scanner = MarketScanner()

        self.ranker = OpportunityRanker()


    def run_once(self):

        print("=" * 50)
        print("TRADING AI AUTONOMOUS SCAN")
        print("=" * 50)


        print("\nScansione mercato...")


        signals = self.scanner.scan()


        print(
            f"Segnali trovati: {len(signals)}"
        )


        ranked = self.ranker.rank(
            signals
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
Direction: {signal.direction}
Confidence: {signal.confidence}
Score: {trade["score"]}
"""
            )


        return ranked


    def start(self):

        return self.run_once()