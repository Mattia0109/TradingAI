from market.universe import MarketUniverse


class MarketScanner:
    """
    Scanner del mercato.

    Recupera gli strumenti disponibili
    dal Market Universe.
    """

    def __init__(self):

        self.universe = MarketUniverse()



    def get_tickers(self):

        return self.universe.get_all_assets()



    def scan(self):

        assets = self.get_tickers()

        return {

            "count": len(assets),

            "assets": assets

        }