class MarketUniverse:
    """
    Contiene tutti gli strumenti disponibili
    per il trading automatico.
    """


    def __init__(self):

        self.stocks = [

            "AAPL",
            "MSFT",
            "NVDA",
            "AMZN",
            "GOOGL",
            "META",
            "TSLA"

        ]


        self.etfs = [

            "SPY",
            "QQQ",
            "VTI"

        ]


        self.crypto = [

            "BTC-USD",
            "ETH-USD"

        ]


        self.forex = [

            "EURUSD",
            "GBPUSD",
            "USDJPY"

        ]


        self.commodities = [

            "GC=F",   # Gold
            "CL=F"    # Crude Oil

        ]



    def get_all_assets(self):

        return (

            self.stocks
            +
            self.etfs
            +
            self.crypto
            +
            self.forex
            +
            self.commodities

        )