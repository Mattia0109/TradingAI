from datetime import datetime


class MarketDataEngine:
    """
    Gestisce e normalizza i dati di mercato.

    In futuro sarà collegato a:
    - Yahoo Finance
    - Interactive Brokers
    - Binance
    - Forex broker
    """



    def __init__(self):

        self.cache = {}



    def get_snapshot(self, ticker, price):

        data = {

            "ticker": ticker,

            "price": price,

            "timestamp": datetime.now()

        }


        self.cache[ticker] = data


        return data



    def get_cached_data(self, ticker):

        return self.cache.get(ticker)