import pandas as pd

from data_engine.providers.yahoo_provider import YahooProvider
from data_engine.normalizer import DataNormalizer


class MarketDataPipeline:
    """
    Pipeline completa per ottenere dati
    di mercato normalizzati.
    """

    def __init__(self):

        self.provider = YahooProvider()

        self.normalizer = DataNormalizer()


    def clean_yahoo_data(self, data):
        """
        Pulisce i dati Yahoo Finance
        per analisi tecniche e indicatori.
        """

        if data.empty:
            return None


        # Gestione MultiIndex Yahoo Finance

        if isinstance(data.columns, pd.MultiIndex):

            data.columns = [
                column[0]
                for column in data.columns
            ]


        # Rimuove Adjusted Close
        # perché utilizziamo Close

        if "Adj Close" in data.columns:

            data = data.drop(
                columns=["Adj Close"]
            )


        # Porta la data da indice a colonna

        data = data.reset_index()


        # Normalizza i nomi colonne

        data.columns = [
            str(column)
            .lower()
            .strip()
            for column in data.columns
        ]


        return data



    def get_market_data(self, ticker):
        """
        Recupera l'ultima candela disponibile
        e la normalizza per il sistema.
        """

        history = self.provider.get_history(
            ticker,
            period="5d",
            interval="1d"
        )


        if history.empty:

            return None


        # Gestione MultiIndex Yahoo

        if isinstance(history.columns, pd.MultiIndex):

            history.columns = [
                column[0]
                for column in history.columns
            ]


        # NON convertiamo i nomi in minuscolo
        # perché DataNormalizer usa:
        # Open, High, Low, Close, Volume

        latest = history.iloc[-1]


        normalized = self.normalizer.normalize_candle(
            ticker,
            latest
        )


        return normalized



    def get_historical_data(self, ticker):
        """
        Recupera storico OHLCV pulito
        per indicatori, strategie e analisi.
        """

        history = self.provider.get_history(
            ticker,
            period="6mo",
            interval="1d"
        )


        history = self.clean_yahoo_data(
            history
        )


        if history is None:

            return None



        required_columns = [
            "close",
            "high",
            "low",
            "open",
            "volume"
        ]


        for column in required_columns:

            if column not in history.columns:

                raise ValueError(
                    f"Colonna mancante: {column}. "
                    f"Ricevute: {list(history.columns)}"
                )


        return history