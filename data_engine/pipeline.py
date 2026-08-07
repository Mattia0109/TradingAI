import pandas as pd

from data_engine.providers.yahoo_provider import YahooProvider
from data_engine.normalizer import DataNormalizer


class MarketDataPipeline:
    """
    Recupera, pulisce e normalizza
    dati di mercato Yahoo Finance.
    """

    def __init__(self):

        self.provider = YahooProvider()

        self.normalizer = DataNormalizer()


    def clean_yahoo_data(self, data):
        """
        Converte i dati Yahoo in un DataFrame
        standard con colonne minuscole.
        """

        if data is None or data.empty:
            return None

        cleaned = data.copy()

        if isinstance(
            cleaned.columns,
            pd.MultiIndex
        ):
            cleaned.columns = [
                column[0]
                for column in cleaned.columns
            ]

        if "Adj Close" in cleaned.columns:
            cleaned = cleaned.drop(
                columns=["Adj Close"]
            )

        cleaned = cleaned.reset_index()

        cleaned.columns = [
            str(column)
            .lower()
            .strip()
            for column in cleaned.columns
        ]

        if (
            "datetime" in cleaned.columns
            and "date" not in cleaned.columns
        ):
            cleaned = cleaned.rename(
                columns={
                    "datetime": "date"
                }
            )

        required_columns = [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in cleaned.columns
        ]

        if missing_columns:
            raise ValueError(
                "Colonne Yahoo mancanti: "
                f"{missing_columns}. "
                f"Ricevute: {list(cleaned.columns)}"
            )

        cleaned = cleaned[
            required_columns
        ].copy()

        cleaned["date"] = pd.to_datetime(
            cleaned["date"],
            errors="coerce"
        )

        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]

        for column in numeric_columns:
            cleaned[column] = pd.to_numeric(
                cleaned[column],
                errors="coerce"
            )

        cleaned = cleaned.dropna(
            subset=[
                "date",
                "open",
                "high",
                "low",
                "close"
            ]
        )

        cleaned = cleaned.sort_values(
            "date"
        ).reset_index(
            drop=True
        )

        return cleaned


    def get_market_data(self, ticker):
        """
        Restituisce l'ultima candela normalizzata.
        """

        history = self.provider.get_history(
            ticker=ticker,
            period="5d",
            interval="1d"
        )

        if history is None or history.empty:
            return None

        if isinstance(
            history.columns,
            pd.MultiIndex
        ):
            history.columns = [
                column[0]
                for column in history.columns
            ]

        latest = history.iloc[-1]

        return self.normalizer.normalize_candle(
            ticker,
            latest
        )


    def get_historical_data(
        self,
        ticker,
        period="6mo",
        interval="1d"
    ):
        """
        Restituisce storico OHLCV configurabile.

        Esempi:
        - period='6mo', interval='1d'
        - period='60d', interval='5m'
        - period='30d', interval='15m'
        """

        history = self.provider.get_history(
            ticker=ticker,
            period=period,
            interval=interval
        )

        return self.clean_yahoo_data(
            history
        )