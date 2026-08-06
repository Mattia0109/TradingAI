import pandas as pd
import yfinance as yf


class YahooProvider:
    """
    Provider dati mercato tramite Yahoo Finance.
    """

    def get_history(
        self,
        ticker,
        period="1mo",
        interval="1d"
    ):

        data = yf.download(
            tickers=ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False
        )

        if data.empty:
            return data

        # Se yfinance restituisce un MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        return data

    def get_latest_price(
        self,
        ticker
    ):

        history = self.get_history(
            ticker,
            period="5d"
        )

        if history.empty:
            return None

        return float(history["Close"].iloc[-1])