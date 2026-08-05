from datetime import datetime
import sqlite3

import pandas as pd

from core.models import MarketSnapshot
from core.decision_engine import DecisionEngine
from core.risk_manager import RiskManager


class MarketScanner:
    """
    Scanner del mercato.

    Recupera gli asset dal database,
    crea snapshot e genera segnali.
    """

    def __init__(self, database_path="database/market.db"):

        self.database_path = database_path

        self.decision_engine = DecisionEngine()

        self.risk_manager = RiskManager()


    def get_tickers(self):

        connection = sqlite3.connect(
            self.database_path
        )

        query = """
        SELECT DISTINCT ticker
        FROM market_data
        """

        data = pd.read_sql_query(
            query,
            connection
        )

        connection.close()

        return data["ticker"].tolist()


    def get_latest_data(self, ticker):

        connection = sqlite3.connect(
            self.database_path
        )

        query = """
        SELECT *
        FROM market_data
        WHERE ticker = ?
        ORDER BY date DESC
        LIMIT 30
        """

        data = pd.read_sql_query(
            query,
            connection,
            params=(ticker,)
        )

        connection.close()

        return data.sort_values(
            "date"
        )


    def create_snapshot(
        self,
        ticker,
        data
    ):

        latest = data.iloc[-1]


        sma20 = (
            data["close"]
            .rolling(20)
            .mean()
            .iloc[-1]
        )


        return MarketSnapshot(

            ticker=ticker,

            timeframe="1D",

            timestamp=datetime.now(),

            price=float(
                latest["close"]
            ),

            volume=float(
                latest["volume"]
            ),

            sma_20=float(
                sma20
            )

        )


    def scan(self):

        results = []


        tickers = self.get_tickers()


        for ticker in tickers:


            data = self.get_latest_data(
                ticker
            )


            if len(data) < 20:
                continue


            snapshot = self.create_snapshot(
                ticker,
                data
            )


            signal = self.decision_engine.analyze(
                snapshot
            )


            validation = self.risk_manager.validate(
                signal
            )


            results.append(
                validation
            )


        return results