import sqlite3
import pandas as pd

from config.assets import STOCKS, ETFS

from analysis.indicators import (
    calculate_returns,
    calculate_sma,
    calculate_volatility
)

from analysis.analyzer import analyze_asset


DATABASE_PATH = "database/market.db"


def load_data(ticker):
    connection = sqlite3.connect(DATABASE_PATH)

    query = f"""
    SELECT date, open, high, low, close, volume
    FROM market_data
    WHERE ticker = '{ticker}'
    ORDER BY date
    """

    data = pd.read_sql_query(query, connection)

    connection.close()

    return data


def analyze_portfolio():

    results = []

    assets = STOCKS + ETFS

    for ticker in assets:

        print(f"Analizzo {ticker}...")

        data = load_data(ticker)

        if data.empty:
            print(f"Nessun dato disponibile per {ticker}")
            continue

        data = calculate_returns(data)
        data = calculate_sma(data)
        data = calculate_volatility(data)

        analysis = analyze_asset(data)

        results.append({
            "ticker": ticker,
            "score": analysis["score"],
            "trend": analysis["trend"],
            "risk": analysis["risk"],
            "decision": analysis["decision"]
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    return results