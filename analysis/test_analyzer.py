import sqlite3
import pandas as pd

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
    """

    data = pd.read_sql_query(
        query,
        connection
    )

    connection.close()

    return data



if __name__ == "__main__":

    print("Analisi AAPL")

    data = load_data("AAPL")


    data = calculate_returns(data)

    data = calculate_sma(
        data,
        period=20
    )

    data = calculate_volatility(
        data,
        period=20
    )


    result = analyze_asset(data)


    print("\nRisultato Analyzer:")
    print("-------------------")

    for key, value in result.items():
        print(f"{key}: {value}")