import sqlite3
import pandas as pd

from analysis.indicators import (
    calculate_returns,
    calculate_sma,
    calculate_volatility
)


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

    print("Caricamento dati AAPL...")

    data = load_data("AAPL")

    print("\nDati originali:")
    print(data.head())


    data = calculate_returns(data)

    data = calculate_sma(
        data,
        period=20
    )

    data = calculate_volatility(
        data,
        period=20
    )


    print("\nAnalisi completata:")
    print(data.tail())