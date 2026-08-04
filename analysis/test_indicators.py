import sqlite3
import pandas as pd

from analysis.indicators import (
    calculate_returns,
    calculate_sma,
    calculate_volatility,
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands
)


DATABASE_PATH = "database/market.db"


def load_asset(ticker):

    connection = sqlite3.connect(DATABASE_PATH)

    query = f"""
    SELECT date, open, high, low, close, volume
    FROM market_data
    WHERE ticker = '{ticker}'
    ORDER BY date
    """

    data = pd.read_sql_query(
        query,
        connection
    )

    connection.close()

    return data



if __name__ == "__main__":

    ticker = "AAPL"

    print(f"Caricamento dati {ticker}...")

    data = load_asset(ticker)


    print("\nDati originali:")

    print(data.head())


    # Indicatori base

    data = calculate_returns(data)

    data = calculate_sma(data)

    data = calculate_volatility(data)


    # Indicatori avanzati

    data = calculate_ema(data)

    data = calculate_rsi(data)

    data = calculate_macd(data)

    data = calculate_bollinger_bands(data)


    print("\nAnalisi indicatori completata:")

    print(
        data[
            [
                "date",
                "close",
                "Returns",
                "SMA_20",
                "EMA_12",
                "RSI",
                "MACD",
                "MACD_signal",
                "BB_upper",
                "BB_lower",
                "Volatility"
            ]
        ].tail()
    )