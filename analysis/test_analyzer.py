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

from analysis.analyzer import analyze_asset


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

    print(f"Analisi {ticker}")

    data = load_asset(ticker)


    # Indicatori base

    data = calculate_returns(data)

    data = calculate_sma(data)

    data = calculate_volatility(data)


    # Indicatori avanzati

    data = calculate_ema(data)

    data = calculate_rsi(data)

    data = calculate_macd(data)

    data = calculate_bollinger_bands(data)


    result = analyze_asset(data)


    print("\nRisultato Analyzer:")
    print("-------------------")

    print(f"trend: {result['trend']}")
    print(f"risk: {result['risk']}")
    print(f"score: {result['score']}")
    print(f"decision: {result['decision']}")

    print("\nMotivazioni:")

    for reason in result["reasons"]:
        print("-", reason)