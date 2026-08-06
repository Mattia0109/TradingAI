import sqlite3
import pandas as pd

from config.assets import STOCKS, ETFS

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


# Portfolio Risk Settings

MAX_POSITION_SIZE = 0.10



def calculate_position_size(
    capital,
    signal
):
    """
    Calcola la dimensione della posizione
    in base alla percentuale massima
    del capitale.
    """

    confidence = signal["confidence"]

    risk_reward = signal["risk_reward"]

    entry_price = signal["entry_price"]


    # Blocca trade con rischio/rendimento insufficiente

    if risk_reward < 2:

        return {

            "allocation_percent": 0,

            "position_value": 0,

            "shares": 0

        }



    # Allocazione dinamica

    if confidence < 50:

        allocation = 0


    elif confidence < 65:

        allocation = 0.03


    elif confidence < 80:

        allocation = 0.05


    elif confidence < 90:

        allocation = 0.08


    else:

        allocation = 0.10



    # Limite massimo assoluto

    allocation = min(
        allocation,
        MAX_POSITION_SIZE
    )



    position_value = (
        capital *
        allocation
    )


    shares = int(
        position_value /
        entry_price
    )



    return {

        "allocation_percent": allocation,

        "position_value": position_value,

        "shares": shares

    }



def load_asset(ticker):

    connection = sqlite3.connect(
        DATABASE_PATH
    )

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



def prepare_data(data):

    data = calculate_returns(data)

    data = calculate_sma(data)

    data = calculate_volatility(data)

    data = calculate_ema(data)

    data = calculate_rsi(data)

    data = calculate_macd(data)

    data = calculate_bollinger_bands(data)

    return data



def analyze_portfolio():

    assets = STOCKS + ETFS

    results = []


    for ticker in assets:

        print(f"Analizzo {ticker}...")


        data = load_asset(ticker)


        if data.empty:

            print(
                f"Nessun dato disponibile per {ticker}"
            )

            continue


        data = prepare_data(data)


        result = analyze_asset(data)


        results.append(
            {
                "ticker": ticker,
                **result
            }
        )


    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )


    return results



if __name__ == "__main__":


    print(
        "\nTRADING AI - PORTFOLIO ANALYSIS"
    )

    print("=" * 60)


    portfolio = analyze_portfolio()


    print("\n")


    for index, asset in enumerate(
        portfolio,
        start=1
    ):

        print(
            f"{index:2}. "
            f"{asset['ticker']:5} | "
            f"Score: {asset['score']:3} | "
            f"Trend: {asset['trend']:9} | "
            f"Risk: {asset['risk']:6} | "
            f"Decision: {asset['decision']}"
        )