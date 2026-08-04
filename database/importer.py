print("Importer avviato")

import sqlite3
import pandas as pd
import os


DATABASE_PATH = "database/market.db"
DATA_FOLDER = "data/raw"


def import_csv_to_database(ticker):

    file_path = f"{DATA_FOLDER}/{ticker}.csv"

    if not os.path.exists(file_path):
        print(f"File non trovato: {file_path}")
        return

    print(f"Importazione {ticker}...")

    data = pd.read_csv(
        file_path,
        header=[0, 1],
        index_col=0
    )

    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

    for index, row in data.iterrows():

        cursor.execute(
            """
            INSERT INTO market_data
            (
                ticker,
                date,
                open,
                high,
                low,
                close,
                volume
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker,
                str(index),
                float(row[("Open", ticker)]),
                float(row[("High", ticker)]),
                float(row[("Low", ticker)]),
                float(row[("Close", ticker)]),
                int(row[("Volume", ticker)])
            )
        )

    connection.commit()
    connection.close()

    print(f"{ticker} importato correttamente")


if __name__ == "__main__":

    import_csv_to_database("AAPL")