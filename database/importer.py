import sqlite3
from pathlib import Path

import pandas as pd

from config.assets import STOCKS, ETFS

DATABASE_PATH = "database/market.db"
DATA_FOLDER = Path("data/raw")


def import_csv_to_database(ticker):
    """
    Importa un file CSV nel database SQLite.
    Se il ticker è già presente, i vecchi dati vengono sostituiti.
    """

    csv_file = DATA_FOLDER / f"{ticker}.csv"

    if not csv_file.exists():
        print(f"File non trovato: {csv_file}")
        return

    print(f"Importazione {ticker}...")

    # Legge il CSV
    data = pd.read_csv(csv_file)

    # Alcune versioni di yfinance salvano colonne MultiIndex.
    # Se presenti, manteniamo solo il primo livello.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Rinomina le colonne in minuscolo
    data.columns = [str(col).lower() for col in data.columns]

    # Rimuove la colonna "adj close" se presente
    if "adj close" in data.columns:
        data = data.drop(columns=["adj close"])

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    # Elimina eventuali dati già presenti per questo ticker
    cursor.execute(
        "DELETE FROM market_data WHERE ticker = ?",
        (ticker,)
    )

    # Inserisce tutte le righe
    for _, row in data.iterrows():
        cursor.execute(
            """
            INSERT INTO market_data
            (ticker, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker,
                row["date"],
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                int(row["volume"]),
            ),
        )

    connection.commit()
    connection.close()

    print(f"{ticker} importato correttamente")


if __name__ == "__main__":

    print("Importer avviato")

    assets = STOCKS + ETFS

    for ticker in assets:
        import_csv_to_database(ticker)

    print("\nImportazione completata.")