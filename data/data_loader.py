import yfinance as yf
from pathlib import Path

from config.assets import STOCKS, ETFS


RAW_DATA_PATH = Path("data/raw")


def download_data(ticker):

    print(f"Scaricamento dati per {ticker}...")

    data = yf.download(
        ticker,
        period="1y",
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        print(f"Nessun dato trovato per {ticker}")
        return

    # Se yfinance crea colonne multiple (Ticker)
    if hasattr(data.columns, "levels"):
        data.columns = data.columns.get_level_values(0)

    # Trasforma la data da indice a colonna
    data = data.reset_index()

    # Rinomina colonne
    data.columns = [
        str(column).lower().replace(" ", "_")
        for column in data.columns
    ]

    # Manteniamo solo i dati utili
    columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    data = data[columns]

    # Cartella destinazione
    RAW_DATA_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    file_path = RAW_DATA_PATH / f"{ticker}.csv"

    data.to_csv(
        file_path,
        index=False
    )

    print(f"Dati salvati in {file_path}")



if __name__ == "__main__":

    assets = STOCKS + ETFS

    for ticker in assets:
        download_data(ticker)

    print("Download completato")