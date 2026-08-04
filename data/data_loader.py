import yfinance as yf
import os

from config.assets import STOCKS, ETFS


def download_stock_data(ticker, period="1y"):
    print(f"Scaricamento dati per {ticker}...")

    data = yf.download(
        ticker,
        period=period,
        auto_adjust=False
    )

    return data


def save_data(data, ticker):
    folder = "data/raw"

    os.makedirs(folder, exist_ok=True)

    file_path = f"{folder}/{ticker}.csv"

    data.to_csv(file_path)

    print(f"Dati salvati in {file_path}")


def download_all_assets():

    assets = STOCKS + ETFS

    for ticker in assets:

        data = download_stock_data(ticker)

        save_data(data, ticker)


if __name__ == "__main__":

    download_all_assets()

    print("Download completato")