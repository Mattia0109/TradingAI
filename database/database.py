import sqlite3
import os


DATABASE_PATH = "database/market.db"


def create_database():

    os.makedirs("database", exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS market_data (

        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT,
        date TEXT,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume INTEGER

    )
    """)

    connection.commit()
    connection.close()

    print("Database creato correttamente")


if __name__ == "__main__":
    create_database()