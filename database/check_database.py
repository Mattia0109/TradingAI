import sqlite3


DATABASE_PATH = "database/market.db"


connection = sqlite3.connect(DATABASE_PATH)

cursor = connection.cursor()


cursor.execute(
    "SELECT COUNT(*) FROM market_data"
)

count = cursor.fetchone()[0]


print(f"Numero righe nel database: {count}")


cursor.execute(
    """
    SELECT *
    FROM market_data
    LIMIT 5
    """
)


rows = cursor.fetchall()

for row in rows:
    print(row)


connection.close()