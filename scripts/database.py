import sqlite3

DB_NAME = "results/radiologist_credentials.db"


def get_connection():
    return sqlite3.connect(DB_NAME)


def create_users_table():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            radiologist_id INTEGER PRIMARY KEY AUTOINCREMENT,
            radiologist_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password BLOB NOT NULL
        )
    """)

    conn.commit()
    conn.close()
