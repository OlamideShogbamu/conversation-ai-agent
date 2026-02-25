import sqlite3
from pathlib import Path
from config.settings import BASE_DIR

DB_PATH = BASE_DIR / "data" / "globus.db"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database schema."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript(
        """
        CREATE TABLE IF NOT EXISTS customer (
            id INTEGER PRIMARY KEY,
            account_no TEXT UNIQUE NOT NULL,
            account_name TEXT NOT NULL,
            currency TEXT DEFAULT 'NGN',
            account_type TEXT,
            product_type TEXT,
            product_description TEXT,
            current_balance REAL DEFAULT 0,
            account_open_date TEXT
        );

        CREATE TABLE IF NOT EXISTS transaction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_no TEXT NOT NULL,
            transaction_date TEXT,
            transaction_type TEXT,
            transaction_amount REAL,
            destination_account TEXT,
            narration TEXT,
            destination_bank TEXT,
            transaction_status TEXT,
            failure_reason TEXT,
            FOREIGN KEY (account_no) REFERENCES customer(account_no)
        );

        CREATE TABLE IF NOT EXISTS card (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_no TEXT NOT NULL,
            card_issuer TEXT,
            card_type TEXT,
            card_last_four TEXT,
            card_activation_date TEXT,
            status TEXT DEFAULT 'Active',
            FOREIGN KEY (account_no) REFERENCES customer(account_no)
        );

        CREATE INDEX IF NOT EXISTS idx_customer_account_no ON customer(account_no);
        CREATE INDEX IF NOT EXISTS idx_transaction_account_no ON transaction_history(account_no);
        CREATE INDEX IF NOT EXISTS idx_card_account_no ON card(account_no);
        CREATE INDEX IF NOT EXISTS idx_transaction_date ON transaction_history(transaction_date);
    """
    )

    conn.commit()
    conn.close()


def reset_db():
    """Drop and recreate all tables."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.executescript(
        """
        DROP TABLE IF EXISTS transaction_history;
        DROP TABLE IF EXISTS card;
        DROP TABLE IF EXISTS customer;
    """
    )
    conn.commit()
    conn.close()
    init_db()
