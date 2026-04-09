import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from config.settings import BASE_DIR
from src.logger import get_logger

DB_PATH = BASE_DIR / "data" / "globus.db"
logger = get_logger(__name__)

_local = threading.local()


def get_connection() -> sqlite3.Connection:
    """Return a per-thread cached SQLite connection, creating one if needed."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        _local.conn = conn
        logger.info("db_connection_created", extra={"thread": threading.current_thread().name})
    return conn


@contextmanager
def timed_query(operation: str):
    """Context manager that logs SQLite query duration."""
    t0 = time.time()
    try:
        yield
    finally:
        logger.info("db_query", extra={"operation": operation, "duration_ms": round((time.time() - t0) * 1000)})


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
    # Invalidate the cached connection for this thread so init_db gets a fresh one
    _local.conn = None
    init_db()
