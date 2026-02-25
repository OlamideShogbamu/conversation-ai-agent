#!/usr/bin/env python3
"""
Export Excel sheets to CSV and ingest into SQLite.
Resets the database before re-ingesting.

Usage:
    python scripts/ingest_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import BASE_DIR
from src.db.schema import reset_db, get_connection
from src.db.ingest_excel import ingest_excel, export_sheet_csvs

EXCEL_PATH = BASE_DIR / "customer_and_banking_data.xlsx"
CSV_DIR = BASE_DIR / "data" / "csv"


def show_schema():
    conn = get_connection()
    cursor = conn.cursor()

    tables = {
        "customer": "one row per account holder",
        "transaction_history": "FK -> customer(account_no)",
        "card": "FK -> customer(account_no)",
    }

    sep = "=" * 62
    print(f"\n{sep}")
    print("  SQLite Schema  —  data/globus.db")
    print(sep)

    for table, note in tables.items():
        cursor.execute(f"PRAGMA table_info({table})")
        cols = cursor.fetchall()
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]

        print(f"\n  TABLE: {table}  ({count} rows)  [{note}]")
        print("  " + "-" * 58)
        print(f"  {'Column':<30} {'Type':<12} Notes")
        print("  " + "-" * 58)
        for col in cols:
            notes = "PRIMARY KEY" if col["pk"] else ""
            print(f"  {col['name']:<30} {col['type']:<12} {notes}")

        cursor.execute(f"PRAGMA foreign_key_list({table})")
        fks = cursor.fetchall()
        if fks:
            for fk in fks:
                print(f"\n  FK: {table}.{fk['from']}  -->  {fk['table']}.{fk['to']}")

    print(f"\n{sep}")
    print("  Relationship Diagram")
    print(sep)
    print("""
  customer
  ├── account_no  (PK)
  ├── account_name, currency, account_type, product_type ...
  │
  ├──< transaction_history  (account_no FK)
  │       transaction_date, type, amount, status, narration ...
  │
  └──< card  (account_no FK)
          card_issuer, card_type, card_last_four, status ...
""")
    conn.close()


def main():
    print("=" * 50)
    print("  Data Ingest — SQLite")
    print("=" * 50)

    if not EXCEL_PATH.exists():
        print(f"ERROR: {EXCEL_PATH} not found.")
        sys.exit(1)

    print("\n[1/3] Resetting SQLite database...")
    reset_db()
    print("      Done.")

    print(f"\n[2/3] Exporting Excel sheets to CSV  ({CSV_DIR})")
    export_sheet_csvs(str(EXCEL_PATH), str(CSV_DIR))
    print("      Done.")

    print("\n[3/3] Ingesting into SQLite...")
    ingest_excel(str(EXCEL_PATH), reset=False)
    print("      Done.")

    show_schema()


if __name__ == "__main__":
    main()
