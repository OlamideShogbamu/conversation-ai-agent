import random
import string
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.db.schema import get_connection, init_db, reset_db


def parse_date(date_str: str | None) -> str | None:
    """Parse date from DD/MM/YYYY HH:MM or DD/MM/YYYY HH:MM:SS format."""
    if not date_str or pd.isna(date_str):
        return None

    date_str = str(date_str).strip()

    formats = [
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%m/%d/%y %H:%M",
        "%d/%m/%y %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue

    return date_str


def generate_last_four(account_no: str = "", card_issuer: str = "") -> str:
    """Generate deterministic 4-digit number from account_no + card_issuer."""
    if account_no and card_issuer:
        # Create consistent hash from account + issuer
        seed = f"{account_no}_{card_issuer}"
        hash_val = hash(seed) % 10000
        return f"{hash_val:04d}"
    # Fallback to random if no data
    return "".join(random.choices(string.digits, k=4))


def export_sheet_csvs(filepath: str, output_dir: str) -> list[str]:
    """Export each sheet of an Excel file as a CSV. Returns list of CSV paths."""
    filepath = Path(filepath)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    excel = pd.ExcelFile(filepath)
    csv_paths = []

    for sheet_name in excel.sheet_names:
        df = pd.read_excel(excel, sheet_name=sheet_name)
        csv_name = sheet_name.strip().lower().replace(" ", "_") + ".csv"
        csv_path = output_dir / csv_name
        df.to_csv(csv_path, index=False)
        print(f"   Exported: {sheet_name} -> {csv_path} ({len(df)} rows)")
        csv_paths.append(str(csv_path))

    return csv_paths


def ingest_excel(filepath: str, reset: bool = False):
    """
    Ingest Excel file with Customer, Transaction, and Card sheets.

    Expected sheets:
    - Customer: ID, Account_No, Account_Name, Currency, Account_Type,
                Product_Type, Product_Description, Current_Balance, Account_Open_Date
    - Transaction: Account_No, Transaction_Date, Transaction_Type, Transaction_Amount,
                   Destination_Account, Narration, Destination_Account_Bank,
                   Transaction_Status, Failure_Reason
    - Card: Account_No, Card_Issuer, Card_Type, Card_Activation_Date, Status
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Excel file not found: {filepath}")

    if reset:
        reset_db()
    else:
        init_db()

    # Read Excel sheets
    excel = pd.ExcelFile(filepath)
    sheets = excel.sheet_names
    print(f"Found sheets: {sheets}")

    conn = get_connection()
    cursor = conn.cursor()

    # Ingest Customer sheet
    customer_sheet = next((s for s in sheets if "customer" in s.lower()), None)
    if customer_sheet:
        df = pd.read_excel(excel, sheet_name=customer_sheet)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        count = 0
        for _, row in df.iterrows():
            try:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO customer
                    (id, account_no, account_name, currency, account_type,
                     product_type, product_description, current_balance, account_open_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        int(row.get("id", 0)),
                        str(row.get("account_no", "")),
                        str(row.get("account_name", "")),
                        str(row.get("currency", "NGN")),
                        str(row.get("account_type", "")),
                        str(row.get("product_type", "")),
                        str(row.get("product_description", "")),
                        float(row.get("current_balance", 0) or 0),
                        parse_date(row.get("account_open_date")),
                    ),
                )
                count += 1
            except Exception as e:
                print(f"Error inserting customer row: {e}")
        print(f"Ingested {count} customers")

    # Ingest Transaction sheet
    txn_sheet = next((s for s in sheets if "transaction" in s.lower()), None)
    if txn_sheet:
        df = pd.read_excel(excel, sheet_name=txn_sheet)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        count = 0
        for _, row in df.iterrows():
            try:
                cursor.execute(
                    """
                    INSERT INTO transaction_history
                    (account_no, transaction_date, transaction_type, transaction_amount,
                     destination_account, narration, destination_bank,
                     transaction_status, failure_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        str(row.get("account_no", "")),
                        parse_date(row.get("transaction_date")),
                        str(row.get("transaction_type", "")),
                        float(row.get("transaction_amount", 0) or 0),
                        str(row.get("destination_account", "")),
                        str(row.get("narration", "")),
                        str(row.get("destination_account_bank", "")),
                        str(row.get("transaction_status", "")),
                        str(row.get("failure_reason", "") or ""),
                    ),
                )
                count += 1
            except Exception as e:
                print(f"Error inserting transaction row: {e}")
        print(f"Ingested {count} transactions")

    # Ingest Card sheet
    card_sheet = next((s for s in sheets if "card" in s.lower()), None)
    if card_sheet:
        df = pd.read_excel(excel, sheet_name=card_sheet)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        count = 0
        for _, row in df.iterrows():
            try:
                account_no = str(row.get("account_no", ""))
                card_issuer = str(row.get("card_issuer", ""))
                card_last_four = row.get("card_last_four") or generate_last_four(account_no, card_issuer)

                cursor.execute(
                    """
                    INSERT INTO card
                    (account_no, card_issuer, card_type, card_last_four,
                     card_activation_date, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        account_no,
                        card_issuer,
                        str(row.get("card_type", "")),
                        str(card_last_four),
                        parse_date(row.get("card_activation_date")),
                        str(row.get("status", "Active")),
                    ),
                )
                count += 1
            except Exception as e:
                print(f"Error inserting card row: {e}")
        print(f"Ingested {count} cards")

    conn.commit()
    conn.close()
    print("Excel ingestion complete!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.db.ingest_excel <excel_file_path> [--reset]")
        sys.exit(1)

    reset = "--reset" in sys.argv
    ingest_excel(sys.argv[1], reset=reset)
