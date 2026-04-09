from src.db.schema import get_connection, timed_query


class CustomerRepository:
    @staticmethod
    def get_by_account_no(account_no: str) -> dict | None:
        with timed_query("customer.get_by_account_no"):
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM customer WHERE account_no = ?",
                (account_no,),
            )
            row = cursor.fetchone()
        return dict(row) if row else None

    @staticmethod
    def get_by_id(customer_id: int) -> dict | None:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM customer WHERE id = ?", (customer_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    @staticmethod
    def search_by_name(name: str) -> list[dict]:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM customer WHERE account_name LIKE ?",
            (f"%{name}%",),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def get_balance(account_no: str) -> float | None:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT current_balance FROM customer WHERE account_no = ?",
            (account_no,),
        )
        row = cursor.fetchone()
        return row["current_balance"] if row else None


class TransactionRepository:
    @staticmethod
    def get_by_account(
        account_no: str,
        limit: int = 10,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict]:
        conn = get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM transaction_history WHERE account_no = ?"
        params = [account_no]

        if start_date:
            query += " AND transaction_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND transaction_date <= ?"
            params.append(end_date)

        query += " ORDER BY transaction_date DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def get_recent(account_no: str, n: int = 5) -> list[dict]:
        with timed_query("transaction.get_recent"):
            return TransactionRepository.get_by_account(account_no, limit=n)

    @staticmethod
    def get_summary(account_no: str) -> dict:
        with timed_query("transaction.get_summary"):
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    COUNT(*) as total_transactions,
                    SUM(CASE WHEN transaction_type = 'Credit' THEN transaction_amount ELSE 0 END) as total_credits,
                    SUM(CASE WHEN transaction_type = 'Debit' THEN ABS(transaction_amount) ELSE 0 END) as total_debits,
                    COUNT(CASE WHEN transaction_status = 'Failed' THEN 1 END) as failed_count
                FROM transaction_history
                WHERE account_no = ?
            """,
                (account_no,),
            )
            row = cursor.fetchone()
        return dict(row) if row else {}


class CardRepository:
    @staticmethod
    def get_by_account(account_no: str) -> list[dict]:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM card WHERE account_no = ?",
            (account_no,),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def get_active_cards(account_no: str) -> list[dict]:
        with timed_query("card.get_active_cards"):
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM card WHERE account_no = ? AND status = 'Active'",
                (account_no,),
            )
            rows = cursor.fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def get_by_last_four(account_no: str, last_four: str) -> dict | None:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM card WHERE account_no = ? AND card_last_four = ?",
            (account_no, last_four),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    @staticmethod
    def block_card(card_id: int) -> bool:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE card SET status = 'Blocked' WHERE id = ?",
            (card_id,),
        )
        affected = cursor.rowcount
        conn.commit()
        return affected > 0

    @staticmethod
    def update_status(card_id: int, status: str) -> bool:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE card SET status = ? WHERE id = ?",
            (status, card_id),
        )
        affected = cursor.rowcount
        conn.commit()
        return affected > 0
