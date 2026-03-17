from src.tools.registry import Tool, ToolParam, ToolRegistry
from src.memory.vector_store import VectorStore
from src.inference.embedder import Embedder
from src.rag.retriever import Retriever
from src.db.repository import CustomerRepository, TransactionRepository, CardRepository


def create_banking_tools(
    registry: ToolRegistry,
    vector_store: VectorStore,
    embedder: Embedder,
):
    """Register all banking-related tools."""

    retriever = Retriever(vector_store, embedder)

    # ==================== PRODUCT SEARCH TOOL ====================

    def search_products_handler(query: str, category: str | None = None) -> str:
        results = retriever.retrieve(query, limit=3, category=category)
        return retriever.format_context(results) or "No relevant product information found."

    registry.register(
        Tool(
            name="search_products",
            description="Search Globus Bank product catalog for information about loans, savings accounts, investments, or cards. Use this when the customer asks about product features, rates, eligibility, or how to apply.",
            parameters=[
                ToolParam("query", "string", "What to search for, e.g. 'salary advance loan requirements'"),
                ToolParam("category", "string", "Optional category filter", required=False,
                          enum=["loans", "investments", "savings", "cards"]),
            ],
            handler=search_products_handler,
        )
    )

    # ==================== PRODUCT TOOLS ====================

    def loans_handler(
        action: str,
        loan_type: str | None = None,
        compare_types: list[str] | None = None,
    ) -> str:
        if action == "list":
            query_vec = embedder.embed("loan products")
            results = vector_store.search(query_vec, limit=10, category="loans")
            if not results:
                return "Available loans: Salary Advance, Home Extension, Personal, Business, Mortgage, Vehicle, Overdraft, Working Capital. Ask about any specific loan for details."
            return "\n".join(f"- {r['name']}: {r.get('description', '')[:100]}" for r in results)

        elif action == "details" and loan_type:
            product = vector_store.get_by_id(f"loan_{loan_type}")
            if product:
                return (
                    f"**{product['name']}**\n"
                    f"Description: {product.get('description', 'N/A')}\n"
                    f"Features: {', '.join(product.get('features', []))}\n"
                    f"Benefits: {', '.join(product.get('benefits', []))}"
                )
            return f"Loan type '{loan_type}' not found. Available: salary_advance, home_extension, personal, business, mortgage, vehicle, overdraft, working_capital."

        elif action == "compare" and compare_types:
            products = [vector_store.get_by_id(f"loan_{t}") for t in compare_types]
            products = [p for p in products if p]
            if not products:
                return "No matching loan products found."
            lines = ["| Feature | " + " | ".join(p["name"] for p in products) + " |"]
            lines.append("|---|" + "---|" * len(products))
            for feature in ["description", "features"]:
                row = f"| {feature.title()} | "
                row += " | ".join(str(p.get(feature, "N/A"))[:50] for p in products)
                row += " |"
                lines.append(row)
            return "\n".join(lines)

        elif action == "eligibility" and loan_type:
            product = vector_store.get_by_id(f"loan_{loan_type}")
            if product and "eligibility" in product:
                return f"Eligibility for {product['name']}: {product['eligibility']}"
            return "Please visit a branch for detailed eligibility assessment."

        return "Invalid action or missing parameters."

    registry.register(
        Tool(
            name="loans",
            description="Get information about Globus Bank loan products",
            parameters=[
                ToolParam("action", "string", "Action to perform", enum=["list", "details", "compare", "eligibility"]),
                ToolParam("loan_type", "string", "Type of loan", required=False, enum=["salary_advance", "home_extension", "personal", "business", "mortgage", "vehicle", "overdraft", "working_capital"]),
                ToolParam("compare_types", "list", "Loan types to compare", required=False),
            ],
            handler=loans_handler,
        )
    )

    def investments_handler(action: str, product_type: str | None = None) -> str:
        if action == "list":
            query_vec = embedder.embed("investment products")
            results = vector_store.search(query_vec, limit=10, category="investments")
            if not results:
                return "Available: Bonds, Commercial Papers, Money Market Deposits, Treasury Bills."
            return "\n".join(f"- {r['name']}: {r.get('description', '')[:100]}" for r in results)

        elif action == "details" and product_type:
            product = vector_store.get_by_id(f"investment_{product_type}")
            if product:
                return (
                    f"**{product['name']}**\n"
                    f"Description: {product.get('description', 'N/A')}\n"
                    f"Features: {', '.join(product.get('features', []))}\n"
                    f"Benefits: {', '.join(product.get('benefits', []))}"
                )
            return f"Investment type '{product_type}' not found."

        elif action == "rates":
            return (
                "Current indicative rates (subject to change):\n"
                "- Treasury Bills: 10-15% p.a.\n"
                "- Bonds: 12-18% p.a.\n"
                "- Commercial Papers: 12-20% p.a.\n"
                "- Money Market: 8-14% p.a.\n"
                "Contact your relationship manager for exact rates."
            )

        return "Invalid action or missing parameters."

    registry.register(
        Tool(
            name="investments",
            description="Get information about investment products",
            parameters=[
                ToolParam("action", "string", "Action to perform", enum=["list", "details", "rates"]),
                ToolParam("product_type", "string", "Type of investment", required=False, enum=["bonds", "commercial_papers", "money_market", "treasury_bills"]),
            ],
            handler=investments_handler,
        )
    )

    def savings_handler(action: str, account_type: str | None = None) -> str:
        if action == "list":
            query_vec = embedder.embed("savings account products")
            results = vector_store.search(query_vec, limit=10, category="savings")
            if not results:
                return "Available: Domiciliary (USD/EUR/GBP), Kiddies, Savings, Current, Non-Resident."
            return "\n".join(f"- {r['name']}: {r.get('description', '')[:100]}" for r in results)

        elif action == "details" and account_type:
            product = vector_store.get_by_id(f"savings_{account_type}")
            if product:
                return (
                    f"**{product['name']}**\n"
                    f"Description: {product.get('description', 'N/A')}\n"
                    f"Features: {', '.join(product.get('features', []))}"
                )
            return f"Account type '{account_type}' not found."

        elif action == "requirements":
            return (
                "Account opening requirements:\n"
                "- Valid ID (National ID, Passport, Driver's License)\n"
                "- Proof of address (Utility bill < 3 months)\n"
                "- Passport photograph\n"
                "- BVN (Bank Verification Number)\n"
                "- Minimum opening balance varies by account type"
            )

        return "Invalid action or missing parameters."

    registry.register(
        Tool(
            name="savings_accounts",
            description="Get information about savings and account products",
            parameters=[
                ToolParam("action", "string", "Action to perform", enum=["list", "details", "compare", "requirements"]),
                ToolParam("account_type", "string", "Type of account", required=False, enum=["domiciliary", "kiddies", "savings", "current", "non_resident"]),
            ],
            handler=savings_handler,
        )
    )

    def debit_cards_handler(action: str, card_scheme: str | None = None) -> str:
        if action == "info":
            product = vector_store.get_by_id("debit_cards")
            if product:
                return (
                    f"**Globus Bank Debit Cards**\n"
                    f"Description: {product.get('description', 'N/A')}\n"
                    f"Features: {', '.join(product.get('features', []))}\n"
                    f"Available: Verve, Visa, MasterCard"
                )
            return (
                "Globus Bank Debit Cards - easy, convenient, secure payments.\n"
                "Available in Verve, Visa, MasterCard.\n"
                "Chip + PIN security, multi-channel (ATM, POS, Web), 3-year validity."
            )

        elif action == "features":
            return (
                "Debit Card Features:\n"
                "- Chip + PIN technology\n"
                "- Verified by Visa, OTP, SecureCode\n"
                "- Multi-channel: ATM, POS, Web\n"
                "- Validity: minimum 3 years\n"
                "- Embossed and personalized"
            )

        elif action == "apply":
            return (
                "To apply for a debit card:\n"
                "1. Visit any Globus Bank branch with valid ID\n"
                "2. Fill out the card request form\n"
                "3. Ready in 3-5 business days\n"
                "4. Collect and activate at branch"
            )

        return "Invalid action."

    registry.register(
        Tool(
            name="debit_cards",
            description="Get information about Globus Bank debit cards",
            parameters=[
                ToolParam("action", "string", "Action to perform", enum=["info", "features", "apply"]),
                ToolParam("card_scheme", "string", "Card scheme", required=False, enum=["verve", "visa", "mastercard"]),
            ],
            handler=debit_cards_handler,
        )
    )

    # ==================== ACCOUNT TOOLS (Database) ====================

    def account_info_handler(
        action: str,
        account_no: str | None = None,
    ) -> str:
        if action == "types":
            return (
                "Globus Bank Account Types:\n"
                "1. **Savings Account** - Personal savings, 4.05% interest\n"
                "2. **Current Account** - Frequent transactions, cheque book\n"
                "3. **Domiciliary Account** - Foreign currency (USD, EUR, GBP)\n"
                "4. **Kiddies Account** - For children, financial literacy\n"
                "5. **Non-Resident Account** - For Nigerians abroad"
            )

        elif action == "balance" and account_no:
            customer = CustomerRepository.get_by_account_no(account_no)
            if customer:
                return (
                    f"Customer name is {customer['account_name']}. "
                    f"Account type is {customer['account_type']}. "
                    f"Current balance is {customer['currency']} {customer['current_balance']:,.2f}."
                )
            return "Account not found. Please verify your account number."

        elif action == "details" and account_no:
            customer = CustomerRepository.get_by_account_no(account_no)
            if customer:
                return (
                    f"Customer name is {customer['account_name']}. "
                    f"Account number is {customer['account_no']}. "
                    f"Account type is {customer['account_type']}. "
                    f"Product is {customer['product_description']}. "
                    f"Currency is {customer['currency']}. "
                    f"Current balance is {customer['currency']} {customer['current_balance']:,.2f}. "
                    f"Account opened on {customer['account_open_date']}."
                )
            return "Account not found."

        return "Invalid action or missing account number."

    registry.register(
        Tool(
            name="account_info",
            description="Get customer account information, balance, and details",
            parameters=[
                ToolParam("action", "string", "Action to perform", enum=["types", "balance", "details"]),
                ToolParam("account_no", "string", "Customer account number", required=False),
            ],
            handler=account_info_handler,
        )
    )

    def transaction_history_handler(
        action: str,
        account_no: str,
        limit: int = 5,
    ) -> str:
        if action == "recent":
            transactions = TransactionRepository.get_recent(account_no, n=limit)
            if not transactions:
                return "No transactions found for this account."

            lines = [f"Here are the {len(transactions)} most recent transactions:"]
            for i, txn in enumerate(transactions, 1):
                amount = txn["transaction_amount"]
                txn_type = txn["transaction_type"]
                lines.append(
                    f"{i}. On {txn['transaction_date']}, a {txn_type} of NGN {amount:,.2f} "
                    f"for '{txn['narration']}' with status {txn['transaction_status']}."
                )
            return "\n".join(lines)

        elif action == "summary":
            summary = TransactionRepository.get_summary(account_no)
            if not summary or summary.get("total_transactions", 0) == 0:
                return "No transaction history found."

            return (
                f"Total number of transactions is {summary['total_transactions']}. "
                f"Total credits amount to NGN {summary['total_credits'] or 0:,.2f}. "
                f"Total debits amount to NGN {summary['total_debits'] or 0:,.2f}. "
                f"Number of failed transactions is {summary['failed_count']}."
            )

        return "Invalid action."

    registry.register(
        Tool(
            name="transaction_history",
            description="Get customer transaction history and summary",
            parameters=[
                ToolParam("action", "string", "Action to perform", enum=["recent", "summary"]),
                ToolParam("account_no", "string", "Customer account number"),
                ToolParam("limit", "integer", "Number of transactions to retrieve", required=False),
            ],
            handler=transaction_history_handler,
        )
    )

    # ==================== CARD BLOCKING TOOL (Database) ====================

    def block_card_handler(
        action: str,
        account_no: str | None = None,
        card_last_four: str | None = None,
        reason: str | None = None,
        confirmed: bool = False,
    ) -> str:
        if action == "list_cards" and account_no:
            cards = CardRepository.get_by_account(account_no)
            if not cards:
                return "NO_CARDS_FOUND"

            lines = ["Cards linked to your account:"]
            for i, card in enumerate(cards, 1):
                status_icon = "🟢" if card["status"] == "Active" else "🔴"
                lines.append(
                    f"{i}. {card['card_issuer']} {card['card_type']} "
                    f"ending in {card['card_last_four']} - {status_icon} {card['status']}"
                )
            return "\n".join(lines)

        elif action == "check_multiple" and account_no:
            active_cards = CardRepository.get_active_cards(account_no)

            if not active_cards:
                return "NO_ACTIVE_CARDS"
            elif len(active_cards) == 1:
                card = active_cards[0]
                return f"SINGLE_CARD:{card['card_issuer']} {card['card_type']}:{card['card_last_four']}"
            else:
                card_list = "\n".join(
                    f"- {c['card_issuer']} {c['card_type']} ending in {c['card_last_four']}"
                    for c in active_cards
                )
                return f"MULTIPLE_CARDS:{card_list}"

        elif action == "block" and account_no and card_last_four:
            card = CardRepository.get_by_last_four(account_no, card_last_four)
            if not card:
                return f"No card ending in {card_last_four} found on this account."

            if card["status"] == "Blocked":
                return f"This card ({card['card_issuer']} {card['card_type']} ending in {card_last_four}) is already blocked."

            if not confirmed:
                return (
                    f"CONFIRM_REQUIRED\n"
                    f"Card: {card['card_issuer']} {card['card_type']} ending in {card_last_four}\n"
                    f"Reason: {reason or 'Not specified'}\n"
                    f"Please confirm to proceed."
                )

            # Block the card
            success = CardRepository.block_card(card["id"])
            if success:
                return (
                    f"SUCCESS\n"
                    f"Your {card['card_issuer']} {card['card_type']} ending in {card_last_four} "
                    f"has been blocked successfully.\n\n"
                    f"Next steps:\n"
                    f"1. If lost/stolen, file a police report\n"
                    f"2. Visit any Globus Bank branch for replacement\n"
                    f"3. Replacement ready in 3-5 business days\n\n"
                    f"Reference: BLK{account_no[-4:]}{card_last_four}"
                )
            return "Failed to block card. Please try again or visit a branch."

        return "Invalid action or missing parameters."

    registry.register(
        Tool(
            name="block_card",
            description="Block a customer's debit or credit card. Handles multiple cards scenario.",
            parameters=[
                ToolParam("action", "string", "Action to perform", enum=["list_cards", "check_multiple", "block"]),
                ToolParam("account_no", "string", "Customer account number", required=False),
                ToolParam("card_last_four", "string", "Last 4 digits of card", required=False),
                ToolParam("reason", "string", "Reason for blocking", required=False),
                ToolParam("confirmed", "boolean", "Customer confirmation", required=False),
            ],
            handler=block_card_handler,
        )
    )

    # ==================== CALCULATOR TOOL ====================

    def calculate_handler(
        calculation_type: str,
        principal: float,
        rate: float,
        tenure_months: int,
    ) -> str:
        if calculation_type == "loan_emi":
            monthly_rate = rate / 100 / 12
            if monthly_rate == 0:
                emi = principal / tenure_months
            else:
                emi = (
                    principal
                    * monthly_rate
                    * ((1 + monthly_rate) ** tenure_months)
                    / (((1 + monthly_rate) ** tenure_months) - 1)
                )
            total = emi * tenure_months
            interest = total - principal
            return (
                f"**Loan EMI Calculation**\n"
                f"- Principal: ₦{principal:,.2f}\n"
                f"- Rate: {rate}% p.a.\n"
                f"- Tenure: {tenure_months} months\n"
                f"- Monthly EMI: ₦{emi:,.2f}\n"
                f"- Total Payment: ₦{total:,.2f}\n"
                f"- Total Interest: ₦{interest:,.2f}"
            )

        elif calculation_type == "investment_return":
            final = principal * ((1 + rate / 100) ** (tenure_months / 12))
            returns = final - principal
            return (
                f"**Investment Return**\n"
                f"- Principal: ₦{principal:,.2f}\n"
                f"- Rate: {rate}% p.a.\n"
                f"- Tenure: {tenure_months} months\n"
                f"- Final Value: ₦{final:,.2f}\n"
                f"- Returns: ₦{returns:,.2f}"
            )

        elif calculation_type == "interest":
            interest = principal * (rate / 100) * (tenure_months / 12)
            return (
                f"**Simple Interest**\n"
                f"- Principal: ₦{principal:,.2f}\n"
                f"- Rate: {rate}% p.a.\n"
                f"- Tenure: {tenure_months} months\n"
                f"- Interest: ₦{interest:,.2f}"
            )

        return "Invalid calculation type."

    registry.register(
        Tool(
            name="calculate",
            description="Calculate loan EMI, investment returns, or interest",
            parameters=[
                ToolParam("calculation_type", "string", "Type of calculation", enum=["loan_emi", "investment_return", "interest"]),
                ToolParam("principal", "number", "Principal amount in Naira"),
                ToolParam("rate", "number", "Annual interest rate in %"),
                ToolParam("tenure_months", "integer", "Tenure in months"),
            ],
            handler=calculate_handler,
        )
    )
