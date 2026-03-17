import time

from src.inference.llm import LLMEngine
from src.memory.conversation import ConversationMemory
from src.tools.registry import ToolRegistry
from src.tools.executor import ToolExecutor
from src.rag.retriever import Retriever
from src.db.repository import CustomerRepository, TransactionRepository, CardRepository
from src.agent.intent_extractor import IntentExtractor, CHAINABLE_PAIRS
from src.agent.prompts import (
    MULTI_CARD_CLARIFICATION,
    CARD_BLOCK_CONFIRMATION,
    build_response_prompt,
    build_general_chat_prompt,
)


class AgentOrchestrator:
    def __init__(self, llm: LLMEngine, tool_registry: ToolRegistry, retriever: Retriever = None):
        self.llm = llm
        self.memory = ConversationMemory()
        self.tool_executor = ToolExecutor(tool_registry)
        self.tool_registry = tool_registry
        self.retriever = retriever
        self.intent_extractor = IntentExtractor(llm)
        self.pending_action = None

    def run(self, user_input: str) -> str:
        request_start = time.time()

        # Handle pending card block confirmation
        if self._is_confirmation_response(user_input):
            return self._handle_confirmation(user_input)

        history = self.memory.format_for_prompt()

        # ── Stage 1: Extract intent (3-vote majority) ────────────────────────
        print(f"\n[Agent] Stage 1: extracting intent...", flush=True)
        intent, entities, confidence = self.intent_extractor.extract(user_input, history)
        print(f"[Agent] Intent={intent}, confidence={confidence}", flush=True)

        # Pop chain_intent before clarification check (it's optional, never blocks)
        chain_intent = entities.pop("chain_intent", None)
        if chain_intent:
            print(f"[Agent] Chain intent detected: {chain_intent}", flush=True)

        # ── Clarification gate — only block if required entities are missing ──
        missing = self.intent_extractor.get_missing_entities(intent, entities)
        if missing:
            clarification = self.intent_extractor.get_clarification_question(missing)
            self.memory.add("user", user_input)
            self.memory.add("assistant", clarification)
            print(f"[Agent] Clarifying — missing: {missing}", flush=True)
            return clarification

        # ── Stage 2: Execute intent (direct DB/retriever, no LLM) ────────────
        print(f"[Agent] Stage 2: executing intent...", flush=True)
        t2 = time.time()
        tool_result = self._execute_intent(intent, entities, user_input)

        # ── Stage 2b: Execute chain intent if valid and entities permit ───────
        if (
            chain_intent
            and (intent, chain_intent) in CHAINABLE_PAIRS
            and tool_result
            and not tool_result.startswith("__DIRECT_RESPONSE__:")
        ):
            print(f"[Agent] Stage 2b: chaining → {chain_intent}...", flush=True)
            chain_result = self._execute_intent(chain_intent, entities, user_input)
            if chain_result and not chain_result.startswith("__DIRECT_RESPONSE__:"):
                tool_result = f"{tool_result}\n\n{chain_result}"
                print(f"[Agent] Chain executed successfully", flush=True)

        print(f"[Agent] Stage 2 done ({time.time() - t2:.1f}s)", flush=True)

        # Short-circuit for direct responses (e.g. card blocking confirmation prompts)
        if isinstance(tool_result, str) and tool_result.startswith("__DIRECT_RESPONSE__:"):
            response = tool_result.removeprefix("__DIRECT_RESPONSE__:")
            self.memory.add("user", user_input)
            self.memory.add("assistant", response)
            print(f"[Agent] Total time: {time.time() - request_start:.1f}s", flush=True)
            return response

        # ── Stage 3: Format response (focused LLM call) ───────────────────────
        if tool_result:
            print(f"[Agent] Stage 3: formatting response...", flush=True)
            t3 = time.time()
            prompt = build_response_prompt(user_input, tool_result)
            response = self.llm.generate(prompt, stop=["Customer:", "\nCustomer:", "Data retrieved"])
            print(f"[Agent] Stage 3 done ({time.time() - t3:.1f}s)", flush=True)
        else:
            # general_chat or no data — use conversation-aware prompt
            print(f"[Agent] Stage 3: general chat response...", flush=True)
            t3 = time.time()
            prompt = build_general_chat_prompt(user_input, history)
            response = self.llm.generate(prompt, stop=["Customer:", "\nCustomer:"])
            print(f"[Agent] Stage 3 done ({time.time() - t3:.1f}s)", flush=True)

        response = response.strip()
        print(f"[Agent] Total time: {time.time() - request_start:.1f}s", flush=True)

        self.memory.add("user", user_input)
        self.memory.add("assistant", response)
        return response

    # ── Intent → Data execution ───────────────────────────────────────────────

    def _execute_intent(self, intent: str, entities: dict, user_input: str) -> str:
        account_no = entities.get("account_no")

        if intent == "check_balance":
            customer = CustomerRepository.get_by_account_no(account_no)
            if customer:
                return (
                    f"Customer name is {customer['account_name']}. "
                    f"Account type is {customer['account_type']}. "
                    f"Current balance is {customer['currency']} {customer['current_balance']:,.2f}."
                )
            return "No account found with that account number."

        elif intent == "account_details":
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
            return "No account found with that account number."

        elif intent == "check_transactions":
            limit = int(entities.get("limit") or 5)
            transactions = TransactionRepository.get_recent(account_no, n=limit)
            if not transactions:
                return "No transactions found for this account."
            lines = [f"Here are the {len(transactions)} most recent transactions:"]
            for i, txn in enumerate(transactions, 1):
                lines.append(
                    f"{i}. On {txn['transaction_date']}, a {txn['transaction_type']} "
                    f"of NGN {txn['transaction_amount']:,.2f} for '{txn['narration']}' "
                    f"— status: {txn['transaction_status']}."
                )
            return "\n".join(lines)

        elif intent == "transaction_summary":
            summary = TransactionRepository.get_summary(account_no)
            if not summary or summary.get("total_transactions", 0) == 0:
                return "No transaction history found for this account."
            return (
                f"Total number of transactions is {summary['total_transactions']}. "
                f"Total credits amount to NGN {summary['total_credits'] or 0:,.2f}. "
                f"Total debits amount to NGN {summary['total_debits'] or 0:,.2f}. "
                f"Number of failed transactions is {summary['failed_count']}."
            )

        elif intent == "account_types":
            return (
                "Globus Bank offers the following account types: "
                "Savings Account (personal savings, 4.05% interest), "
                "Current Account (frequent transactions, cheque book), "
                "Domiciliary Account (foreign currency in USD, EUR, GBP), "
                "Kiddies Account (for children), "
                "and Non-Resident Account (for Nigerians abroad)."
            )

        elif intent == "product_search":
            query = entities.get("query", user_input)
            if self.retriever:
                results = self.retriever.retrieve(query, limit=3)
                return self.retriever.format_context(results) or "No relevant products found."
            return ""

        elif intent == "loan_info":
            query = entities.get("loan_type") or "loan products"
            if self.retriever:
                results = self.retriever.retrieve(query, limit=3, category="loans")
                return self.retriever.format_context(results) or (
                    "Available loans: Salary Advance, Home Extension, Personal, Business, "
                    "Mortgage, Vehicle, Overdraft, Working Capital."
                )
            return ""

        elif intent == "investment_info":
            query = entities.get("product_type") or "investment products"
            if self.retriever:
                results = self.retriever.retrieve(query, limit=3, category="investments")
                return self.retriever.format_context(results) or (
                    "Available investments: Bonds, Commercial Papers, Money Market Deposits, Treasury Bills."
                )
            return ""

        elif intent == "savings_info":
            query = entities.get("account_type") or "savings account"
            if self.retriever:
                results = self.retriever.retrieve(query, limit=3, category="savings")
                return self.retriever.format_context(results) or (
                    "Available accounts: Domiciliary, Kiddies, Savings, Current, Non-Resident."
                )
            return ""

        elif intent == "card_info":
            if self.retriever:
                results = self.retriever.retrieve("debit card", limit=2, category="cards")
                return self.retriever.format_context(results) or (
                    "Globus Bank offers Verve, Visa, and MasterCard debit cards with chip + PIN security."
                )
            return ""

        elif intent == "block_card":
            return self._handle_block_card(account_no, entities)

        elif intent == "calculate":
            return self._handle_calculate(entities)

        # general_chat — no data, handled by LLM directly
        return ""

    # ── Card blocking (multi-step, kept separate) ─────────────────────────────

    def _handle_block_card(self, account_no: str, entities: dict) -> str:
        active_cards = CardRepository.get_active_cards(account_no)

        if not active_cards:
            return (
                "__DIRECT_RESPONSE__:I couldn't find any active cards linked to your account. "
                "Please visit a branch or call customer service for assistance."
            )

        card_last_four = entities.get("card_last_four")

        # If a specific card was already identified, find it directly
        if card_last_four:
            card = next((c for c in active_cards if c["card_last_four"] == card_last_four), None)
            if not card:
                return f"__DIRECT_RESPONSE__:I couldn't find an active card ending in {card_last_four} on your account."
        elif len(active_cards) > 1:
            # Multiple cards, none specified — ask which one and await selection
            card_list = "\n".join(
                f"- {c['card_issuer']} {c['card_type']} ending in {c['card_last_four']}"
                for c in active_cards
            )
            self.pending_action = {
                "type": "select_card",
                "args": {
                    "account_no": account_no,
                    "active_cards": active_cards,
                    "reason": entities.get("reason", "Not specified"),
                },
            }
            response = MULTI_CARD_CLARIFICATION.format(card_list=card_list)
            return f"__DIRECT_RESPONSE__:{response}"
        else:
            card = active_cards[0]

        # Ask for confirmation before blocking
        self.pending_action = {
            "type": "block_card",
            "args": {
                "account_no": account_no,
                "card_last_four": card["card_last_four"],
                "card_type": f"{card['card_issuer']} {card['card_type']}",
                "reason": entities.get("reason", "Not specified"),
            },
        }
        response = CARD_BLOCK_CONFIRMATION.format(
            card_type=f"{card['card_issuer']} {card['card_type']}",
            last_four=card["card_last_four"],
            reason=entities.get("reason", "Not specified"),
        )
        return f"__DIRECT_RESPONSE__:{response}"

    def _handle_calculate(self, entities: dict) -> str:
        try:
            calc_type = entities.get("calculation_type")
            principal = float(entities.get("principal", 0))
            rate = float(entities.get("rate", 0))
            tenure = int(entities.get("tenure_months", 0))

            if calc_type == "loan_emi":
                monthly_rate = rate / 100 / 12
                if monthly_rate == 0:
                    emi = principal / tenure
                else:
                    emi = (
                        principal
                        * monthly_rate
                        * ((1 + monthly_rate) ** tenure)
                        / (((1 + monthly_rate) ** tenure) - 1)
                    )
                total = emi * tenure
                interest = total - principal
                return (
                    f"For a loan of NGN {principal:,.2f} at {rate}% per annum over {tenure} months: "
                    f"monthly EMI is NGN {emi:,.2f}, total repayment is NGN {total:,.2f}, "
                    f"and total interest is NGN {interest:,.2f}."
                )

            elif calc_type == "investment_return":
                final = principal * ((1 + rate / 100) ** (tenure / 12))
                returns = final - principal
                return (
                    f"For an investment of NGN {principal:,.2f} at {rate}% per annum over {tenure} months: "
                    f"final value is NGN {final:,.2f} and total return is NGN {returns:,.2f}."
                )

            elif calc_type == "interest":
                interest = principal * (rate / 100) * (tenure / 12)
                return (
                    f"Simple interest on NGN {principal:,.2f} at {rate}% per annum over {tenure} months "
                    f"is NGN {interest:,.2f}."
                )
        except (TypeError, ValueError, ZeroDivisionError):
            pass
        return "Could not calculate — please check the values provided."

    # ── Confirmation handling (card blocking) ─────────────────────────────────

    def _is_confirmation_response(self, user_input: str) -> bool:
        if not self.pending_action:
            return False
        # Any reply routes through pending handler when awaiting card selection
        if self.pending_action["type"] == "select_card":
            return True
        # For confirmation prompts accept "yes"/"no" even with trailing words
        first_word = user_input.lower().strip().split()[0] if user_input.strip() else ""
        return first_word in {"yes", "no", "confirm", "cancel", "y", "n"}

    def _handle_confirmation(self, user_input: str) -> str:
        if not self.pending_action:
            return self.run(user_input)

        # ── Card selection (multi-card scenario) ──────────────────────────────
        if self.pending_action["type"] == "select_card":
            args = self.pending_action["args"]
            text = user_input.lower()

            # Try to match by last-four digits or card issuer/type
            matched = next(
                (c for c in args["active_cards"]
                 if c["card_last_four"] in text
                 or c["card_issuer"].lower() in text
                 or c["card_type"].lower() in text),
                None,
            )

            if not matched:
                card_list = "\n".join(
                    f"- {c['card_issuer']} {c['card_type']} ending in {c['card_last_four']}"
                    for c in args["active_cards"]
                )
                response = (
                    f"I couldn't identify which card you'd like to block. "
                    f"Please specify the card type or the last 4 digits:\n{card_list}"
                )
                self.memory.add("user", user_input)
                self.memory.add("assistant", response)
                return response

            # Move to confirmation step for the identified card
            self.pending_action = {
                "type": "block_card",
                "args": {
                    "account_no": args["account_no"],
                    "card_last_four": matched["card_last_four"],
                    "card_type": f"{matched['card_issuer']} {matched['card_type']}",
                    "reason": args["reason"],
                },
            }
            response = CARD_BLOCK_CONFIRMATION.format(
                card_type=f"{matched['card_issuer']} {matched['card_type']}",
                last_four=matched["card_last_four"],
                reason=args["reason"],
            )
            self.memory.add("user", user_input)
            self.memory.add("assistant", response)
            return response

        is_confirmed = user_input.lower().strip().split()[0] in {"yes", "confirm", "y"}

        if self.pending_action["type"] == "block_card":
            args = self.pending_action["args"]
            self.pending_action = None

            if is_confirmed:
                card = CardRepository.get_by_last_four(args["account_no"], args["card_last_four"])
                if card:
                    success = CardRepository.block_card(card["id"])
                    if success:
                        response = (
                            f"Your {args['card_type']} ending in {args['card_last_four']} "
                            f"has been successfully blocked.\n\n"
                            f"Next steps:\n"
                            f"1. If lost/stolen, file a police report\n"
                            f"2. Visit any Globus Bank branch for a replacement\n"
                            f"3. Replacement ready in 3–5 business days\n\n"
                            f"Reference: BLK{args['account_no'][-4:]}{args['card_last_four']}"
                        )
                    else:
                        response = "Failed to block the card. Please visit a branch or call customer service."
                else:
                    response = "Card not found. Please verify the details and try again."
            else:
                response = "Card blocking has been cancelled. Your card remains active. Is there anything else I can help you with?"

            self.memory.add("user", user_input)
            self.memory.add("assistant", response)
            return response

        self.pending_action = None
        return self.run(user_input)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def reset(self):
        self.memory.clear()
        self.pending_action = None

    def get_conversation_summary(self) -> str:
        return self.memory.summary

    def get_token_usage(self) -> int:
        return self.memory.get_token_usage()
