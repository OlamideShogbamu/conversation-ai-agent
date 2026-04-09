import json
import time

from src.inference.llm import LLMEngine
from src.memory.conversation import ConversationMemory
from src.tools.registry import ToolRegistry
from src.tools.executor import ToolExecutor
from src.rag.retriever import Retriever
from src.agent.intent_extractor import IntentExtractor, CHAINABLE_PAIRS
from src.agent.prompts import (
    MULTI_CARD_CLARIFICATION,
    CARD_BLOCK_CONFIRMATION,
    build_response_prompt,
    build_general_chat_prompt,
)
from src.logger import get_logger

logger = get_logger(__name__)


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
        logger.info("stage_start", extra={"stage": 1, "description": "intent extraction"})
        intent, entities, confidence = self.intent_extractor.extract(user_input, history)
        logger.info("intent_extracted", extra={"intent": intent, "confidence": confidence, "entities": entities})

        # Pop chain_intent before clarification check (it's optional, never blocks)
        chain_intent = entities.pop("chain_intent", None)
        if chain_intent:
            logger.info("chain_intent_detected", extra={"chain_intent": chain_intent})

        # ── Clarification gate — only block if required entities are missing ──
        missing = self.intent_extractor.get_missing_entities(intent, entities)
        if missing:
            clarification = self.intent_extractor.get_clarification_question(missing)
            self.memory.add("user", user_input)
            self.memory.add("assistant", clarification)
            logger.info("clarification_required", extra={"missing_entities": missing})
            return clarification

        # ── Stage 2: Execute intent (direct DB/retriever, no LLM) ────────────
        logger.info("stage_start", extra={"stage": 2, "description": "intent execution", "intent": intent})
        t2 = time.time()
        tool_result = self._execute_intent(intent, entities, user_input)

        # ── Stage 2b: Execute chain intent if valid and entities permit ───────
        if (
            chain_intent
            and (intent, chain_intent) in CHAINABLE_PAIRS
            and tool_result
            and not tool_result.startswith("__DIRECT_RESPONSE__:")
        ):
            logger.info("stage_start", extra={"stage": "2b", "description": f"chain intent → {chain_intent}"})
            chain_result = self._execute_intent(chain_intent, entities, user_input)
            if chain_result and not chain_result.startswith("__DIRECT_RESPONSE__:"):
                tool_result = f"{tool_result}\n\n{chain_result}"
                logger.info("chain_intent_complete", extra={"chain_intent": chain_intent})

        logger.info("stage_done", extra={"stage": 2, "duration_s": round(time.time() - t2, 2)})

        # Short-circuit for direct responses (e.g. card blocking confirmation prompts)
        if isinstance(tool_result, str) and tool_result.startswith("__DIRECT_RESPONSE__:"):
            response = tool_result.removeprefix("__DIRECT_RESPONSE__:")
            self.memory.add("user", user_input)
            self.memory.add("assistant", response)
            logger.info("request_complete", extra={"path": "direct_response", "total_s": round(time.time() - request_start, 2)})
            return response

        # ── Stage 3: Format response (focused LLM call) ───────────────────────
        t3 = time.time()
        if tool_result:
            logger.info("stage_start", extra={"stage": 3, "description": "response formatting (tool result)"})
            prompt = build_response_prompt(user_input, tool_result)
            response = self.llm.generate(prompt, stop=["Customer:", "\nCustomer:", "Data retrieved"])
        else:
            logger.info("stage_start", extra={"stage": 3, "description": "general chat response"})
            prompt = build_general_chat_prompt(user_input, history)
            response = self.llm.generate(prompt, stop=["Customer:", "\nCustomer:"])

        logger.info("stage_done", extra={"stage": 3, "duration_s": round(time.time() - t3, 2)})
        response = response.strip()
        logger.info("request_complete", extra={"path": "llm_response", "total_s": round(time.time() - request_start, 2), "token_usage": self.get_token_usage()})

        self.memory.add("user", user_input)
        self.memory.add("assistant", response)
        return response

    # ── Intent → Tool dispatch table ──────────────────────────────────────────
    # Maps intent name → (tool_name, args_builder).
    # Adding a new intent is a one-line change here + a handler in banking.py.

    _INTENT_TO_TOOL = {
        "check_balance":       ("account_info",        lambda e, u: {"action": "balance",  "account_no": e.get("account_no")}),
        "account_details":     ("account_info",        lambda e, u: {"action": "details",  "account_no": e.get("account_no")}),
        "account_types":       ("account_info",        lambda e, u: {"action": "types"}),
        "check_transactions":  ("transaction_history", lambda e, u: {"action": "recent",   "account_no": e.get("account_no"), "limit": int(e.get("limit") or 5)}),
        "transaction_summary": ("transaction_history", lambda e, u: {"action": "summary",  "account_no": e.get("account_no")}),
        "product_search":      ("search_products",     lambda e, u: {"query": e.get("query", u)}),
        "loan_info":           ("search_products",     lambda e, u: {"query": e.get("loan_type") or "loan products",          "category": "loans"}),
        "investment_info":     ("search_products",     lambda e, u: {"query": e.get("product_type") or "investment products", "category": "investments"}),
        "savings_info":        ("search_products",     lambda e, u: {"query": e.get("account_type") or "savings account",    "category": "savings"}),
        "card_info":           ("search_products",     lambda e, u: {"query": "debit card", "category": "cards"}),
        "calculate":           ("calculate",           lambda e, u: {
            "calculation_type": e.get("calculation_type"),
            "principal":        float(e.get("principal", 0)),
            "rate":             float(e.get("rate", 0)),
            "tenure_months":    int(e.get("tenure_months", 0)),
        }),
    }

    # Personal-data keywords that should never route to RAG product search
    _PERSONAL_SIGNALS = frozenset(["my transaction", "my transfer", "my payment", "my last", "my recent",
                                    "my balance", "my account", "i spent", "i paid", "i transferred"])

    def _execute_intent(self, intent: str, entities: dict, user_input: str) -> str:
        # Guard: reroute personal-data queries misclassified as product_search
        if intent == "product_search":
            lower = user_input.lower()
            if any(sig in lower for sig in self._PERSONAL_SIGNALS):
                if "transaction" in lower or "transfer" in lower or "payment" in lower:
                    logger.warning("intent_rerouted", extra={"from": "product_search", "to": "check_transactions", "query": user_input[:80]})
                    intent = "check_transactions"
                elif "balance" in lower:
                    logger.warning("intent_rerouted", extra={"from": "product_search", "to": "check_balance", "query": user_input[:80]})
                    intent = "check_balance"

        if intent == "block_card":
            return self._handle_block_card(entities.get("account_no"), entities)

        if intent == "general_chat":
            return ""

        mapping = self._INTENT_TO_TOOL.get(intent)
        if not mapping:
            return ""

        tool_name, args_fn = mapping
        try:
            args = args_fn(entities, user_input)
        except (TypeError, ValueError) as exc:
            logger.warning("intent_args_error", extra={"intent": intent, "error": str(exc)})
            return ""

        result = self.tool_executor.execute(tool_name, args)
        if result.startswith("Error:"):
            logger.error("tool_execution_error", extra={"tool": tool_name, "result": result[:120]})
            return ""
        return result

    # ── Card blocking (multi-step, kept separate) ─────────────────────────────

    def _handle_block_card(self, account_no: str, entities: dict) -> str:
        raw = self.tool_executor.execute("block_card", {"action": "get_active", "account_no": account_no})

        if raw == "NO_ACTIVE_CARDS" or raw.startswith("Error:"):
            return (
                "__DIRECT_RESPONSE__:I couldn't find any active cards linked to your account. "
                "Please visit a branch or call customer service for assistance."
            )

        try:
            active_cards = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            logger.error("block_card_parse_error", extra={"raw": raw[:100]})
            return "__DIRECT_RESPONSE__:Unable to retrieve card information. Please try again."

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

    # ── Confirmation handling (card blocking) ─────────────────────────────────

    _CANCEL_WORDS = frozenset({"no", "n", "cancel", "nope", "stop", "nevermind", "never", "don't", "dont", "abort"})
    _CONFIRM_WORDS = frozenset({"yes", "y", "confirm", "sure", "ok", "okay", "proceed", "go", "block"})

    def _is_confirmation_response(self, user_input: str) -> bool:
        if not self.pending_action:
            return False
        # Any reply is intercepted when awaiting card selection or confirmation —
        # this ensures we can re-prompt rather than leaking into general chat.
        return self.pending_action["type"] in {"select_card", "block_card"}

    def _handle_confirmation(self, user_input: str) -> str:
        if not self.pending_action:
            return self.run(user_input)

        # ── Card selection (multi-card scenario) ──────────────────────────────
        if self.pending_action["type"] == "select_card":
            args = self.pending_action["args"]
            text = user_input.lower()
            first_word = text.strip().split()[0] if text.strip() else ""

            # Allow customer to cancel at the card-selection stage
            if first_word in self._CANCEL_WORDS:
                self.pending_action = None
                response = "Card blocking has been cancelled. Your cards remain active. Is there anything else I can help you with?"
                self.memory.add("user", user_input)
                self.memory.add("assistant", response)
                return response

            # Match by last-four digits (most specific) first, then issuer/type
            last_four_match = next((c for c in args["active_cards"] if c["card_last_four"] in text), None)
            if last_four_match:
                matched = last_four_match
            else:
                # Collect all cards whose issuer or type appears in the text
                broad_matches = [
                    c for c in args["active_cards"]
                    if c["card_issuer"].lower() in text or c["card_type"].lower() in text
                ]
                if len(broad_matches) == 1:
                    matched = broad_matches[0]
                elif len(broad_matches) > 1:
                    # Ambiguous — multiple cards share the same issuer/type keyword
                    card_list = "\n".join(
                        f"- {c['card_issuer']} {c['card_type']} ending in {c['card_last_four']}"
                        for c in broad_matches
                    )
                    response = (
                        f"I found {len(broad_matches)} cards matching that description. "
                        f"Please specify the last 4 digits to avoid blocking the wrong card:\n{card_list}"
                    )
                    self.memory.add("user", user_input)
                    self.memory.add("assistant", response)
                    return response
                else:
                    matched = None

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

        first_word = user_input.lower().strip().split()[0] if user_input.strip() else ""

        if self.pending_action["type"] == "block_card":
            args = self.pending_action["args"]

            is_confirmed = first_word in self._CONFIRM_WORDS
            is_declined = first_word in self._CANCEL_WORDS

            if not is_confirmed and not is_declined:
                # Unrecognised reply — re-prompt without clearing pending_action
                response = (
                    f"Sorry, I didn't catch that. Please reply:\n"
                    f"- \"Yes\" to confirm blocking your {args['card_type']} ending in {args['card_last_four']}\n"
                    f"- \"No\" to cancel"
                )
                self.memory.add("user", user_input)
                self.memory.add("assistant", response)
                return response

            # Clear pending_action only after we know the intent
            self.pending_action = None

            if is_confirmed:
                result = self.tool_executor.execute("block_card", {
                    "action": "block",
                    "account_no": args["account_no"],
                    "card_last_four": args["card_last_four"],
                    "reason": args["reason"],
                    "confirmed": True,
                })
                # Tool returns "SUCCESS\n{message}" on success, plain message otherwise
                response = result.removeprefix("SUCCESS\n") if result.startswith("SUCCESS\n") else result
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

    def get_token_usage(self) -> dict:
        """Return actual LLM token counts (prompt + completion) for this session."""
        return self.llm.get_token_usage()
