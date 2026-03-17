import re
from collections import Counter

from src.inference.llm import LLMEngine

# Simple key-value format — small models handle this far better than JSON
INTENT_PROMPT = """Classify the banking customer message. Reply with ONLY the lines below filled in. No extra text.

intents: check_balance, check_transactions, transaction_summary, account_details, account_types, product_search, loan_info, investment_info, savings_info, card_info, block_card, calculate, general_chat
chain_intent: only fill if the query clearly needs TWO steps (e.g. "check my balance and calculate if I can afford a loan" → chain_intent: calculate). Valid chain_intents: calculate, investment_info, loan_info, check_transactions. Otherwise leave blank.

Recent conversation:
{history}

Customer: {message}

intent:
account_no:
query:
limit:
loan_type:
calculation_type:
principal:
rate:
tenure_months:
card_last_four:
reason:
chain_intent: """

# Valid (primary → secondary) chains. Secondary is only attempted if its
# required entities are present — never blocks with a clarification question.
CHAINABLE_PAIRS = {
    ("check_balance", "calculate"),
    ("check_balance", "investment_info"),
    ("check_balance", "loan_info"),
    ("account_details", "check_transactions"),
}

REQUIRED_ENTITIES = {
    "check_balance": ["account_no"],
    "check_transactions": ["account_no"],
    "transaction_summary": ["account_no"],
    "account_details": ["account_no"],
    "account_types": [],
    "product_search": ["query"],
    "loan_info": [],
    "investment_info": [],
    "savings_info": [],
    "card_info": [],
    "block_card": ["account_no"],
    "calculate": ["calculation_type", "principal", "rate", "tenure_months"],
    "general_chat": [],
}

CLARIFICATION_QUESTIONS = {
    "account_no": "Could you please provide your account number?",
    "query": "What product or service would you like to know about?",
    "calculation_type": "What would you like to calculate — loan EMI, investment return, or simple interest?",
    "principal": "What is the principal amount (in Naira)?",
    "rate": "What is the annual interest rate (%)?",
    "tenure_months": "What is the tenure in months?",
}

VALID_INTENTS = set(REQUIRED_ENTITIES.keys())


class IntentExtractor:
    def __init__(self, llm: LLMEngine, n_votes: int = 3):
        self.llm = llm
        self.n_votes = n_votes

    def extract(self, message: str, history: str = "") -> tuple[str, dict, str]:
        """
        Run extraction n_votes times and return majority-voted result.
        Returns: (intent, entities, confidence)
        """
        prompt = INTENT_PROMPT.format(
            message=message,
            history=history[-400:] if history else "None",
        )

        results = []
        for i in range(self.n_votes):
            print(f"[Extractor] Vote {i + 1}/{self.n_votes}...", flush=True)
            raw = self.llm.generate(prompt, max_tokens=150, temperature=0.1, stop=["\n\n\n", "Customer:"])
            parsed = self._parse_kv(raw)
            if parsed:
                results.append(parsed)
                print(f"[Extractor] Vote {i + 1}: intent={parsed.get('intent')}", flush=True)
            else:
                print(f"[Extractor] Vote {i + 1}: parse failed, raw={repr(raw[:80])}", flush=True)

        if not results:
            # All parses failed — fall back to general_chat, proceed without clarifying
            return "general_chat", {}, "medium"

        return self._majority_vote(results)

    def get_missing_entities(self, intent: str, entities: dict) -> list[str]:
        """Return required entity keys that are missing."""
        required = REQUIRED_ENTITIES.get(intent, [])
        return [key for key in required if not entities.get(key)]

    def get_clarification_question(self, missing: list[str]) -> str:
        for key in missing:
            if key in CLARIFICATION_QUESTIONS:
                return CLARIFICATION_QUESTIONS[key]
        return "Could you please provide more details?"

    def _parse_kv(self, text: str) -> dict | None:
        """Parse key: value lines into a dict. Returns None if no valid intent found."""
        result = {}
        for line in text.strip().splitlines():
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip().lower()
            value = value.strip()
            if value and value.lower() not in ("none", "null", "n/a", "-", ""):
                result[key] = value

        intent = result.get("intent", "").lower().strip()
        # Validate intent is one of the known ones
        if intent not in VALID_INTENTS:
            # Try to find a valid intent anywhere in the text
            for vi in VALID_INTENTS:
                if vi in text.lower():
                    intent = vi
                    break
            else:
                return None

        entities = {k: v for k, v in result.items() if k != "intent"}

        # Validate chain_intent against allowed values; discard if invalid
        if "chain_intent" in entities:
            ci = entities["chain_intent"].lower().strip()
            valid_chains = {"calculate", "investment_info", "loan_info", "check_transactions"}
            entities["chain_intent"] = ci if ci in valid_chains else None
            if entities["chain_intent"] is None:
                del entities["chain_intent"]

        # Coerce numeric fields
        for num_field in ("limit", "principal", "rate", "tenure_months"):
            if num_field in entities:
                try:
                    entities[num_field] = float(entities[num_field])
                except ValueError:
                    del entities[num_field]

        return {"intent": intent, "entities": entities}

    def _majority_vote(self, results: list[dict]) -> tuple[str, dict, str]:
        intents = [r.get("intent", "general_chat") for r in results]
        intent_counts = Counter(intents)
        top_intent, top_count = intent_counts.most_common(1)[0]

        if top_count == self.n_votes:
            confidence = "high"
        elif top_count > self.n_votes / 2:
            confidence = "medium"
        else:
            confidence = "low"

        # Merge entities from winning votes
        winning = [r for r in results if r.get("intent") == top_intent]
        all_keys = set()
        for r in winning:
            all_keys.update(r.get("entities", {}).keys())

        merged = {}
        for key in all_keys:
            values = [r.get("entities", {}).get(key) for r in winning]
            non_null = [v for v in values if v is not None]
            if non_null:
                value_counts = Counter(str(v) for v in non_null)
                best_str = value_counts.most_common(1)[0][0]
                merged[key] = next(v for v in non_null if str(v) == best_str)

        print(f"[Extractor] Final: intent={top_intent}, confidence={confidence}, entities={merged}", flush=True)
        return top_intent, merged, confidence
