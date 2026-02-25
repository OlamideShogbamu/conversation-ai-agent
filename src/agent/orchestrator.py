from src.inference.llm import LLMEngine
from src.memory.conversation import ConversationMemory
from src.tools.registry import ToolRegistry
from src.tools.executor import ToolExecutor
from src.agent.prompts import build_prompt, MULTI_CARD_CLARIFICATION, CARD_BLOCK_CONFIRMATION
from config.settings import MAX_AGENT_ITERATIONS


class AgentOrchestrator:
    def __init__(
        self,
        llm: LLMEngine,
        tool_registry: ToolRegistry,
    ):
        self.llm = llm
        self.memory = ConversationMemory()
        self.tool_executor = ToolExecutor(tool_registry)
        self.tool_registry = tool_registry

        # State for multi-step operations
        self.pending_action = None

    def run(self, user_input: str) -> str:
        # Handle confirmation responses
        if self._is_confirmation_response(user_input):
            return self._handle_confirmation(user_input)

        self.memory.add("user", user_input)

        for iteration in range(MAX_AGENT_ITERATIONS):
            prompt = build_prompt(
                tool_descriptions=self.tool_registry.format_for_prompt(),
                conversation_history=self.memory.format_for_prompt(),
            )

            response = self.llm.generate(
                prompt,
                stop=["User:", "\nUser:"],
            )

            # Check for tool call
            if self.tool_executor.has_tool_call(response):
                parsed = self.tool_executor.parse_tool_call(response)
                if parsed:
                    tool_name, args = parsed
                    tool_result = self.tool_executor.execute(tool_name, args)

                    # Handle special tool responses
                    if tool_name == "block_card":
                        handled, user_response = self._handle_block_card_result(
                            tool_result, args
                        )
                        if handled:
                            self.memory.add("assistant", user_response)
                            return user_response

                    self.memory.add(
                        "tool", tool_result, tool_name=tool_name, tool_args=args
                    )
                    # Continue loop to process tool result
                    continue

            # No tool call - this is the final response
            self.memory.add("assistant", response)
            return response

        # Max iterations reached
        final_response = "I apologize, but I'm having trouble processing your request. Please try rephrasing or contact our customer service for assistance."
        self.memory.add("assistant", final_response)
        return final_response

    def _is_confirmation_response(self, user_input: str) -> bool:
        """Check if user input is a confirmation to a pending action."""
        if not self.pending_action:
            return False
        normalized = user_input.lower().strip()
        return normalized in ["yes", "no", "confirm", "cancel", "y", "n"]

    def _handle_confirmation(self, user_input: str) -> str:
        """Handle user confirmation for pending actions."""
        if not self.pending_action:
            return self.run(user_input)

        normalized = user_input.lower().strip()
        is_confirmed = normalized in ["yes", "confirm", "y"]

        if self.pending_action["type"] == "block_card":
            if is_confirmed:
                # Execute the block with confirmation
                args = self.pending_action["args"]
                args["confirmed"] = True
                result = self.tool_executor.execute("block_card", args)

                self.pending_action = None

                if "SUCCESS" in result:
                    response = result.replace("SUCCESS\n", "")
                    self.memory.add("user", user_input)
                    self.memory.add("assistant", response)
                    return response
                else:
                    self.memory.add("user", user_input)
                    self.memory.add("assistant", result)
                    return result
            else:
                self.pending_action = None
                response = "Card blocking has been cancelled. Your card remains active. Is there anything else I can help you with?"
                self.memory.add("user", user_input)
                self.memory.add("assistant", response)
                return response

        self.pending_action = None
        return self.run(user_input)

    def _handle_block_card_result(
        self, tool_result: str, args: dict
    ) -> tuple[bool, str]:
        """Handle special responses from block_card tool."""

        if tool_result.startswith("MULTIPLE_CARDS:"):
            card_list = tool_result.replace("MULTIPLE_CARDS:", "")
            response = MULTI_CARD_CLARIFICATION.format(card_list=card_list)
            return True, response

        elif tool_result.startswith("CONFIRM_REQUIRED"):
            # Store pending action for confirmation
            self.pending_action = {
                "type": "block_card",
                "args": args,
            }
            lines = tool_result.split("\n")
            card_info = lines[1] if len(lines) > 1 else "your card"
            reason = lines[2] if len(lines) > 2 else "Not specified"

            response = CARD_BLOCK_CONFIRMATION.format(
                card_type=args.get("card_type", ""),
                last_four=args.get("card_last_four", ""),
                reason=args.get("reason", "Not specified"),
            )
            return True, response

        elif tool_result.startswith("SUCCESS"):
            response = tool_result.replace("SUCCESS\n", "")
            return True, response

        elif tool_result == "CUSTOMER_NOT_FOUND":
            response = "I couldn't find your account. Please verify your customer ID and try again, or visit a branch for assistance."
            return True, response

        elif tool_result == "NO_ACTIVE_CARDS":
            response = "There are no active cards linked to your account that can be blocked. If you believe this is an error, please visit a branch."
            return True, response

        return False, ""

    def reset(self):
        """Reset conversation and pending actions."""
        self.memory.clear()
        self.pending_action = None

    def get_conversation_summary(self) -> str:
        """Get a summary of the current conversation context."""
        return self.memory.summary

    def get_token_usage(self) -> int:
        """Get estimated token usage of current conversation."""
        return self.memory.get_token_usage()
