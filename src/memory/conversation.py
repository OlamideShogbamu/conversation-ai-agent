from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Message:
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    tool_name: str | None = None
    tool_args: dict | None = None
    token_estimate: int = 0

    def __post_init__(self):
        # Rough token estimate: ~4 chars per token
        self.token_estimate = len(self.content) // 4


@dataclass
class ConversationMemory:
    """
    Sliding window conversation memory with token-aware pruning.
    Maintains natural conversational context by keeping recent exchanges.
    """

    messages: list[Message] = field(default_factory=list)
    max_tokens: int = 2048  # Reserve tokens for conversation history
    max_turns: int = 10  # Maximum conversation turns to keep
    summary: str = ""  # Summary of pruned conversation

    def add(self, role: str, content: str, **kwargs):
        msg = Message(role=role, content=content, **kwargs)
        self.messages.append(msg)
        self._prune_to_window()

    def _prune_to_window(self):
        """Prune old messages while maintaining context coherence."""
        # Always keep system messages
        system_msgs = [m for m in self.messages if m.role == "system"]
        other_msgs = [m for m in self.messages if m.role != "system"]

        # Calculate current token usage
        total_tokens = sum(m.token_estimate for m in other_msgs)

        # Prune if over token limit or turn limit
        while (total_tokens > self.max_tokens or len(other_msgs) > self.max_turns * 2) and len(other_msgs) > 2:
            # Remove oldest non-system message pair (user + assistant)
            removed = other_msgs.pop(0)
            total_tokens -= removed.token_estimate

            # Update summary with removed context
            if removed.role == "user":
                self._update_summary(removed.content)

        self.messages = system_msgs + other_msgs

    def _update_summary(self, content: str):
        """Maintain a brief summary of pruned conversation for context."""
        # Keep summary concise - just key topics
        if len(self.summary) > 200:
            self.summary = self.summary[-150:]
        if content:
            topic = content[:50].strip()
            if topic:
                self.summary += f" {topic}..."

    def get_history(self) -> list[Message]:
        return self.messages.copy()

    def get_recent_context(self, n_turns: int = 3) -> list[Message]:
        """Get the most recent n conversation turns for immediate context."""
        non_system = [m for m in self.messages if m.role != "system"]
        return non_system[-(n_turns * 2) :]

    def format_for_prompt(self) -> str:
        """Format conversation history for prompt injection."""
        lines = []

        # Include summary of earlier context if available
        if self.summary:
            lines.append(f"[Earlier context: {self.summary.strip()}]")
            lines.append("")

        for msg in self.messages:
            if msg.role == "system":
                continue  # System prompt handled separately
            elif msg.role == "user":
                lines.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                lines.append(f"Assistant: {msg.content}")
            elif msg.role == "tool":
                lines.append(
                    f"<tool_result name=\"{msg.tool_name}\">\n{msg.content[:500]}\n</tool_result>\n"
                    f"Using only the data above, now respond to the customer in plain English:"
                )

        return "\n".join(lines)

    def get_last_user_message(self) -> str | None:
        """Get the most recent user message."""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return None

    def get_last_assistant_message(self) -> str | None:
        """Get the most recent assistant message."""
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                return msg.content
        return None

    def has_pending_confirmation(self) -> bool:
        """Check if the last assistant message was asking for confirmation."""
        last = self.get_last_assistant_message()
        if last:
            return any(
                phrase in last.lower()
                for phrase in ["confirm", "yes or no", "would you like to proceed", "reply yes"]
            )
        return False

    def clear(self):
        """Clear conversation but keep system messages."""
        self.messages = [m for m in self.messages if m.role == "system"]
        self.summary = ""

    def get_token_usage(self) -> int:
        """Get estimated token usage of current conversation."""
        return sum(m.token_estimate for m in self.messages)
