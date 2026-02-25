from dataclasses import dataclass
from typing import Callable


@dataclass
class ToolParam:
    name: str
    type: str
    description: str
    required: bool = True
    enum: list[str] | None = None


@dataclass
class Tool:
    name: str
    description: str
    parameters: list[ToolParam]
    handler: Callable


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def format_for_prompt(self) -> str:
        lines = ["Available tools:"]
        for tool in self._tools.values():
            params = ", ".join(
                f"{p.name}: {p.type}" + (f" (options: {p.enum})" if p.enum else "")
                for p in tool.parameters
            )
            lines.append(f"- {tool.name}({params}): {tool.description}")
        return "\n".join(lines)
