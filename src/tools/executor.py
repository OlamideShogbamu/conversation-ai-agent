import json
import re
from src.tools.registry import ToolRegistry


class ToolExecutor:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def parse_tool_call(self, text: str) -> tuple[str, dict] | None:
        """Parse tool call from LLM output. Expected format: TOOL: name(arg1=val1, arg2=val2)"""
        pattern = r"TOOL:\s*(\w+)\((.*?)\)"
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            return None

        tool_name = match.group(1)
        args_str = match.group(2).strip()

        args = {}
        if args_str:
            # Parse key=value pairs
            for pair in re.findall(r'(\w+)=(?:"([^"]*)"|\'([^\']*)\'|(\S+))', args_str):
                key = pair[0]
                value = pair[1] or pair[2] or pair[3]
                # Try to parse as JSON for complex types
                try:
                    value = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    pass
                args[key] = value

        return tool_name, args

    def execute(self, tool_name: str, args: dict) -> str:
        tool = self.registry.get(tool_name)
        if not tool:
            return f"Error: Tool '{tool_name}' not found."

        try:
            result = tool.handler(**args)
            return str(result)
        except TypeError as e:
            return f"Error: Invalid arguments for {tool_name}: {e}"
        except Exception as e:
            return f"Error executing {tool_name}: {e}"

    def has_tool_call(self, text: str) -> bool:
        return "TOOL:" in text
