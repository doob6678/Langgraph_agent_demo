from typing import Any, Dict, Protocol

from backend.agent.tool_ex.execution_context import ToolExecutionContext


class ToolStrategy(Protocol):
    tool_name: str

    async def execute(self, state: Any, tool_args: Dict[str, Any], ctx: ToolExecutionContext) -> Any: ...

