from typing import Any, Dict

from backend.agent.tool_ex.execution_context import ToolExecutionContext


class WebReadStrategy:
    tool_name = "web_read"

    async def execute(self, state: Any, tool_args: Dict[str, Any], ctx: ToolExecutionContext) -> Any:
        u = (tool_args.get("url") or "").strip()
        fmt = (tool_args.get("format") or "markdown").strip().lower()
        if fmt not in ("json", "markdown"):
            fmt = "markdown"
        return await ctx.web_read_invoke({"url": u, "format": fmt})

