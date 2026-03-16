import time
from typing import Any, Callable, Dict, List, Optional

from backend.agent.tool_ex.analyze_image_strategy import AnalyzeImageStrategy
from backend.agent.tool_ex.execution_context import ToolExecutionContext
from backend.agent.tool_ex.rag_image_search_strategy import RagImageSearchStrategy
from backend.agent.tool_ex.tool_strategy import ToolStrategy
from backend.agent.tool_ex.web_read_strategy import WebReadStrategy
from backend.agent.tool_ex.web_search_strategy import WebSearchStrategy


class ToolExecutor:
    def __init__(self, strategies: Optional[List[ToolStrategy]] = None) -> None:
        self._strategies: List[ToolStrategy] = strategies or [
            RagImageSearchStrategy(),
            WebSearchStrategy(),
            WebReadStrategy(),
            AnalyzeImageStrategy(),
        ]

    def execute_tool_call_into_state(
        self,
        state: Any,
        tool_call: Any,
        *,
        ctx: ToolExecutionContext,
        safe_parse_tool_arguments: Callable[[Any], Dict[str, Any]],
        append_tool_trace: Callable[[Any, str, Dict[str, Any], bool, float, str], None],
        ai_message_factory: Callable[..., Any],
    ) -> Dict[str, Any]:
        if isinstance(tool_call, dict):
            tool_name = (tool_call.get("function") or {}).get("name") or tool_call.get("name") or ""
            raw_args = (tool_call.get("function") or {}).get("arguments") or tool_call.get("arguments")
        else:
            tool_name = getattr(getattr(tool_call, "function", None), "name", "") or ""
            raw_args = getattr(getattr(tool_call, "function", None), "arguments", None)

        tool_args = safe_parse_tool_arguments(raw_args)
        t0 = time.time()
        ok = True
        result: Any = ""
        try:
            strategy = None
            for s in self._strategies:
                if s.tool_name == tool_name:
                    strategy = s
                    break
            if strategy is None:
                result = f"未知工具: {tool_name}"
            else:
                result = strategy.execute(state, tool_args, ctx)
        except Exception as e:
            ok = False
            result = f"工具 {tool_name} 执行失败: {str(e)}"
            print(f"[tool] {result}")
        elapsed_s = time.time() - t0

        if getattr(state, "timing", None) is None:
            state.timing = {}
        if tool_name:
            state.timing[f"tool_{tool_name}"] = float(elapsed_s)

        if getattr(state, "tool_results", None) is None:
            state.tool_results = {}
        if tool_name:
            state.tool_results[tool_name] = result

        if getattr(state, "messages", None) is None:
            state.messages = []
        if tool_name:
            state.messages.append(ai_message_factory(content=f"工具 {tool_name} 返回: {result}"))

        preview = (str(result) or "").replace("\r\n", "\n").replace("\r", "\n")
        if len(preview) > 500:
            preview = preview[:500] + "..."
        append_tool_trace(state, tool_name or "unknown", tool_args, ok, elapsed_s, preview)

        return {"tool": tool_name or "unknown", "args": tool_args, "ok": ok, "elapsed_s": elapsed_s, "result": result}

