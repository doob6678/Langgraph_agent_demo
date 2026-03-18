import os
import time
from typing import Any, Dict

from langchain_core.messages import AIMessage

from backend.agent.service_ex.agent_services import milvus_service, search_service
from backend.agent.state_ex.agent_state import AgentState
from backend.agent.tool_ex.execution_context import ToolExecutionContext
from backend.agent.tool_ex.tool_executor import ToolExecutor
from backend.agent.tool_ex.tools import analyze_image, rag_image_search, save_user_fact, save_user_image, web_read, web_search
from backend.agent.util_ex.common import coerce_top_k, safe_parse_tool_arguments


_tool_executor = ToolExecutor()


def _append_tool_trace(
    state: AgentState,
    tool_name: str,
    tool_args: Dict[str, Any],
    ok: bool,
    elapsed_s: float,
    result_preview: str,
) -> None:
    if state.metadata is None:
        state.metadata = {}
    trace = state.metadata.get("tool_trace")
    if not isinstance(trace, list):
        trace = []
        state.metadata["tool_trace"] = trace
    trace.append(
        {
            "tool": tool_name,
            "args": tool_args if isinstance(tool_args, dict) else {"_raw": str(tool_args)},
            "ok": bool(ok),
            "elapsed_s": float(elapsed_s),
            "result_preview": result_preview,
        }
    )


async def execute_tool_call_into_state(state: AgentState, tool_call: Any) -> Dict[str, Any]:
    ctx = ToolExecutionContext(
        milvus_service=milvus_service,
        search_service=search_service,
        rag_image_search_invoke=rag_image_search.ainvoke,
        web_search_invoke=web_search.ainvoke,
        web_read_invoke=web_read.ainvoke,
        analyze_image_invoke=analyze_image.ainvoke,
        save_user_fact_invoke=save_user_fact.ainvoke,
        save_user_image_invoke=save_user_image.ainvoke,
        coerce_top_k=coerce_top_k,
        getenv=os.getenv,
    )
    return await _tool_executor.execute_tool_call_into_state(
        state,
        tool_call,
        ctx=ctx,
        safe_parse_tool_arguments=safe_parse_tool_arguments,
        append_tool_trace=_append_tool_trace,
        ai_message_factory=AIMessage,
    )


async def process_tool_results(state: AgentState) -> AgentState:
    tool_calls = getattr(state, "tool_calls", None)
    if isinstance(tool_calls, list) and tool_calls:
        for tool_call in tool_calls:
            await execute_tool_call_into_state(state, tool_call)
    state.needs_tool = False
    state.tool_calls = []
    return state

