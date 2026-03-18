import json
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from backend.agent.config_ex.model_config import get_runtime_model_settings
from backend.agent.fallback_ex.tool_fallback_planner import ToolFallbackPlanner
from backend.agent.state_ex.agent_state import AgentState
from backend.agent.stream_ex.image_buffer import ImageMarkdownBuffer
from backend.agent.stream_ex.openai_chunk_utils import merge_tool_call_delta, openai_chunk
from backend.agent.tool_ex.state_tool_runner import execute_tool_call_into_state
from backend.agent.tool_ex.tools import analyze_image, rag_image_search, save_user_fact, save_user_image, web_read, web_search
from backend.agent.util_ex.common import coerce_top_k, get_langchain_chat_model


def stream_chat_with_tools(state: AgentState, chat_id: str, created: int, max_rounds: int = 6) -> Any:
    q = (state.user_input or "").strip()
    if not q and state.image_data:
        q = "请分析这张图片"

    model_settings = get_runtime_model_settings()
    model = (model_settings.get("base_model") or "doubao-seed-2-0-lite-260215").strip()
    if not model:
        model = "doubao-seed-2-0-lite-260215"

    system_prompt = """你是一个智能助手，具备以下工具能力：

可用工具：
1. rag_image_search: 基于文本搜索相关图片（使用Milvus向量数据库）
   - 参数: query (str), top_k (int, 默认5)
   
2. web_search: 进行网页搜索获取信息（使用DuckDuckGo）
   - 参数: query (str), max_results (int, 默认5)

3. web_read: 读取指定URL的网页内容（使用Metaso Web Reader）
   - 参数: url (str), format (json|markdown, 默认markdown)
   
4. analyze_image: 分析图片内容（使用CLIP模型）
   - 参数: image_data_base64 (str), description (str)

5. save_user_fact: 保存用户长期事实记忆
   - 参数: fact (str), visibility (private|department), dept_id (str)

6. save_user_image: 保存当前会话上传图片到图片记忆库
   - 参数: description (str), visibility (private|department), dept_id (str)

使用规则：
- 根据用户需求选择合适的工具
- 可以组合使用多个工具
- 工具调用后，必须基于工具结果生成最终回答
- 需要网页详情时：先 web_search，再对选中的链接调用 web_read
- **重要：如果要在回答中展示图片，必须严格按照以下Markdown格式输出：`![图片名称](图片完整名称.png)`。请确保图片名称两边有中括号`[]`，图片链接两边有小括号`()`，且必须包含闭合的右括号`)`，绝不能省略后缀名，也不能使用HTML标签。**"""

    tools: List[Any] = []
    if state.use_rag:
        tools.append(rag_image_search)
    if state.use_search:
        tools.append(web_search)
        tools.append(web_read)
    if state.image_data:
        tools.append(analyze_image)
        tools.append(save_user_image)
    tools.append(save_user_fact)

    user_content = q
    if state.image_data:
        user_content = f"{user_content}\n[用户上传了一张图片，请先调用 analyze_image 工具做分析。只有当用户明确要求“记住/保存这张图”时，才调用 save_user_image 保存到图片记忆库。]"

    lc_messages: List[Any] = [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
    answer_parts: List[str] = []

    api_key = (model_settings.get("api_key") or "").strip()
    if not api_key:
        fallback = ToolFallbackPlanner(coerce_top_k=coerce_top_k, execute_tool_call=execute_tool_call_into_state, openai_chunk=openai_chunk)
        if not (bool(getattr(state, "use_search", False)) or bool(getattr(state, "use_rag", False)) or bool(getattr(state, "image_data", None))):
            state.answer = q or "OK"
            for i in range(0, len(state.answer), 64):
                chunk = state.answer[i : i + 64]
                if chunk:
                    yield openai_chunk(chat_id, created, model, {"content": chunk})
            if state.metadata is None:
                state.metadata = {}
            state.metadata["model"] = "no_key_fallback"
            return
        for obj in fallback.stream_if_needed(state, q, chat_id=chat_id, created=created, model=model):
            yield obj
        if state.metadata is None:
            state.metadata = {}
        state.metadata["model"] = "no_key_fallback"
        return

    llm = get_langchain_chat_model(model, temperature=0.7, max_tokens=1000, streaming=True)
    if tools:
        llm = llm.bind_tools(tools)

    for _ in range(coerce_top_k(max_rounds, default=6, min_value=1, max_value=12)):
        pending_tool_calls: Dict[int, Dict[str, Any]] = {}
        finish_reason: Optional[str] = None
        llm_t0 = time.time()
        image_buffer = ImageMarkdownBuffer()
        print(f"[DEBUG] ImageMarkdownBuffer initialized for round {_}")

        for chunk in llm.stream(lc_messages):
            try:
                content = getattr(chunk, "content", None)
                if isinstance(content, str) and content:
                    answer_parts.append(content)
                    buffered = image_buffer.process(content)
                    if buffered:
                        yield openai_chunk(chat_id, created, model, {"content": buffered})

                ak = getattr(chunk, "additional_kwargs", None)
                if isinstance(ak, dict):
                    tc_delta = ak.get("tool_calls")
                    if isinstance(tc_delta, list) and tc_delta:
                        merge_tool_call_delta(pending_tool_calls, tc_delta)
                        yield openai_chunk(chat_id, created, model, {"tool_calls": tc_delta})

                rm = getattr(chunk, "response_metadata", None)
                if isinstance(rm, dict):
                    fr = rm.get("finish_reason")
                    if isinstance(fr, str) and fr:
                        finish_reason = fr
                    usage = rm.get("usage")
                    if usage:
                        if state.metadata is None:
                            state.metadata = {}
                        if isinstance(usage, dict):
                            state.metadata["usage"] = usage
            except Exception:
                continue

        remaining = image_buffer.flush()
        if remaining:
            yield openai_chunk(chat_id, created, model, {"content": remaining})

        if state.timing is None:
            state.timing = {}
        state.timing["llm"] = float(time.time() - llm_t0)

        if pending_tool_calls:
            tool_calls = [pending_tool_calls[i] for i in sorted(pending_tool_calls.keys())]
            for idx, tc in enumerate(tool_calls):
                if not (tc.get("id") or "").strip():
                    tc["id"] = f"call_{int(time.time() * 1000)}_{idx}"
                fn = tc.get("function")
                if not isinstance(fn, dict):
                    fn = {"name": "unknown", "arguments": "{}"}
                    tc["function"] = fn
                if not (fn.get("name") or "").strip():
                    fn["name"] = "unknown"
                if not isinstance(fn.get("arguments"), str):
                    fn["arguments"] = json.dumps(fn.get("arguments") or {}, ensure_ascii=False)

            lc_messages.append(AIMessage(content="", additional_kwargs={"tool_calls": tool_calls}))

            for tc in tool_calls:
                tool_event = execute_tool_call_into_state(
                    state,
                    {
                        "id": tc.get("id"),
                        "type": "function",
                        "function": {"name": (tc.get("function") or {}).get("name"), "arguments": (tc.get("function") or {}).get("arguments")},
                    },
                )
                slim = dict(tool_event or {})
                r = str(slim.get("result") or "")
                if len(r) > 800:
                    r = r[:800] + "..."
                slim["result_preview"] = r
                slim.pop("result", None)
                yield {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [],
                    "x_tool_event": slim,
                }

                tool_content = str(tool_event.get("result") if isinstance(tool_event, dict) else "")
                lc_messages.append(ToolMessage(content=tool_content, tool_call_id=str(tc.get("id") or "")))

            continue

        if finish_reason == "stop":
            break
        if finish_reason and finish_reason != "tool_calls":
            break

    state.answer = ("".join(answer_parts)).strip()
    fallback = ToolFallbackPlanner(coerce_top_k=coerce_top_k, execute_tool_call=execute_tool_call_into_state, openai_chunk=openai_chunk)
    for obj in fallback.stream_if_needed(state, q, chat_id=chat_id, created=created, model=model):
        yield obj

    if state.metadata is None:
        state.metadata = {}
    state.metadata["model"] = model
