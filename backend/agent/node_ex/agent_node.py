import json
import os
import time
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backend.agent.state_ex.agent_state import AgentState
from backend.agent.tool_ex.tools import analyze_image, rag_image_search, web_read, web_search
from backend.agent.util_ex.common import coerce_top_k, get_langchain_chat_model, to_lc_message


def agent_node(state: AgentState) -> AgentState:
    start_time = time.time()
    try:
        api_key = (os.getenv("ARK_API_KEY") or "").strip()
        if not api_key:
            if state.timing is None:
                state.timing = {}
            if state.metadata is None:
                state.metadata = {}

            q = (state.user_input or "").strip()
            has_any_result = bool(
                getattr(state, "tool_results", None)
                or (isinstance(getattr(state, "search_results", None), list) and getattr(state, "search_results", None))
                or (isinstance(getattr(state, "images", None), list) and getattr(state, "images", None))
            )
            if has_any_result and not (getattr(state, "answer", "") or "").strip():
                parts: List[str] = []
                search_results = getattr(state, "search_results", None)
                if isinstance(search_results, list) and search_results:
                    first = search_results[0] if search_results else {}
                    title = first.get("title") if isinstance(first, dict) else ""
                    parts.append(f"联网搜索结果: {len(search_results)}条 {title}".strip())
                images = getattr(state, "images", None)
                if isinstance(images, list) and images:
                    first_img = images[0] if images else {}
                    fn = first_img.get("filename") if isinstance(first_img, dict) else ""
                    parts.append(f"RAG图片结果: {len(images)}张 {fn}".strip())
                if not parts:
                    parts.append("done")
                state.answer = " ".join(parts).strip()

            if (bool(getattr(state, "use_search", False)) or bool(getattr(state, "use_rag", False)) or bool(getattr(state, "image_data", None))) and not (
                isinstance(getattr(state, "tool_calls", None), list) and getattr(state, "tool_calls", None)
            ):
                ts = int(time.time() * 1000)
                top_k = getattr(state, "top_k", 5)
                tool_calls: List[Dict[str, Any]] = []
                if bool(getattr(state, "use_search", False)):
                    tool_calls.append(
                        {
                            "id": f"call_{ts}_search",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": json.dumps(
                                    {"query": q, "max_results": coerce_top_k(top_k, default=5, min_value=1, max_value=50)},
                                    ensure_ascii=False,
                                ),
                            },
                        }
                    )
                if bool(getattr(state, "use_rag", False)):
                    tool_calls.append(
                        {
                            "id": f"call_{ts}_rag",
                            "type": "function",
                            "function": {
                                "name": "rag_image_search",
                                "arguments": json.dumps(
                                    {"query": q, "top_k": coerce_top_k(top_k, default=5, min_value=1, max_value=20)},
                                    ensure_ascii=False,
                                ),
                            },
                        }
                    )
                image_data = getattr(state, "image_data", None)
                if image_data:
                    import base64

                    tool_calls.append(
                        {
                            "id": f"call_{ts}_img",
                            "type": "function",
                            "function": {
                                "name": "analyze_image",
                                "arguments": json.dumps(
                                    {"image_data_base64": base64.b64encode(image_data).decode("ascii"), "description": ""},
                                    ensure_ascii=False,
                                ),
                            },
                        }
                    )

                state.tool_calls = tool_calls
                state.needs_tool = bool(tool_calls)
                state.metadata["model"] = "no_key_fallback"
                state.timing["agent_time"] = time.time() - start_time
                return state

            if not (getattr(state, "answer", "") or "").strip():
                state.answer = q or "OK"
            state.needs_tool = False
            state.tool_calls = []
            state.metadata["model"] = "no_key_fallback"
            state.timing["agent_time"] = time.time() - start_time
            return state

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

使用规则：
- 根据用户需求选择合适的工具
- 可以组合使用多个工具
- 工具调用格式: 工具名(参数)
- 返回结果后，基于结果生成最终回答
- 需要网页详情时：先 web_search，再对选中的链接调用 web_read
- **重要：如果要在回答中展示图片，必须严格按照以下Markdown格式输出：`![图片名称](图片完整名称.png)`。请确保图片名称两边有中括号`[]`，图片链接两边有小括号`()`，且必须包含闭合的右括号`)`，绝不能省略后缀名，也不能使用HTML标签。**

请分析用户需求，如果需要使用工具，请明确调用。"""

        tools: List[Any] = []
        if state.use_rag:
            tools.append(rag_image_search)
        if state.use_search:
            tools.append(web_search)
            tools.append(web_read)
        if state.image_data:
            tools.append(analyze_image)

        lc_messages: List[Any] = [SystemMessage(content=system_prompt)]
        if state.messages:
            for msg in state.messages:
                lc_messages.append(to_lc_message(msg))
        else:
            user_content = state.user_input or ""
            if state.image_data:
                user_content = f"{user_content}\n[用户上传了一张图片，请调用 analyze_image 工具进行分析。你可以直接传递空的 image_data_base64 参数，系统会自动处理]"
            lc_messages.append(HumanMessage(content=user_content))

        llm = get_langchain_chat_model(
            (os.getenv("ARK_MODEL") or os.getenv("ARK_MODEL_NAME") or "doubao-seed-2-0-lite-260215").strip() or "doubao-seed-2-0-lite-260215",
            temperature=0.7,
            max_tokens=1000,
            streaming=False,
        )
        if tools:
            llm = llm.bind_tools(tools)

        resp_msg = llm.invoke(lc_messages)
        rm = getattr(resp_msg, "response_metadata", None)
        if isinstance(rm, dict):
            usage = rm.get("usage")
            if isinstance(usage, dict):
                state.metadata["usage"] = usage

        tool_calls_raw = getattr(resp_msg, "tool_calls", None)
        if not (isinstance(tool_calls_raw, list) and tool_calls_raw):
            ak = getattr(resp_msg, "additional_kwargs", None)
            if isinstance(ak, dict):
                tc2 = ak.get("tool_calls")
                if isinstance(tc2, list) and tc2:
                    tool_calls_raw = tc2

        normalized_tool_calls: List[Dict[str, Any]] = []
        if isinstance(tool_calls_raw, list) and tool_calls_raw:
            for i, tc in enumerate(tool_calls_raw):
                call_id = ""
                name = ""
                args: Any = None
                if isinstance(tc, dict):
                    call_id = str(tc.get("id") or "")
                    fn = tc.get("function")
                    if isinstance(fn, dict):
                        name = str(fn.get("name") or "")
                        args = fn.get("arguments")
                    if not name:
                        name = str(tc.get("name") or "")
                    if args is None:
                        args = tc.get("args") if "args" in tc else tc.get("arguments")
                if not call_id:
                    call_id = f"call_{int(time.time() * 1000)}_{i}"
                if not name:
                    name = "unknown"
                if isinstance(args, dict):
                    args_str = json.dumps(args, ensure_ascii=False)
                elif args is None:
                    args_str = "{}"
                else:
                    args_str = str(args)
                normalized_tool_calls.append({"id": call_id, "type": "function", "function": {"name": name, "arguments": args_str}})

        if normalized_tool_calls:
            state.messages = lc_messages + [AIMessage(content=getattr(resp_msg, "content", "") or "", additional_kwargs={"tool_calls": normalized_tool_calls})]
            state.tool_calls = normalized_tool_calls
            state.needs_tool = True
            state.tool_results = {}
        else:
            state.messages = lc_messages + [resp_msg]
            state.answer = (getattr(resp_msg, "content", "") or "").strip()
            state.needs_tool = False

        state.timing["agent_time"] = time.time() - start_time
        state.metadata["model"] = os.getenv("ARK_MODEL_NAME", "doubao-seed-2-0-lite-260215")
        return state
    except Exception as e:
        state.answer = f"处理请求时出错: {str(e)}"
        state.timing["agent_time"] = time.time() - start_time
        return state
