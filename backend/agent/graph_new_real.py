import os
import time
import json
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from openai import OpenAI
from dotenv import load_dotenv

from backend.services.milvus_service import MilvusService
from backend.services.search_service import SearchService
from backend.services.image_service import image_service
from backend.services.metrics_service import metrics_collector

load_dotenv()

# 全局服务实例
milvus_service = MilvusService()
search_service = SearchService()

def _safe_parse_tool_arguments(raw_args: Any) -> Dict[str, Any]:
    if raw_args is None:
        return {}
    if isinstance(raw_args, dict):
        return raw_args
    if not isinstance(raw_args, str):
        return {"_raw": raw_args}

    s = raw_args.strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        import ast

        try:
            v = ast.literal_eval(s)
            return v if isinstance(v, dict) else {"_raw": v}
        except Exception:
            return {"_raw": raw_args}

def _coerce_top_k(value: Any, default: int = 5, min_value: int = 1, max_value: int = 20) -> int:
    try:
        k = int(value)
    except Exception:
        k = default
    if k < min_value:
        return min_value
    if k > max_value:
        return max_value
    return k

def _bing_text_search(query: str, max_results: int) -> List[Dict[str, Any]]:
    import re
    import html as _html
    import requests
    from urllib.parse import quote_plus

    q = (query or "").strip()
    if not q:
        return []

    timeout_s = 10
    try:
        timeout_s = int((os.getenv("SEARCH_TIMEOUT") or "10").strip() or "10")
    except Exception:
        timeout_s = 10
    if timeout_s < 3:
        timeout_s = 3
    if timeout_s > 30:
        timeout_s = 30

    def _fetch(url: str) -> str:
        resp = requests.get(
            url,
            timeout=timeout_s,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            },
        )
        resp.raise_for_status()
        return resp.text or ""

    base = "https://www.bing.com/search?q="
    html_text = _fetch(f"{base}{quote_plus(q)}&setlang=zh-hans&cc=CN")
    if "b_results" not in html_text:
        html_text = _fetch(f"{base}{quote_plus(q)}")

    blocks = re.findall(
        r'<li[^>]+class="[^"]*\bb_algo\b[^"]*"[^>]*>.*?</li>',
        html_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    results: List[Dict[str, Any]] = []
    for b in blocks:
        m = re.search(
            r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            b,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if not m:
            continue
        link = _html.unescape(m.group(1)).strip()
        title_raw = re.sub(r"<.*?>", "", m.group(2), flags=re.DOTALL)
        title = _html.unescape(title_raw).strip()

        sm = re.search(r'<div class="b_caption".*?<p>(.*?)</p>', b, flags=re.DOTALL | re.IGNORECASE)
        snippet_raw = re.sub(r"<.*?>", "", sm.group(1), flags=re.DOTALL) if sm else ""
        snippet = _html.unescape(snippet_raw).strip()

        results.append({"title": title, "link": link, "snippet": snippet, "source": "bing"})
        if len(results) >= max_results:
            break
    return results

def _ddg_text_search(query: str, max_results: int) -> List[Dict[str, Any]]:
    from duckduckgo_search import DDGS

    q = (query or "").strip()
    if not q:
        return []

    backends = []
    preferred = (os.getenv("DDG_BACKEND") or "").strip().lower()
    if preferred:
        backends.append(preferred)
    backends.extend(["lite", "html", "api", "auto"])

    seen = set()
    backend_order = []
    for b in backends:
        if b and b not in seen:
            seen.add(b)
            backend_order.append(b)

    last_error: Optional[Exception] = None
    for backend in backend_order:
        try:
            ddgs = DDGS()
            results: List[Dict[str, Any]] = []
            search_results = ddgs.text(
                q,
                region=os.getenv("DDG_REGION", "wt-wt"),
                max_results=max_results,
                backend=backend,
            )
            for r in search_results:
                results.append(
                    {
                        "title": r.get("title", ""),
                        "link": r.get("href", ""),
                        "snippet": r.get("body", ""),
                        "source": f"duckduckgo:{backend}",
                    }
                )
            if results:
                return results
            continue
        except Exception as e:
            last_error = e
            continue

    if last_error:
        provider = (os.getenv("WEB_SEARCH_PROVIDER") or "bing").strip().lower()
        if provider in ("bing", "bing_html", "bing_scrape", "default"):
            return _bing_text_search(q, max_results)
        raise last_error
    return _bing_text_search(q, max_results)

@dataclass
class AgentState:
    """Agent状态"""
    messages: List[Any]  # 支持 HumanMessage 和 AIMessage
    user_input: str = ""
    image_data: Optional[bytes] = None
    image_filename: Optional[str] = None
    use_rag: bool = True
    use_search: bool = True
    top_k: int = 5
    answer: str = ""
    images: List[Dict[str, Any]] = None
    search_results: List[Dict[str, Any]] = None
    timing: Dict[str, float] = None
    metadata: Dict[str, Any] = None
    tool_calls: List[Dict[str, Any]] = None
    needs_tool: bool = False
    current_tool: str = ""
    tool_results: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.images is None:
            self.images = []
        if self.search_results is None:
            self.search_results = []
        if self.timing is None:
            self.timing = {}
        if self.metadata is None:
            self.metadata = {}
        if self.tool_calls is None:
            self.tool_calls = []
        if self.tool_results is None:
            self.tool_results = {}

# 初始化OpenAI客户端
client = OpenAI(
    api_key=os.getenv("ARK_API_KEY"),
    base_url=os.getenv("ARK_BASE_URL", os.getenv("ARK_API_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")),
)
_client_lock = threading.Lock()

def configure_ark(api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
    global client
    k = (api_key or "").strip() if api_key is not None else (os.getenv("ARK_API_KEY") or "").strip()
    u = (base_url or "").strip() if base_url is not None else (os.getenv("ARK_BASE_URL") or os.getenv("ARK_API_BASE_URL") or "").strip()
    m = (model or "").strip() if model is not None else (os.getenv("ARK_MODEL") or os.getenv("ARK_MODEL_NAME") or "").strip()

    if u:
        os.environ["ARK_BASE_URL"] = u
    if m:
        os.environ["ARK_MODEL"] = m
    if api_key is not None:
        if k:
            os.environ["ARK_API_KEY"] = k
        else:
            try:
                os.environ.pop("ARK_API_KEY", None)
            except Exception:
                pass

    with _client_lock:
        client = OpenAI(
            api_key=(os.getenv("ARK_API_KEY") or "").strip(),
            base_url=os.getenv("ARK_BASE_URL", os.getenv("ARK_API_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")),
        )

    key_set = bool((os.getenv("ARK_API_KEY") or "").strip())
    key_val = (os.getenv("ARK_API_KEY") or "").strip()
    masked = (key_val[:6] + "..." + key_val[-4:]) if key_val and len(key_val) >= 12 else ("set" if key_set else "")
    return {"ark_api_key": masked, "ark_base_url": (os.getenv("ARK_BASE_URL") or "").strip(), "ark_model": (os.getenv("ARK_MODEL") or "").strip(), "key_set": key_set}

def _should_force_rule_based() -> bool:
    ark_key = (os.getenv("ARK_API_KEY") or "").strip()
    agent_mode = (os.getenv("AGENT_MODE") or "").strip().lower()
    force_rule = agent_mode in ("rule", "rule_based", "deterministic", "local", "offline")
    return bool(force_rule or (not ark_key))

def _openai_chunk(chat_id: str, created: int, model: str, delta: Dict[str, Any], finish_reason: Optional[str] = None) -> Dict[str, Any]:
    return {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }

def _merge_tool_call_delta(pending: Dict[int, Dict[str, Any]], tool_calls_delta: Any) -> None:
    if not isinstance(tool_calls_delta, list):
        return
    for tc in tool_calls_delta:
        if not isinstance(tc, dict):
            continue
        try:
            idx = int(tc.get("index", 0))
        except Exception:
            idx = 0
        cur = pending.get(idx)
        if not isinstance(cur, dict):
            cur = {"index": idx, "id": "", "type": "function", "function": {"name": "", "arguments": ""}}
            pending[idx] = cur
        if not cur.get("id") and isinstance(tc.get("id"), str):
            cur["id"] = tc.get("id") or ""
        if isinstance(tc.get("type"), str) and tc.get("type"):
            cur["type"] = tc.get("type")
        fn = tc.get("function")
        if isinstance(fn, dict):
            fcur = cur.get("function")
            if not isinstance(fcur, dict):
                fcur = {"name": "", "arguments": ""}
                cur["function"] = fcur
            if isinstance(fn.get("name"), str) and fn.get("name"):
                fcur["name"] = fn.get("name") or ""
            if isinstance(fn.get("arguments"), str) and fn.get("arguments"):
                fcur["arguments"] = (fcur.get("arguments") or "") + (fn.get("arguments") or "")

def stream_chat_with_tools(state: AgentState, chat_id: str, created: int, max_rounds: int = 6) -> Any:
    q = (state.user_input or "").strip()
    if not q and state.image_data:
        q = "请分析这张图片"

    model = (os.getenv("ARK_MODEL") or os.getenv("ARK_MODEL_NAME") or "doubao-seed-2-0-lite-260215").strip()
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

使用规则：
- 根据用户需求选择合适的工具
- 可以组合使用多个工具
- 工具调用后，必须基于工具结果生成最终回答
- 需要网页详情时：先 web_search，再对选中的链接调用 web_read"""

    tools_schema: List[Dict[str, Any]] = []
    if state.use_rag:
        tools_schema.append(
            {
                "type": "function",
                "function": {
                    "name": "rag_image_search",
                    "description": "基于文本搜索相关图片",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "搜索查询"},
                            "top_k": {"type": "integer", "description": "返回结果数量", "default": 5},
                        },
                        "required": ["query"],
                    },
                },
            }
        )
    if state.use_search:
        tools_schema.append(
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "进行网页搜索获取信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "搜索查询"},
                            "max_results": {"type": "integer", "description": "返回结果数量", "default": 5},
                        },
                        "required": ["query"],
                    },
                },
            }
        )
        tools_schema.append(
            {
                "type": "function",
                "function": {
                    "name": "web_read",
                    "description": "读取指定URL的网页内容",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "要读取的URL地址"},
                            "format": {"type": "string", "description": "输出格式：json 或 markdown", "default": "markdown"},
                        },
                        "required": ["url"],
                    },
                },
            }
        )
    if state.image_data:
        tools_schema.append(
            {
                "type": "function",
                "function": {
                    "name": "analyze_image",
                    "description": "分析图片内容",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "image_data_base64": {"type": "string", "description": "base64编码的图片数据"},
                            "description": {"type": "string", "description": "图片描述", "default": ""},
                        },
                        "required": ["image_data_base64"],
                    },
                },
            }
        )

    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}, {"role": "user", "content": q}]
    answer_parts: List[str] = []

    for _ in range(_coerce_top_k(max_rounds, default=6, min_value=1, max_value=12)):
        pending_tool_calls: Dict[int, Dict[str, Any]] = {}
        finish_reason: Optional[str] = None
        llm_t0 = time.time()

        stream_options = None
        try:
            stream_options = {"include_usage": True}
        except Exception:
            stream_options = None

        create_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1000,
            "tools": (tools_schema if tools_schema else None),
            "stream": True,
        }
        if stream_options:
            create_kwargs["stream_options"] = stream_options
        with _client_lock:
            local_client = client
        try:
            resp_stream = local_client.chat.completions.create(**create_kwargs)
        except TypeError:
            create_kwargs.pop("stream_options", None)
            resp_stream = local_client.chat.completions.create(**create_kwargs)

        for chunk in resp_stream:
            try:
                usage_obj = getattr(chunk, "usage", None)
                if usage_obj:
                    if state.metadata is None:
                        state.metadata = {}
                    if isinstance(usage_obj, dict):
                        state.metadata["usage"] = usage_obj
                    else:
                        state.metadata["usage"] = {
                            "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
                            "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
                            "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
                        }

                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                c0 = choices[0]
                delta = getattr(c0, "delta", None)
                if delta is None:
                    continue
                content = getattr(delta, "content", None)
                if isinstance(content, str) and content:
                    answer_parts.append(content)
                    yield _openai_chunk(chat_id, created, model, {"content": content})

                tool_calls_delta = getattr(delta, "tool_calls", None)
                if tool_calls_delta:
                    tool_calls_list = []
                    for item in tool_calls_delta:
                        if hasattr(item, "model_dump"):
                            tool_calls_list.append(item.model_dump())
                        elif hasattr(item, "__dict__"):
                            tool_calls_list.append(dict(item.__dict__))
                        elif isinstance(item, dict):
                            tool_calls_list.append(item)
                    _merge_tool_call_delta(pending_tool_calls, tool_calls_list)
                    yield _openai_chunk(chat_id, created, model, {"tool_calls": tool_calls_list})

                fr = getattr(c0, "finish_reason", None)
                if isinstance(fr, str) and fr:
                    finish_reason = fr
            except Exception:
                continue

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

            messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})

            for tc in tool_calls:
                tool_event = _execute_tool_call_into_state(
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

                messages.append({"role": "tool", "tool_call_id": tc.get("id"), "content": str(tool_event.get("result") if isinstance(tool_event, dict) else "")})

            continue

        if finish_reason == "stop":
            break

        if finish_reason and finish_reason != "tool_calls":
            break

    state.answer = ("".join(answer_parts)).strip()
    state.metadata["model"] = model

def agent_node(state: AgentState) -> AgentState:
    """Agent节点：处理用户输入并决定使用哪些工具"""
    start_time = time.time()
    
    try:
        ark_key = (os.getenv("ARK_API_KEY") or "").strip()
        agent_mode = (os.getenv("AGENT_MODE") or "").strip().lower()
        force_rule = agent_mode in ("rule", "rule_based", "deterministic", "local", "offline")
        if force_rule or (not ark_key):
            q = (state.user_input or "").strip()
            if not state.tool_results and not state.answer and not state.tool_calls:
                tool_calls: List[Dict[str, Any]] = []
                if state.use_search:
                    tool_calls.append(
                        {
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": {"query": q, "max_results": _coerce_top_k(state.top_k, default=5, min_value=1, max_value=50)},
                            },
                        }
                    )
                if state.use_rag:
                    tool_calls.append(
                        {
                            "type": "function",
                            "function": {
                                "name": "rag_image_search",
                                "arguments": {"query": q, "top_k": _coerce_top_k(state.top_k, default=5, min_value=1, max_value=20)},
                            },
                        }
                    )
                if state.image_data:
                    import base64

                    tool_calls.append(
                        {
                            "type": "function",
                            "function": {
                                "name": "analyze_image",
                                "arguments": {
                                    "image_data_base64": base64.b64encode(state.image_data).decode("ascii"),
                                    "description": "",
                                },
                            },
                        }
                    )

                if tool_calls:
                    state.tool_calls = tool_calls
                    state.needs_tool = True
                    state.timing["agent_time"] = time.time() - start_time
                    state.metadata["model"] = "rule_based"
                    return state

                state.answer = q or "OK"
                state.needs_tool = False
                state.timing["agent_time"] = time.time() - start_time
                state.metadata["model"] = "rule_based"
                return state

            if state.tool_results and not state.answer:
                parts: List[str] = []
                if isinstance(state.search_results, list) and state.search_results:
                    first = state.search_results[0] if state.search_results else {}
                    title = first.get("title") if isinstance(first, dict) else ""
                    parts.append(f"联网搜索结果: {len(state.search_results)}条 {title}".strip())
                if isinstance(state.images, list) and state.images:
                    first_img = state.images[0] if state.images else {}
                    fn = first_img.get("filename") if isinstance(first_img, dict) else ""
                    parts.append(f"RAG图片结果: {len(state.images)}张 {fn}".strip())
                if not parts:
                    parts.append("done")
                state.answer = " ".join(parts)
            state.needs_tool = False
            state.tool_calls = []
            state.timing["agent_time"] = time.time() - start_time
            state.metadata["model"] = "rule_based"
            return state

        # 构建消息
        messages = []
        
        # 添加系统提示，让模型知道可以使用工具
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

请分析用户需求，如果需要使用工具，请明确调用。"""
        
        messages.append({"role": "system", "content": system_prompt})
        
        # 如果有之前的消息历史，添加它们
        if state.messages:
            for msg in state.messages:
                if isinstance(msg, dict):
                    messages.append(msg)
                elif hasattr(msg, 'content') and hasattr(msg, 'role'):
                    # AIMessage或HumanMessage对象
                    messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
                else:
                    # 其他类型的消息，转换为assistant消息
                    messages.append({
                        "role": "assistant", 
                        "content": str(msg)
                    })
        
        # 如果是第一次调用，添加用户输入
        if not state.messages:
            # 添加用户输入
            user_content = state.user_input
            if state.image_data and state.image_filename:
                # 处理图片
                image_info = image_service.process_uploaded_image(state.image_data, state.image_filename)
                if image_info["success"]:
                    user_content += f"\n[用户上传了图片: {state.image_filename}]"
            
            messages.append({"role": "user", "content": user_content})
        
        # 如果有工具结果，添加到消息中
        if state.tool_results:
            for tool_name, result in state.tool_results.items():
                messages.append({
                    "role": "assistant",
                    "content": f"使用工具 {tool_name} 的结果:\n{result}"
                })
        
        # 调试：打印要发送的消息
        print(f"[ark] messages: {json.dumps(messages, ensure_ascii=False, indent=2)}")
        
        # 验证所有消息都有role字段
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict) or "role" not in msg:
                print(f"[ark] message missing role index={i} msg={msg}")
                # 修复消息格式
                if isinstance(msg, dict):
                    msg["role"] = msg.get("role", "assistant")
                else:
                    messages[i] = {"role": "assistant", "content": str(msg)}
        
        tools_schema = []
        if state.use_rag:
            tools_schema.append(
                {
                    "type": "function",
                    "function": {
                        "name": "rag_image_search",
                        "description": "基于文本搜索相关图片",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "搜索查询"},
                                "top_k": {"type": "integer", "description": "返回结果数量", "default": 5},
                            },
                            "required": ["query"],
                        },
                    },
                }
            )
        if state.use_search:
            tools_schema.append(
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "进行网页搜索获取信息",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "搜索查询"},
                                "max_results": {"type": "integer", "description": "返回结果数量", "default": 5},
                            },
                            "required": ["query"],
                        },
                    },
                }
            )
            tools_schema.append(
                {
                    "type": "function",
                    "function": {
                        "name": "web_read",
                        "description": "读取指定URL的网页内容",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string", "description": "要读取的URL地址"},
                                "format": {
                                    "type": "string",
                                    "description": "输出格式：json 或 markdown",
                                    "default": "markdown",
                                },
                            },
                            "required": ["url"],
                        },
                    },
                }
            )
        if state.image_data:
            tools_schema.append(
                {
                    "type": "function",
                    "function": {
                        "name": "analyze_image",
                        "description": "分析图片内容",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "image_data_base64": {"type": "string", "description": "base64编码的图片数据"},
                                "description": {"type": "string", "description": "图片描述", "default": ""},
                            },
                            "required": ["image_data_base64"],
                        },
                    },
                }
            )

        # 调用模型，启用函数调用
        with _client_lock:
            local_client = client
        response = local_client.chat.completions.create(
            model=os.getenv("ARK_MODEL", os.getenv("ARK_MODEL_NAME", "doubao-seed-2-0-lite-260215")),
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            tools=tools_schema if tools_schema else None
        )

        usage = getattr(response, "usage", None)
        if usage:
            if isinstance(usage, dict):
                state.metadata["usage"] = usage
            else:
                state.metadata["usage"] = {
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                    "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
                }
        
        # 检查是否有工具调用
        if response.choices[0].message.tool_calls:
            # 有工具调用，返回需要执行工具的状态
            state.messages = messages
            state.tool_calls = response.choices[0].message.tool_calls
            state.needs_tool = True
            state.tool_results = {}  # 清空之前的结果
            
            # 添加工具调用消息到状态中 - 使用标准OpenAI消息格式
            assistant_message = response.choices[0].message
            tool_calls = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                }
                for tool_call in response.choices[0].message.tool_calls
            ]
            
            # 创建标准的assistant消息格式
            assistant_msg = {
                "role": "assistant",
                "content": assistant_message.content or ""
            }
            
            # 只有在有内容或工具调用时才添加消息
            if assistant_msg["content"] or tool_calls:
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                state.messages.append(assistant_msg)
        else:
            # 没有工具调用，直接返回回答
            assistant_message = response.choices[0].message.content
            state.messages = messages + [{"role": "assistant", "content": assistant_message}]
            state.answer = assistant_message
            state.needs_tool = False
        
        state.timing["agent_time"] = time.time() - start_time
        state.metadata["model"] = os.getenv("ARK_MODEL_NAME", "doubao-seed-2-0-lite-260215")
        
        return state
        
    except Exception as e:
        state.answer = f"处理请求时出错: {str(e)}"
        state.timing["agent_time"] = time.time() - start_time
        return state

@tool
def rag_image_search(query: str, top_k: int = 5) -> str:
    """基于文本搜索相关图片"""
    start_time = time.time()
    
    try:
        # 使用Milvus服务搜索图片
        k = _coerce_top_k(top_k, default=5, min_value=1, max_value=20)
        q = (query or "").strip()
        if not q:
            return "搜索词为空，无法进行图片检索。"
        results = milvus_service.search_images_by_text(q, k)
        
        if results:
            execution_time = time.time() - start_time
            metrics_collector.record_tool_usage("rag_image_search", execution_time)
            
            # 格式化结果
            formatted_results = []
            for i, result in enumerate(results, 1):
                filename = result.get("filename", "unknown")
                score = result.get("score", 0)
                formatted_results.append(f"{i}. {filename} (相似度: {score:.3f})")
            
            return "找到以下相关图片:\n" + "\n".join(formatted_results)
        else:
            execution_time = time.time() - start_time
            metrics_collector.record_tool_usage("rag_image_search", execution_time)
            return "未找到相关图片。"
            
    except Exception as e:
        execution_time = time.time() - start_time
        metrics_collector.record_tool_usage("rag_image_search", execution_time, success=False)
        return f"图片搜索出错: {str(e)}"

@tool
def web_search(query: str, max_results: int = 5, mode: str = "", _results: Optional[List[Dict[str, Any]]] = None) -> str:
    """进行网页搜索"""
    start_time = time.time()
    
    try:
        q = (query or "").strip()
        if not q:
            return "搜索词为空，无法进行网页搜索。"

        k = _coerce_top_k(max_results, default=5, min_value=1, max_value=50)
        results = _results if isinstance(_results, list) else search_service.search_web_sync(q, k, mode=mode or None)
        
        execution_time = time.time() - start_time
        metrics_collector.record_tool_usage("web_search", execution_time)
        
        if results:
            return search_service.format_search_results(results)
        else:
            return "未找到相关网页结果。"
            
    except Exception as e:
        execution_time = time.time() - start_time
        metrics_collector.record_tool_usage("web_search", execution_time, success=False)
        return f"网页搜索出错: {str(e)}"

@tool
def web_read(url: str, format: str = "markdown", mode: str = "") -> str:
    """读取指定URL的网页内容"""
    start_time = time.time()

    try:
        u = (url or "").strip()
        if not u:
            return "URL为空，无法读取网页内容。"
        fmt = (format or "").strip().lower()
        if fmt not in ("json", "markdown"):
            fmt = "markdown"

        content = search_service.read_webpage_sync(u, output_format=fmt, mode=mode or None)
        execution_time = time.time() - start_time
        metrics_collector.record_tool_usage("web_read", execution_time, success=bool(content))
        return content or "未读取到网页内容。"
    except Exception as e:
        execution_time = time.time() - start_time
        metrics_collector.record_tool_usage("web_read", execution_time, success=False)
        return f"网页读取出错: {str(e)}"

@tool
def analyze_image(image_data_base64: str, description: str = "") -> str:
    """分析图片内容"""
    start_time = time.time()
    
    try:
        # 解码base64图片数据
        import base64
        image_bytes = base64.b64decode(image_data_base64.split(',')[1])
        
        # 处理图片
        image_info = image_service.process_uploaded_image(image_bytes, "uploaded_image.jpg")
        
        if not image_info["success"]:
            execution_time = time.time() - start_time
            metrics_collector.record_tool_usage("analyze_image", execution_time, success=False)
            return f"图片处理失败: {image_info['error']}"
        
        # 获取图片基本信息
        img_info = image_info["data"]
        
        # 使用真正的CLIP模型分析图片
        try:
            from backend.services.clip_service_local import clip_service
            image_features = clip_service.encode_image(image_bytes)
            feature_summary = f"提取了{image_features.shape[1]}维特征向量"
        except Exception as clip_error:
            feature_summary = f"CLIP模型分析失败: {str(clip_error)}"
        
        # 构建分析结果
        analysis_result = f"图片分析结果:\n"
        analysis_result += f"尺寸: {img_info['width']}x{img_info['height']}\n"
        analysis_result += f"格式: {img_info['format']}\n"
        analysis_result += f"CLIP特征: {feature_summary}\n"
        
        if description:
            analysis_result += f"用户描述: {description}\n"
        
        analysis_result += "内容分析: 已使用CLIP模型提取图片特征向量，可用于语义搜索和相似度计算。"
        
        execution_time = time.time() - start_time
        metrics_collector.record_tool_usage("analyze_image", execution_time)
        
        return analysis_result
        
    except Exception as e:
        execution_time = time.time() - start_time
        metrics_collector.record_tool_usage("analyze_image", execution_time, success=False)
        return f"图片分析出错: {str(e)}"

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

def _execute_tool_call_into_state(state: AgentState, tool_call: Any) -> Dict[str, Any]:
    if isinstance(tool_call, dict):
        tool_name = (tool_call.get("function") or {}).get("name") or tool_call.get("name") or ""
        raw_args = (tool_call.get("function") or {}).get("arguments") or tool_call.get("arguments")
    else:
        tool_name = getattr(getattr(tool_call, "function", None), "name", "") or ""
        raw_args = getattr(getattr(tool_call, "function", None), "arguments", None)

    tool_args = _safe_parse_tool_arguments(raw_args)
    t0 = time.time()
    ok = True
    result: Any = ""
    try:
        if tool_name == "rag_image_search":
            q = (tool_args.get("query") or "").strip()
            k = _coerce_top_k(tool_args.get("top_k", state.top_k), default=state.top_k, min_value=1, max_value=20)
            raw_results = milvus_service.search_images_by_text(q, k) if q else []
            state.images = [
                {
                    "id": r.get("id"),
                    "filename": r.get("filename", "unknown"),
                    "similarity": float(r.get("score", 0.0)),
                    "score": float(r.get("score", 0.0)),
                }
                for r in (raw_results or [])
            ]
            result = rag_image_search.invoke({"query": q, "top_k": k})
        elif tool_name == "web_search":
            q = (tool_args.get("query") or "").strip()
            k = _coerce_top_k(tool_args.get("max_results", 5), default=5, min_value=1, max_value=50)
            results: List[Dict[str, Any]] = []
            search_mode = (os.getenv("WEB_SEARCH_MODE") or "ark").strip().lower()
            if q:
                try:
                    results = search_service.search_web_sync(q, k, mode=search_mode)
                except Exception:
                    results = []
            state.search_results = results
            if results:
                result = web_search.invoke({"query": q, "max_results": k, "mode": search_mode, "_results": results})
            else:
                result = web_search.invoke({"query": q, "max_results": k, "mode": search_mode})
            fetch_all = (os.getenv("WEB_SEARCH_FETCH_ALL") or "").strip().lower() in ("1", "true", "yes", "y")
            if fetch_all and results:
                urls = [(r.get("link") or "").strip() for r in results if (r.get("link") or "").strip()]
                if urls:
                    pages = search_service.batch_read_webpages_sync(urls, output_format="markdown", mode=search_mode)
                    if pages:
                        merged = []
                        for u, content in pages.items():
                            c = (content or "").strip()
                            if not c:
                                continue
                            merged.append(f"URL: {u}\n{c}\n")
                        if merged:
                            result = f"{result}\n\n并行抓取到的网页内容（WEB_SEARCH_FETCH_ALL=1）：\n\n" + "\n".join(merged)
        elif tool_name == "web_read":
            u = (tool_args.get("url") or "").strip()
            fmt = (tool_args.get("format") or "markdown").strip().lower()
            if fmt not in ("json", "markdown"):
                fmt = "markdown"
            result = web_read.invoke({"url": u, "format": fmt})
        elif tool_name == "analyze_image":
            result = analyze_image.invoke(tool_args)
        else:
            result = f"未知工具: {tool_name}"
    except Exception as e:
        ok = False
        result = f"工具 {tool_name} 执行失败: {str(e)}"
        print(f"[tool] {result}")
    elapsed_s = time.time() - t0

    if state.timing is None:
        state.timing = {}
    if tool_name:
        state.timing[f"tool_{tool_name}"] = float(elapsed_s)

    if state.tool_results is None:
        state.tool_results = {}
    if tool_name:
        state.tool_results[tool_name] = result

    if state.messages is None:
        state.messages = []
    if tool_name:
        state.messages.append(AIMessage(content=f"工具 {tool_name} 返回: {result}"))

    preview = (str(result) or "").replace("\r\n", "\n").replace("\r", "\n")
    if len(preview) > 500:
        preview = preview[:500] + "..."
    _append_tool_trace(state, tool_name or "unknown", tool_args, ok, elapsed_s, preview)

    return {"tool": tool_name or "unknown", "args": tool_args, "ok": ok, "elapsed_s": elapsed_s, "result": result}

def process_tool_results(state: AgentState) -> AgentState:
    """处理工具结果并更新状态"""
    if state.tool_calls:
        for tool_call in state.tool_calls:
            _execute_tool_call_into_state(state, tool_call)
    
    # 重置工具调用状态
    state.needs_tool = False
    state.tool_calls = []
    
    return state

def create_agent_graph():
    """创建Agent图"""
    
    # 定义工具列表（不再使用ToolNode）
    tools = [rag_image_search, web_search, web_read, analyze_image]
    
    def should_continue(state: AgentState) -> str:
        """决定是否继续执行"""
        # 如果Agent标记需要工具调用，继续执行工具
        if state.needs_tool:
            return "continue"
        
        # 如果已有最终答案，结束
        if state.answer:
            return "end"
        
        # 否则继续调用工具
        return "continue"
    
    # 创建状态图
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("agent", agent_node)
    workflow.add_node("process_results", process_tool_results)
    
    # 设置入口点
    workflow.set_entry_point("agent")
    
    # 添加条件边 - 直接到结果处理节点
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "process_results",
            "end": END
        }
    )
    
    # 添加结果处理到Agent的边
    workflow.add_edge("process_results", "agent")
    
    # 编译图
    return workflow.compile()

# 创建全局Agent图实例
agent_graph = create_agent_graph()
