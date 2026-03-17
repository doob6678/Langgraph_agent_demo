import json
import time
from typing import Any, Callable, Dict, Iterable, List


class ToolFallbackPlanner:
    def __init__(
        self,
        *,
        coerce_top_k: Callable[..., int],
        execute_tool_call: Callable[[Any, Any], Dict[str, Any]],
        openai_chunk: Callable[[str, int, str, Dict[str, Any], Any], Dict[str, Any]],
    ) -> None:
        self._coerce_top_k = coerce_top_k
        self._execute_tool_call = execute_tool_call
        self._openai_chunk = openai_chunk

    def stream_if_needed(self, state: Any, query: str, *, chat_id: str, created: int, model: str) -> Iterable[Dict[str, Any]]:
        if getattr(state, "answer", ""):
            return []
        if getattr(state, "tool_results", None):
            return []
        if not (bool(getattr(state, "use_search", False)) or bool(getattr(state, "use_rag", False)) or bool(getattr(state, "image_data", None))):
            return []

        q = (query or "").strip()
        tool_calls = self._build_tool_calls(state, q)
        for tc in tool_calls:
            tool_event = self._execute_tool_call(state, tc)
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

        answer = self._synthesize_answer(state)
        try:
            state.answer = answer
        except Exception:
            pass
        if answer:
            from backend.agent.stream_ex.image_buffer import ImageMarkdownBuffer
            buffer = ImageMarkdownBuffer()

            for i in range(0, len(answer), 64):
                chunk = answer[i : i + 64]
                if chunk:
                    safe = buffer.process(chunk)
                    if safe:
                        yield self._openai_chunk(chat_id, created, model, {"content": safe})
            
            rem = buffer.flush()
            if rem:
                yield self._openai_chunk(chat_id, created, model, {"content": rem})

    def _build_tool_calls(self, state: Any, query: str) -> List[Dict[str, Any]]:
        tool_calls: List[Dict[str, Any]] = []
        ts = int(time.time() * 1000)
        top_k = getattr(state, "top_k", 5)

        if bool(getattr(state, "use_search", False)):
            tool_calls.append(
                {
                    "id": f"call_{ts}_search",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": json.dumps(
                            {"query": query, "max_results": self._coerce_top_k(top_k, default=5, min_value=1, max_value=50)},
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
                            {"query": query, "top_k": self._coerce_top_k(top_k, default=5, min_value=1, max_value=20)},
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

        return tool_calls

    def _synthesize_answer(self, state: Any) -> str:
        parts: List[str] = []
        search_results = getattr(state, "search_results", None)
        if isinstance(search_results, list) and search_results:
            lines: List[str] = []
            for i, item in enumerate(search_results[:3], 1):
                if not isinstance(item, dict):
                    continue
                title = (item.get("title") or "").strip()
                link = (item.get("link") or item.get("url") or "").strip()
                if title and link:
                    lines.append(f"{i}. {title} {link}".strip())
                elif title:
                    lines.append(f"{i}. {title}".strip())
                elif link:
                    lines.append(f"{i}. {link}".strip())
            if lines:
                parts.append("联网搜索要点：\n" + "\n".join(lines))

        images = getattr(state, "images", None)
        if isinstance(images, list) and images:
            fns: List[str] = []
            for item in images[:5]:
                if not isinstance(item, dict):
                    continue
                fn = (item.get("filename") or "").strip()
                if fn:
                    fns.append(fn)
            if fns:
                parts.append("相关图片：\n" + "\n".join([f"![{fn}]({fn})" for fn in fns]))

        if not parts:
            parts.append("未获取到可用结果。")
        return "\n\n".join(parts).strip()
