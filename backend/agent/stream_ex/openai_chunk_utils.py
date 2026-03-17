from typing import Any, Dict, Optional


def openai_chunk(chat_id: str, created: int, model: str, delta: Dict[str, Any], finish_reason: Optional[str] = None) -> Dict[str, Any]:
    return {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def merge_tool_call_delta(pending: Dict[int, Dict[str, Any]], tool_calls_delta: Any) -> None:
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
