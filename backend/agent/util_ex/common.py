import json
import os
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from backend.agent.config_ex.model_config import get_runtime_model_settings


def safe_parse_tool_arguments(raw_args: Any) -> Dict[str, Any]:
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


def coerce_top_k(value: Any, default: int = 5, min_value: int = 1, max_value: int = 20) -> int:
    try:
        k = int(value)
    except Exception:
        k = default
    if k < min_value:
        return min_value
    if k > max_value:
        return max_value
    return k


def get_langchain_chat_model(model: str, *, temperature: float, max_tokens: int, streaming: bool):
    settings = get_runtime_model_settings()
    api_key = settings.get("api_key", "")
    base_url = settings.get("base_url", "")
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
    except Exception:
        from langchain.chat_models import ChatOpenAI  # type: ignore

    return ChatOpenAI(
        model_name=(model or settings.get("base_model") or "").strip(),
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        streaming=bool(streaming),
    )


def to_lc_message(msg: Any):
    if isinstance(msg, (SystemMessage, HumanMessage, AIMessage, ToolMessage)):
        return msg
    if isinstance(msg, dict):
        role = (msg.get("role") or "").strip().lower()
        content = msg.get("content")
        if content is None:
            content = ""
        if role == "system":
            return SystemMessage(content=str(content))
        if role == "user":
            return HumanMessage(content=str(content))
        if role == "tool":
            tcid = (msg.get("tool_call_id") or "").strip()
            return ToolMessage(content=str(content), tool_call_id=tcid)
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            return AIMessage(content=str(content), additional_kwargs={"tool_calls": tool_calls})
        return AIMessage(content=str(content))
    if hasattr(msg, "content") and hasattr(msg, "type"):
        return msg
    return AIMessage(content=str(msg))
