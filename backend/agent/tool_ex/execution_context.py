import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass(frozen=True)
class ToolExecutionContext:
    milvus_service: Any
    search_service: Any
    rag_image_search_invoke: Callable[[Dict[str, Any]], Any]
    web_search_invoke: Callable[[Dict[str, Any]], Any]
    web_read_invoke: Callable[[Dict[str, Any]], Any]
    analyze_image_invoke: Callable[[Dict[str, Any]], Any]
    save_user_fact_invoke: Callable[[Dict[str, Any]], Any]
    save_user_image_invoke: Callable[[Dict[str, Any]], Any]
    coerce_top_k: Callable[[Any, int, int, int], int]
    getenv: Callable[[str], Optional[str]] = os.getenv

