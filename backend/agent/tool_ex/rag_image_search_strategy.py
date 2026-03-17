from typing import Any, Dict

from backend.agent.tool_ex.execution_context import ToolExecutionContext


class RagImageSearchStrategy:
    tool_name = "rag_image_search"

    def execute(self, state: Any, tool_args: Dict[str, Any], ctx: ToolExecutionContext) -> Any:
        q = (tool_args.get("query") or "").strip()
        k = ctx.coerce_top_k(tool_args.get("top_k", getattr(state, "top_k", 5)), default=getattr(state, "top_k", 5), min_value=1, max_value=20)
        raw_results = ctx.milvus_service.search_images_by_text(q, k) if q else []
        state.images = [
            {
                "id": (r.get("id") if isinstance(r, dict) else None),
                "filename": ((r.get("filename", "unknown") if isinstance(r, dict) else "unknown") or "unknown"),
                "similarity": float(((r.get("score", 0.0) if isinstance(r, dict) else 0.0) or 0.0)),
                "score": float(((r.get("score", 0.0) if isinstance(r, dict) else 0.0) or 0.0)),
            }
            for r in (raw_results or [])
        ]
        return ctx.rag_image_search_invoke({"query": q, "top_k": k})

