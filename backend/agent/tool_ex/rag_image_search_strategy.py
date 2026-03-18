from typing import Any, Dict

from backend.agent.tool_ex.execution_context import ToolExecutionContext
from backend.agent.tool_ex.image_result_utils import normalize_image_hits
from backend.agent.node_ex.memory_node import MemoryManagerFactory


class RagImageSearchStrategy:
    tool_name = "rag_image_search"

    async def execute(self, state: Any, tool_args: Dict[str, Any], ctx: ToolExecutionContext) -> Any:
        import asyncio
        loop = asyncio.get_running_loop()

        q = (tool_args.get("query") or "").strip()
        k = ctx.coerce_top_k(tool_args.get("top_k", getattr(state, "top_k", 5)), default=getattr(state, "top_k", 5), min_value=1, max_value=20)

        try:
            user_id = getattr(state, "user_id", "default_user")
            dept_id = getattr(state, "dept_id", "default_dept")

            manager = MemoryManagerFactory.get_manager()
            raw_results = await loop.run_in_executor(
                None,
                lambda: manager.image_memory.search_images_by_text(q, k, user_id=user_id, dept_id=dept_id)
            )
        except Exception as e:
            print(f"[RagImageSearchStrategy] MemoryManager search failed: {e}")
            raw_results = []

        normalized_results = normalize_image_hits(raw_results)
        state.images = normalized_results
        return await ctx.rag_image_search_invoke({"query": q, "top_k": k, "results": normalized_results})

