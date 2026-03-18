from typing import Any, Dict

from backend.agent.tool_ex.execution_context import ToolExecutionContext
from backend.agent.tool_ex.image_result_utils import normalize_image_hits
from backend.agent.node_ex.memory_node import MemoryManagerFactory

class AnalyzeImageStrategy:
    tool_name = "analyze_image"

    async def execute(self, state: Any, tool_args: Dict[str, Any], ctx: ToolExecutionContext) -> Any:
        import base64
        import asyncio
        loop = asyncio.get_running_loop()

        image_data = getattr(state, "image_data", None)
        raw_results = []
        if image_data:
            if not tool_args.get("image_data_base64"):
                tool_args["image_data_base64"] = base64.b64encode(image_data).decode("ascii")

            user_id = getattr(state, "user_id", "default_user")
            dept_id = getattr(state, "dept_id", "default_dept")

            manager = MemoryManagerFactory.get_manager()
            image_memory = manager.image_memory

            from backend.services.clip_service_local import clip_service

            image_features = await loop.run_in_executor(None, lambda: clip_service.encode_image(image_data))

            k = ctx.coerce_top_k(tool_args.get("top_k", getattr(state, "top_k", 5)), default=getattr(state, "top_k", 5), min_value=1, max_value=20)

            raw_results = await loop.run_in_executor(
                None,
                lambda: image_memory.search_images(
                    query_vector=image_features,
                    top_k=k,
                    user_id=user_id,
                    dept_id=dept_id
                )
            )

            state.images = normalize_image_hits(raw_results)

        tool_args["results"] = normalize_image_hits(raw_results)
        return await ctx.analyze_image_invoke(tool_args)
