from typing import Any, Dict

from backend.agent.node_ex.memory_node import MemoryManagerFactory
from backend.agent.tool_ex.execution_context import ToolExecutionContext


class SaveUserImageStrategy:
    tool_name = "save_user_image"

    async def execute(self, state: Any, tool_args: Dict[str, Any], ctx: ToolExecutionContext) -> Any:
        import asyncio

        image_data = getattr(state, "image_data", None)
        if not image_data:
            return "当前会话没有可保存的图片。"

        user_id = (tool_args.get("user_id") or getattr(state, "user_id", "") or "").strip()
        state_dept_id = (getattr(state, "dept_id", "default_dept") or "default_dept").strip()
        arg_dept_id = (tool_args.get("dept_id") or "").strip()
        dept_id = state_dept_id if (not arg_dept_id or arg_dept_id == "default_dept") else arg_dept_id
        state_visibility = (getattr(state, "visibility", "private") or "private").strip().lower()
        arg_visibility = (tool_args.get("visibility") or "").strip().lower()
        visibility = state_visibility if (not arg_visibility or arg_visibility == "private") else arg_visibility
        description = (tool_args.get("description") or "").strip()
        if visibility not in ("private", "department"):
            visibility = "private"
        if not user_id:
            return "缺少 user_id，无法保存图片记忆。"

        image_filename = (getattr(state, "image_filename", None) or "uploaded_image.jpg").strip() or "uploaded_image.jpg"

        manager = MemoryManagerFactory.get_manager()
        loop = asyncio.get_running_loop()
        image_id = await loop.run_in_executor(
            None,
            lambda: manager.image_memory.add_image_memory(
                user_id=user_id,
                dept_id=dept_id,
                image_bytes=image_data,
                description=description,
                metadata={"filename": image_filename},
                visibility=visibility,
            ),
        )

        if getattr(state, "memory_data", None) is None:
            state.memory_data = {}
        events = state.memory_data.get("events")
        if not isinstance(events, list):
            events = []
            state.memory_data["events"] = events
        events.append(
            {
                "type": "image_memory_saved",
                "image_id": image_id,
                "visibility": visibility,
                "dept_id": dept_id,
                "filename": image_filename,
            }
        )
        return await ctx.save_user_image_invoke(
            {
                "description": description,
                "visibility": visibility,
                "dept_id": dept_id,
                "image_id": image_id,
                "filename": image_filename,
            }
        )
