from typing import Any, Dict

from backend.agent.node_ex.memory_node import MemoryManagerFactory
from backend.agent.tool_ex.execution_context import ToolExecutionContext


class SaveUserFactStrategy:
    tool_name = "save_user_fact"

    async def execute(self, state: Any, tool_args: Dict[str, Any], ctx: ToolExecutionContext) -> Any:
        user_id = (tool_args.get("user_id") or getattr(state, "user_id", "") or "").strip()
        fact = (tool_args.get("fact") or "").strip()
        state_dept_id = (getattr(state, "dept_id", "default_dept") or "default_dept").strip()
        arg_dept_id = (tool_args.get("dept_id") or "").strip()
        dept_id = state_dept_id if (not arg_dept_id or arg_dept_id == "default_dept") else arg_dept_id
        state_visibility = (getattr(state, "visibility", "private") or "private").strip().lower()
        arg_visibility = (tool_args.get("visibility") or "").strip().lower()
        visibility = state_visibility if (not arg_visibility or arg_visibility == "private") else arg_visibility
        if visibility not in ("private", "department"):
            visibility = "private"
        if not user_id:
            return "缺少 user_id，无法保存长期记忆。"
        if not fact:
            return "事实内容为空，未保存。"

        image_data = getattr(state, "image_data", None)
        image_filename = (getattr(state, "image_filename", None) or "uploaded_image.jpg").strip() or "uploaded_image.jpg"
        memory_type = "image_summary" if image_data else "fact"
        source = "image_upload" if image_data else "user_fact"
        manager = MemoryManagerFactory.get_manager()
        mem_id = await manager.add_user_fact(
            user_id=user_id,
            fact=fact,
            dept_id=dept_id,
            visibility=visibility,
            memory_type=memory_type,
            source=source,
            metadata_extra={"filename": image_filename} if image_data else None,
        )
        image_id = ""
        if image_data:
            import asyncio

            loop = asyncio.get_running_loop()
            image_id = await loop.run_in_executor(
                None,
                lambda: manager.image_memory.add_image_memory(
                    user_id=user_id,
                    dept_id=dept_id,
                    image_bytes=image_data,
                    description=fact,
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
                "type": "long_term_saved",
                "memory_id": mem_id,
                "fact": fact,
                "memory_type": memory_type,
                "visibility": visibility,
                "dept_id": dept_id,
            }
        )
        if image_id:
            events.append(
                {
                    "type": "image_memory_saved",
                    "image_id": image_id,
                    "visibility": visibility,
                    "dept_id": dept_id,
                    "filename": image_filename,
                }
            )
            await ctx.save_user_image_invoke(
                {
                    "description": fact,
                    "visibility": visibility,
                    "dept_id": dept_id,
                    "image_id": image_id,
                    "filename": image_filename,
                }
            )
        return await ctx.save_user_fact_invoke({"fact": fact, "visibility": visibility, "dept_id": dept_id})
