from typing import Any, Dict

from backend.agent.tool_ex.execution_context import ToolExecutionContext


class AnalyzeImageStrategy:
    tool_name = "analyze_image"

    def execute(self, state: Any, tool_args: Dict[str, Any], ctx: ToolExecutionContext) -> Any:
        import base64
        image_data = getattr(state, "image_data", None)
        if image_data:
            if not tool_args.get("image_data_base64"):
                tool_args["image_data_base64"] = base64.b64encode(image_data).decode("ascii")
            
            # Perform image-to-image search to populate state.images
            try:
                from backend.services.clip_service_local import clip_service
                image_features = clip_service.encode_image(image_data)
                k = ctx.coerce_top_k(tool_args.get("top_k", getattr(state, "top_k", 5)), default=getattr(state, "top_k", 5), min_value=1, max_value=20)
                raw_results = ctx.milvus_service.search_images(image_features, k)
                state.images = [
                    {
                        "id": (r.get("id") if isinstance(r, dict) else None),
                        "filename": ((r.get("filename", "unknown") if isinstance(r, dict) else "unknown") or "unknown"),
                        "similarity": float(((r.get("score", 0.0) if isinstance(r, dict) else 0.0) or 0.0)),
                        "score": float(((r.get("score", 0.0) if isinstance(r, dict) else 0.0) or 0.0)),
                    }
                    for r in (raw_results or [])
                ]
            except Exception as e:
                print(f"[AnalyzeImageStrategy] image search error: {e}")

        return ctx.analyze_image_invoke(tool_args)

