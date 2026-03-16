from typing import Any, Dict, List

from backend.agent.tool_ex.execution_context import ToolExecutionContext


class WebSearchStrategy:
    tool_name = "web_search"

    def execute(self, state: Any, tool_args: Dict[str, Any], ctx: ToolExecutionContext) -> Any:
        q = (tool_args.get("query") or "").strip()
        k = ctx.coerce_top_k(tool_args.get("max_results", 5), default=5, min_value=1, max_value=50)
        results: List[Dict[str, Any]] = []
        search_mode = (ctx.getenv("WEB_SEARCH_MODE") or "ark").strip().lower()
        if q:
            try:
                results = ctx.search_service.search_web_sync(q, k, mode=search_mode)
            except Exception:
                results = []
        state.search_results = results
        if results:
            result = ctx.web_search_invoke({"query": q, "max_results": k, "mode": search_mode, "_results": results})
        else:
            result = ctx.web_search_invoke({"query": q, "max_results": k, "mode": search_mode})

        fetch_all = (ctx.getenv("WEB_SEARCH_FETCH_ALL") or "").strip().lower() in ("1", "true", "yes", "y")
        if fetch_all and results:
            urls = [(r.get("link") or "").strip() for r in results if isinstance(r, dict) and (r.get("link") or "").strip()]
            if urls:
                pages = ctx.search_service.batch_read_webpages_sync(urls, output_format="markdown", mode=search_mode)
                if pages:
                    merged = []
                    for u, content in pages.items():
                        c = (content or "").strip()
                        if not c:
                            continue
                        merged.append(f"URL: {u}\n{c}\n")
                    if merged:
                        return f"{result}\n\n并行抓取到的网页内容（WEB_SEARCH_FETCH_ALL=1）：\n\n" + "\n".join(merged)

        return result

