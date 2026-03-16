import os
from typing import Dict, List, Optional

from backend.agent.search_ex.bing_html_text_search import BingHtmlTextSearch


class DuckDuckGoTextSearch:
    def __init__(self, *, bing_fallback: Optional[BingHtmlTextSearch] = None) -> None:
        self._bing = bing_fallback or BingHtmlTextSearch()

    def search(self, query: str, max_results: int) -> List[Dict[str, str]]:
        from duckduckgo_search import DDGS

        q = (query or "").strip()
        if not q:
            return []

        backends: List[str] = []
        preferred = (os.getenv("DDG_BACKEND") or "").strip().lower()
        if preferred:
            backends.append(preferred)
        backends.extend(["lite", "html", "api", "auto"])

        seen = set()
        backend_order: List[str] = []
        for b in backends:
            if b and b not in seen:
                seen.add(b)
                backend_order.append(b)

        last_error: Optional[Exception] = None
        for backend in backend_order:
            try:
                ddgs = DDGS()
                results: List[Dict[str, str]] = []
                search_results = ddgs.text(
                    q,
                    region=os.getenv("DDG_REGION", "wt-wt"),
                    max_results=max_results,
                    backend=backend,
                )
                for r in search_results:
                    if not isinstance(r, dict):
                        continue
                    results.append(
                        {
                            "title": r.get("title", "") or "",
                            "link": r.get("href", "") or "",
                            "snippet": r.get("body", "") or "",
                            "source": f"duckduckgo:{backend}",
                        }
                    )
                if results:
                    return results
                continue
            except Exception as e:
                last_error = e
                continue

        if last_error:
            provider = (os.getenv("WEB_SEARCH_PROVIDER") or "bing").strip().lower()
            if provider in ("bing", "bing_html", "bing_scrape", "default"):
                return self._bing.search(q, max_results)
            raise last_error
        return self._bing.search(q, max_results)

