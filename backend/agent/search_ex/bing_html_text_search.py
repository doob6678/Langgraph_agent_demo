import os
from typing import Dict, List


class BingHtmlTextSearch:
    def search(self, query: str, max_results: int) -> List[Dict[str, str]]:
        import re
        import html as _html
        import requests
        from urllib.parse import quote_plus

        q = (query or "").strip()
        if not q:
            return []

        timeout_s = 10
        try:
            timeout_s = int((os.getenv("SEARCH_TIMEOUT") or "10").strip() or "10")
        except Exception:
            timeout_s = 10
        if timeout_s < 3:
            timeout_s = 3
        if timeout_s > 30:
            timeout_s = 30

        def _fetch(url: str) -> str:
            resp = requests.get(
                url,
                timeout=timeout_s,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                },
            )
            resp.raise_for_status()
            return resp.text or ""

        base = "https://www.bing.com/search?q="
        html_text = _fetch(f"{base}{quote_plus(q)}&setlang=zh-hans&cc=CN")
        if "b_results" not in html_text:
            html_text = _fetch(f"{base}{quote_plus(q)}")

        blocks = re.findall(
            r'<li[^>]+class="[^"]*\bb_algo\b[^"]*"[^>]*>.*?</li>',
            html_text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        results: List[Dict[str, str]] = []
        for b in blocks:
            m = re.search(
                r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                b,
                flags=re.DOTALL | re.IGNORECASE,
            )
            if not m:
                continue
            link = _html.unescape(m.group(1)).strip()
            title_raw = re.sub(r"<.*?>", "", m.group(2), flags=re.DOTALL)
            title = _html.unescape(title_raw).strip()

            sm = re.search(r'<div class="b_caption".*?<p>(.*?)</p>', b, flags=re.DOTALL | re.IGNORECASE)
            snippet_raw = re.sub(r"<.*?>", "", sm.group(1), flags=re.DOTALL) if sm else ""
            snippet = _html.unescape(snippet_raw).strip()

            results.append({"title": title, "link": link, "snippet": snippet, "source": "bing"})
            if len(results) >= max_results:
                break
        return results

