import os
import time
import asyncio
from typing import List, Dict, Any, Optional
from duckduckgo_search import DDGS
import requests

def _metaso_mcp_web_search_sync(query: str, max_results: int, timeout_s: int) -> List[Dict[str, Any]]:
    import json

    q = (query or "").strip()
    if not q:
        return []

    api_key = (os.getenv("METASO_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("missing METASO_API_KEY")

    url = (os.getenv("METASO_MCP_URL") or "https://metaso.cn/api/mcp").strip()
    if not url:
        url = "https://metaso.cn/api/mcp"

    try:
        k = int(max_results)
    except Exception:
        k = 10
    if k < 1:
        k = 1
    if k > 50:
        k = 50

    if timeout_s < 5:
        timeout_s = 5
    if timeout_s > 120:
        timeout_s = 120

    payload = {
        "jsonrpc": "2.0",
        "id": f"metaso_{int(time.time() * 1000)}",
        "method": "tools/call",
        "params": {
            "name": "metaso_web_search",
            "arguments": {
                "q": q,
                "scope": (os.getenv("METASO_SEARCH_SCOPE") or "webpage").strip() or "webpage",
                "includeSummary": True,
                "size": k,
            },
        },
    }

    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout_s,
    )
    resp.raise_for_status()
    data = resp.json() if resp.content else {}
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"metaso_mcp_error={data.get('error')}")

    result = data.get("result") if isinstance(data, dict) else None
    candidates: Any = None
    if isinstance(result, dict):
        if isinstance(result.get("results"), list):
            candidates = result.get("results")
        elif isinstance(result.get("data"), list):
            candidates = result.get("data")
        elif isinstance(result.get("items"), list):
            candidates = result.get("items")
        elif isinstance(result.get("content"), list):
            candidates = result.get("content")
        elif isinstance(result.get("content"), str):
            candidates = result.get("content")
    elif isinstance(result, list):
        candidates = result
    elif isinstance(result, str):
        candidates = result

    extracted: List[Dict[str, Any]] = []

    def _append_item(item: Any) -> None:
        if not isinstance(item, dict):
            return
        title = (item.get("title") or item.get("name") or "").strip()
        link = (item.get("link") or item.get("url") or item.get("href") or "").strip()
        snippet = (item.get("snippet") or item.get("summary") or item.get("content") or "").strip()
        if not title and snippet:
            first_line = (snippet.splitlines()[0] if snippet.splitlines() else "").strip()
            if first_line:
                title = first_line[:120]
        if not title and not link and not snippet:
            return
        extracted.append({"title": title or link or "无标题", "link": link, "snippet": snippet, "source": "metaso_mcp"})

    def _append_from_parsed(parsed: Any) -> None:
        if isinstance(parsed, list):
            for e in parsed:
                _append_item(e)
            return
        if not isinstance(parsed, dict):
            return
        if isinstance(parsed.get("results"), list):
            for e in parsed.get("results") or []:
                _append_item(e)
            return
        if isinstance(parsed.get("data"), list):
            for e in parsed.get("data") or []:
                _append_item(e)
            return
        if isinstance(parsed.get("items"), list):
            for e in parsed.get("items") or []:
                _append_item(e)
            return
        if isinstance(parsed.get("webpages"), list):
            for e in parsed.get("webpages") or []:
                _append_item(
                    {
                        "title": e.get("title", ""),
                        "link": e.get("link", ""),
                        "snippet": e.get("snippet", ""),
                    }
                )
            return
        _append_item(parsed)

    if isinstance(candidates, list):
        for it in candidates:
            if isinstance(it, dict) and it.get("type") and (it.get("text") or it.get("data")):
                raw = it.get("data") if isinstance(it.get("data"), (dict, list)) else it.get("text")
                if isinstance(raw, (dict, list)):
                    if isinstance(raw, list):
                        for e in raw:
                            _append_item(e)
                    elif isinstance(raw, dict):
                        if isinstance(raw.get("results"), list):
                            for e in raw.get("results"):
                                _append_item(e)
                        elif isinstance(raw.get("data"), list):
                            for e in raw.get("data"):
                                _append_item(e)
                        else:
                            _append_item(raw)
                elif isinstance(raw, str):
                    try:
                        parsed = json.loads(raw)
                        _append_from_parsed(parsed)
                    except Exception:
                        pass
            else:
                _append_item(it)
    elif isinstance(candidates, str):
        try:
            parsed = json.loads(candidates)
            _append_from_parsed(parsed)
        except Exception:
            pass

    filtered = _filter_results_by_query(q, extracted)
    if filtered:
        return filtered[:k]
    return extracted[:k]

def _metaso_mcp_web_reader_sync(url: str, output_format: str, timeout_s: int) -> str:
    import json

    u = (url or "").strip()
    if not u:
        return ""
    if not (u.startswith("http://") or u.startswith("https://")):
        raise ValueError("invalid url scheme")
    if len(u) > 2048:
        raise ValueError("url too long")

    fmt = (output_format or "").strip().lower()
    if fmt not in ("json", "markdown"):
        fmt = "markdown"

    api_key = (os.getenv("METASO_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("missing METASO_API_KEY")

    mcp_url = (os.getenv("METASO_MCP_URL") or "https://metaso.cn/api/mcp").strip()
    if not mcp_url:
        mcp_url = "https://metaso.cn/api/mcp"

    try:
        t = int(timeout_s)
    except Exception:
        t = 30
    if t < 5:
        t = 5
    if t > 120:
        t = 120

    payload = {
        "jsonrpc": "2.0",
        "id": f"metaso_reader_{int(time.time() * 1000)}",
        "method": "tools/call",
        "params": {
            "name": "metaso_web_reader",
            "arguments": {"url": u, "format": fmt},
        },
    }

    resp = requests.post(
        mcp_url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=t,
    )
    resp.raise_for_status()
    data = resp.json() if resp.content else {}
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"metaso_mcp_error={data.get('error')}")

    result = data.get("result") if isinstance(data, dict) else None
    content_text = ""
    if isinstance(result, dict) and isinstance(result.get("content"), list):
        for it in result.get("content") or []:
            if isinstance(it, dict) and (it.get("type") == "text") and isinstance(it.get("text"), str):
                content_text = it.get("text") or ""
                break
    elif isinstance(result, str):
        content_text = result

    content_text = (content_text or "").strip()
    if not content_text:
        return ""

    if fmt == "json":
        try:
            parsed = json.loads(content_text)
            return json.dumps(parsed, ensure_ascii=False, indent=2)
        except Exception:
            return content_text

    max_chars = 15000
    try:
        max_chars = int((os.getenv("WEB_READ_MAX_CHARS") or "15000").strip() or "15000")
    except Exception:
        max_chars = 15000
    if max_chars < 1000:
        max_chars = 1000
    if max_chars > 200000:
        max_chars = 200000

    return content_text[:max_chars]

def _filter_results_by_query(query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    import re

    q = (query or "").strip()
    if not q:
        return results

    terms = set()
    for t in re.findall(r"[\u4e00-\u9fff]{2,}", q):
        terms.add(t)
    for t in re.findall(r"[a-zA-Z]{3,}", q):
        terms.add(t.lower())

    if not terms:
        return results

    filtered: List[Dict[str, Any]] = []
    for r in results or []:
        title = (r.get("title") or "").strip()
        snippet = (r.get("snippet") or "").strip()
        hay = f"{title} {snippet}"
        hay_l = hay.lower()
        keep = False
        for term in terms:
            if any("\u4e00" <= ch <= "\u9fff" for ch in term):
                if term in hay:
                    keep = True
                    break
            else:
                if term in hay_l:
                    keep = True
                    break
        if keep:
            filtered.append(r)

    return filtered

def _ddg_text_search_sync(query: str, max_results: int, timeout_s: int) -> List[Dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        return []

    try:
        k = int(max_results)
    except Exception:
        k = 5
    if k < 1:
        k = 1
    if k > 50:
        k = 50

    preferred = (os.getenv("DDG_BACKEND") or "").strip().lower()
    backends = [preferred, "lite", "html", "api", "auto"]
    seen = set()
    backend_order = []
    for b in backends:
        if b and b not in seen:
            seen.add(b)
            backend_order.append(b)

    last_error: Optional[Exception] = None
    for backend in backend_order:
        try:
            ddgs = DDGS()
            results: List[Dict[str, Any]] = []
            search_results = ddgs.text(
                q,
                region=os.getenv("DDG_REGION", "wt-wt"),
                max_results=k,
                backend=backend,
            )
            for result in search_results:
                results.append(
                    {
                        "title": result.get("title", ""),
                        "link": result.get("href", ""),
                        "snippet": result.get("body", ""),
                        "source": f"duckduckgo:{backend}",
                    }
                )
            if results:
                filtered = _filter_results_by_query(q, results)
                return filtered if filtered else []
        except Exception as e:
            last_error = e
            continue

    if last_error is not None:
        try:
            return _bing_text_search(q, k, timeout_s)
        except Exception:
            return []
    return []

def _bing_text_search(query: str, max_results: int, timeout_s: int) -> List[Dict[str, Any]]:
    import re
    import html as _html
    from urllib.parse import quote_plus

    q = (query or "").strip()
    if not q:
        return []

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
    lower = (html_text or "").lower()
    if "b_captcha" in lower or "/challenge" in lower or "unusual traffic" in lower or "2 plus 5" in lower:
        return []

    blocks = re.findall(
        r'<li[^>]+class="[^"]*\bb_algo\b[^"]*"[^>]*>.*?</li>',
        html_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    results: List[Dict[str, Any]] = []
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
    filtered = _filter_results_by_query(q, results)
    return filtered if filtered else []

class SearchService:
    """联网搜索服务"""
    
    def __init__(self):
        self.max_results = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
        self.timeout = int(os.getenv("SEARCH_TIMEOUT", "10"))

    def _ark_base_url(self) -> str:
        base = (os.getenv("BASE_URL") or os.getenv("ARK_BASE_URL") or os.getenv("ARK_API_BASE_URL") or "").strip()
        if not base:
            base = "https://ark.cn-beijing.volces.com/api/v3"
        base = base.rstrip("/")
        if base.endswith("/api/v3"):
            return base
        if base.endswith("/api/v3/"):
            return base[:-1]
        if base.endswith("/api"):
            return f"{base}/v3"
        if base.endswith("/api/"):
            return f"{base}v3"
        if base.endswith("/v3"):
            return base
        return base

    def _ark_web_search_model(self) -> str:
        return (
            (os.getenv("ARK_WEB_SEARCH_MODEL") or "").strip()
            or (os.getenv("BASE_MODEL") or os.getenv("ARK_MODEL") or os.getenv("ARK_MODEL_NAME") or "").strip()
            or "doubao-seed-1-6-250615"
        )

    def _ark_api_key(self) -> str:
        return (os.getenv("BASE_API_KEY") or os.getenv("ARK_API_KEY") or "").strip()

    def _search_web_ark(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return []

        k = max_results
        try:
            k = int(k)
        except Exception:
            k = self.max_results
        if k < 1:
            k = 1
        if k > 50:
            k = 50

        api_key = self._ark_api_key()
        if not api_key:
            raise RuntimeError("missing BASE_API_KEY")

        base_url = self._ark_base_url()
        url = f"{base_url}/responses"

        max_keyword = 2
        try:
            max_keyword = int((os.getenv("ARK_WEB_SEARCH_MAX_KEYWORD") or "2").strip() or "2")
        except Exception:
            max_keyword = 2
        if max_keyword < 1:
            max_keyword = 1
        if max_keyword > 50:
            max_keyword = 50

        payload = {
            "model": self._ark_web_search_model(),
            "stream": False,
            "tools": [
                {
                    "type": "web_search",
                    "max_keyword": max_keyword,
                    "limit": k,
                }
            ],
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": q}],
                }
            ],
        }

        timeout_s = self.timeout if self.timeout > 30 else 30
        try:
            timeout_s = int((os.getenv("ARK_WEB_SEARCH_TIMEOUT") or "").strip() or str(timeout_s))
        except Exception:
            timeout_s = self.timeout if self.timeout > 30 else 30
        if timeout_s < 5:
            timeout_s = 5
        if timeout_s > 120:
            timeout_s = 120

        last_exc: Optional[Exception] = None
        for _ in range(2):
            try:
                resp = requests.post(
                    url,
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=timeout_s,
                )
                last_exc = None
                break
            except Exception as e:
                last_exc = e
        if last_exc is not None:
            raise last_exc
        if resp.status_code != 200:
            body = (resp.text or "")[:2000]
            raise RuntimeError(f"ark_web_search_http_status={resp.status_code} body={body}")

        data = resp.json() if resp.content else {}
        output = data.get("output") or []
        results: List[Dict[str, Any]] = []

        def _add(title: str, link: str, snippet: str) -> None:
            t = (title or "").strip()
            l = (link or "").strip()
            s = (snippet or "").strip()
            if not l and not t and not s:
                return
            results.append({"title": t or l or "无标题", "link": l, "snippet": s, "source": "ark_web_search"})

        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message":
                continue
            content = item.get("content") or []
            for part in content:
                if not isinstance(part, dict):
                    continue
                annotations = part.get("annotations") or []
                for ann in annotations:
                    if not isinstance(ann, dict):
                        continue
                    url = (ann.get("url") or ann.get("link") or ann.get("source_url") or "").strip()
                    title = (ann.get("title") or ann.get("source_title") or "").strip()
                    snippet = (ann.get("snippet") or ann.get("quote") or ann.get("text") or "").strip()
                    if url:
                        _add(title, url, snippet)

        dedup = []
        seen = set()
        for r in results:
            key = (r.get("link") or "") or (r.get("title") or "")
            if not key:
                continue
            if key in seen:
                continue
            seen.add(key)
            dedup.append(r)
        return dedup[:k]

    def search_web_sync(self, query: str, max_results: int = None, mode: Optional[str] = None) -> List[Dict[str, Any]]:
        if not max_results:
            max_results = self.max_results

        q = (query or "").strip()
        try:
            k = int(max_results)
        except Exception:
            k = self.max_results
        if k < 1:
            k = 1
        if k > 50:
            k = 50

        m = (mode or os.getenv("WEB_SEARCH_MODE") or "ark").strip().lower()
        if m in ("metaso", "metaso_mcp", "mcp"):
            try:
                return _metaso_mcp_web_search_sync(q, k, self.timeout)
            except Exception:
                return _ddg_text_search_sync(q, k, self.timeout)
        if m in ("ark", "doubao", "ark_web_search", "builtin"):
            try:
                return self._search_web_ark(q, k)
            except Exception as e:
                strict = (os.getenv("WEB_SEARCH_STRICT") or "").strip() in ("1", "true", "True")
                if strict:
                    raise
                msg = str(e)
                if "ToolNotOpen" in msg or "not activated web search" in msg or "missing BASE_API_KEY" in msg:
                    try:
                        if (os.getenv("METASO_API_KEY") or "").strip():
                            return _metaso_mcp_web_search_sync(q, k, self.timeout)
                    except Exception:
                        pass
                    return _ddg_text_search_sync(q, k, self.timeout)
                raise
        if m in ("local", "ddg"):
            return _ddg_text_search_sync(q, k, self.timeout)
        if m in ("bing",):
            return _bing_text_search(q, k, self.timeout)
        return self._search_web_ark(q, k)

    def read_webpage_sync(self, url: str, output_format: str = "markdown", mode: Optional[str] = None) -> str:
        u = (url or "").strip()
        if not u:
            return ""

        fmt = (output_format or "").strip().lower()
        if fmt not in ("json", "markdown"):
            fmt = "markdown"

        m = (mode or os.getenv("WEB_SEARCH_MODE") or "ark").strip().lower()
        if m in ("metaso", "metaso_mcp", "mcp"):
            return _metaso_mcp_web_reader_sync(u, fmt, self.timeout)

        if (os.getenv("METASO_API_KEY") or "").strip():
            try:
                return _metaso_mcp_web_reader_sync(u, fmt, self.timeout)
            except Exception:
                return ""
        return ""

    def batch_read_webpages_sync(
        self, urls: List[str], output_format: str = "markdown", mode: Optional[str] = None
    ) -> Dict[str, str]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if not isinstance(urls, list) or not urls:
            return {}

        fmt = (output_format or "").strip().lower()
        if fmt not in ("json", "markdown"):
            fmt = "markdown"

        unique: List[str] = []
        seen = set()
        for u in urls:
            s = (u or "").strip()
            if not s:
                continue
            if s in seen:
                continue
            seen.add(s)
            unique.append(s)

        max_urls = 5
        try:
            max_urls = int((os.getenv("WEB_READ_MAX_URLS") or "5").strip() or "5")
        except Exception:
            max_urls = 5
        if max_urls < 1:
            max_urls = 1
        if max_urls > 20:
            max_urls = 20
        unique = unique[:max_urls]

        concurrency = 6
        try:
            concurrency = int((os.getenv("WEB_READ_CONCURRENCY") or "6").strip() or "6")
        except Exception:
            concurrency = 6
        if concurrency < 1:
            concurrency = 1
        if concurrency > 20:
            concurrency = 20

        out: Dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {ex.submit(self.read_webpage_sync, u, fmt, mode): u for u in unique}
            for f in as_completed(futures):
                u = futures.get(f) or ""
                try:
                    out[u] = f.result() or ""
                except Exception:
                    out[u] = ""
        return out
        
    async def search_web(self, query: str, max_results: int = None) -> List[Dict[str, Any]]:
        """执行网页搜索"""
        start_time = time.time()
        
        if not max_results:
            max_results = self.max_results
        try:
            max_results = int(max_results)
        except Exception:
            max_results = self.max_results
        if max_results < 1:
            max_results = 1
        if max_results > 10:
            max_results = 10
        q = (query or "").strip()
        if not q:
            return []

        mode = (os.getenv("WEB_SEARCH_MODE") or "ark").strip().lower()
        if mode in ("metaso", "metaso_mcp", "mcp"):
            try:
                return await asyncio.to_thread(_metaso_mcp_web_search_sync, q, max_results, self.timeout)
            except Exception:
                return await asyncio.to_thread(_ddg_text_search_sync, q, max_results, self.timeout)
        if mode in ("ark", "doubao", "ark_web_search", "builtin"):
            try:
                return await asyncio.to_thread(self._search_web_ark, q, max_results)
            except Exception as e:
                strict = (os.getenv("WEB_SEARCH_STRICT") or "").strip() in ("1", "true", "True")
                if strict:
                    raise
                msg = str(e)
                if "ToolNotOpen" in msg or "not activated web search" in msg or "missing BASE_API_KEY" in msg:
                    try:
                        if (os.getenv("METASO_API_KEY") or "").strip():
                            return await asyncio.to_thread(_metaso_mcp_web_search_sync, q, max_results, self.timeout)
                    except Exception:
                        pass
                    return await asyncio.to_thread(_ddg_text_search_sync, q, max_results, self.timeout)
                raise
            
        try:
            print(f"开始搜索: {q}")
            
            preferred = (os.getenv("DDG_BACKEND") or "").strip().lower()
            backends = [preferred, "lite", "html", "api", "auto"]
            seen = set()
            backend_order = []
            for b in backends:
                if b and b not in seen:
                    seen.add(b)
                    backend_order.append(b)

            last_error: Optional[Exception] = None
            for backend in backend_order:
                try:
                    ddgs = DDGS()
                    results: List[Dict[str, Any]] = []
                    search_results = ddgs.text(
                        q,
                        region=os.getenv("DDG_REGION", "wt-wt"),
                        max_results=max_results,
                        backend=backend,
                    )
                    for result in search_results:
                        results.append(
                            {
                                "title": result.get("title", ""),
                                "link": result.get("href", ""),
                                "snippet": result.get("body", ""),
                                "source": f"duckduckgo:{backend}",
                            }
                        )

                    if results:
                        search_time = time.time() - start_time
                        print(f"搜索完成，耗时: {search_time:.3f}s，找到 {len(results)} 个结果")
                        return results
                    continue
                except Exception as e:
                    last_error = e
                    continue

            if last_error:
                raise last_error
            return []
            
        except Exception as e:
            print(f"搜索失败: {e}")
            try:
                return _bing_text_search(q, max_results, self.timeout)
            except Exception as e2:
                print(f"Bing搜索失败: {e2}")
                return []
    
    async def search_news(self, query: str, max_results: int = None) -> List[Dict[str, Any]]:
        """搜索新闻"""
        start_time = time.time()
        
        if not max_results:
            max_results = self.max_results
            
        try:
            print(f"开始新闻搜索: {query}")
            
            ddgs = DDGS()
            results = []
            
            # 搜索新闻
            news_results = ddgs.news(
                query,
                region='wt-wt',
                max_results=max_results
            )
            
            # 解析结果
            for result in news_results:
                results.append({
                    "title": result.get("title", ""),
                    "link": result.get("url", ""),
                    "snippet": result.get("body", ""),
                    "date": result.get("date", ""),
                    "source": "duckduckgo_news"
                })
            
            search_time = time.time() - start_time
            print(f"新闻搜索完成，耗时: {search_time:.3f}s，找到 {len(results)} 个结果")
            
            return results
            
        except Exception as e:
            print(f"新闻搜索失败: {e}")
            return []
    
    async def search_images(self, query: str, max_results: int = None) -> List[Dict[str, Any]]:
        """搜索图片"""
        start_time = time.time()
        
        if not max_results:
            max_results = self.max_results
            
        try:
            print(f"开始图片搜索: {query}")
            
            ddgs = DDGS()
            results = []
            
            # 搜索图片
            image_results = ddgs.images(
                query,
                region='wt-wt',
                max_results=max_results
            )
            
            # 解析结果
            for result in image_results:
                results.append({
                    "title": result.get("title", ""),
                    "image_url": result.get("image", ""),
                    "source_url": result.get("source", ""),
                    "thumbnail": result.get("thumbnail", ""),
                    "source": "duckduckgo_images"
                })
            
            search_time = time.time() - start_time
            print(f"图片搜索完成，耗时: {search_time:.3f}s，找到 {len(results)} 个结果")
            
            return results
            
        except Exception as e:
            print(f"图片搜索失败: {e}")
            return []
    
    def _get_fallback_results(self, query: str) -> List[Dict[str, Any]]:
        """获取备用搜索结果"""
        return []
    
    def format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """格式化搜索结果"""
        if not results:
            return "未找到相关结果。"
        
        formatted = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "无标题")
            snippet = result.get("snippet", "无摘要")
            link = result.get("link", "")
            
            formatted.append(f"{i}. {title}")
            if snippet:
                formatted.append(f"   {snippet}")
            if link:
                formatted.append(f"   链接: {link}")
            formatted.append("")
        
        return "\n".join(formatted)
    
    async def batch_search(self, queries: List[str], max_results: int = None) -> Dict[str, List[Dict[str, Any]]]:
        """批量搜索"""
        tasks = []
        for query in queries:
            tasks.append(self.search_web(query, max_results))
        
        results = await asyncio.gather(*tasks)
        
        return dict(zip(queries, results))
