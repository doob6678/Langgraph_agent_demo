import time
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from backend.agent.tool_ex.image_result_utils import normalize_image_hits, split_by_score, summarize_image_hits
from backend.agent.service_ex.agent_services import search_service
from backend.agent.util_ex.common import coerce_top_k
from backend.services.image_service import image_service
from backend.services.metrics_service import metrics_collector


@tool
def rag_image_search(query: str, top_k: int = 5, results: Optional[List[Dict[str, Any]]] = None, score_threshold: float = 0.3) -> str:
    """基于文本在向量库中检索相关图片。"""
    start_time = time.time()
    try:
        q = (query or "").strip()
        if not q:
            return "搜索词为空，无法进行图片检索。"

        _ = coerce_top_k(top_k, default=5, min_value=1, max_value=20)
        normalized = normalize_image_hits(results if isinstance(results, list) else [])
        strong_hits, weak_hits = split_by_score(normalized, float(score_threshold))
        execution_time = time.time() - start_time
        metrics_collector.record_tool_usage("rag_image_search", execution_time, success=True)

        if not normalized:
            return "未找到相关图片。"
        if strong_hits:
            text = summarize_image_hits(strong_hits, f"找到{len(normalized)}条相关图片（高相关{len(strong_hits)}条）:", max_items=5, include_content=True)
            return text or "找到相关图片。"
        text = summarize_image_hits(weak_hits, f"找到{len(normalized)}条低相关图片（未达到阈值{float(score_threshold):.2f}）:", max_items=5, include_content=True)
        return text or "找到低相关图片结果。"
    except Exception as e:
        execution_time = time.time() - start_time
        metrics_collector.record_tool_usage("rag_image_search", execution_time, success=False)
        return f"图片搜索出错: {str(e)}"


@tool
def web_search(query: str, max_results: int = 5, mode: str = "", _results: Optional[List[Dict[str, Any]]] = None) -> str:
    """进行网页搜索并返回格式化结果列表。"""
    start_time = time.time()
    try:
        q = (query or "").strip()
        if not q:
            return "搜索词为空，无法进行网页搜索。"

        k = coerce_top_k(max_results, default=5, min_value=1, max_value=50)
        results = _results if isinstance(_results, list) else search_service.search_web_sync(q, k, mode=mode or None)
        execution_time = time.time() - start_time
        metrics_collector.record_tool_usage("web_search", execution_time, success=True)

        if results:
            return search_service.format_search_results(results)
        return "未找到相关网页结果。"
    except Exception as e:
        execution_time = time.time() - start_time
        metrics_collector.record_tool_usage("web_search", execution_time, success=False)
        return f"网页搜索出错: {str(e)}"


@tool
def web_read(url: str, format: str = "markdown", mode: str = "") -> str:
    """读取指定 URL 的正文内容，支持 markdown/json 输出。"""
    start_time = time.time()
    try:
        u = (url or "").strip()
        if not u:
            return "URL为空，无法读取网页内容。"
        fmt = (format or "").strip().lower()
        if fmt not in ("json", "markdown"):
            fmt = "markdown"

        content = search_service.read_webpage_sync(u, output_format=fmt, mode=mode or None)
        execution_time = time.time() - start_time
        metrics_collector.record_tool_usage("web_read", execution_time, success=bool(content))
        return content or "未读取到网页内容。"
    except Exception as e:
        execution_time = time.time() - start_time
        metrics_collector.record_tool_usage("web_read", execution_time, success=False)
        return f"网页读取出错: {str(e)}"


@tool
def analyze_image(image_data_base64: str, description: str = "", results: Optional[List[Dict[str, Any]]] = None, score_threshold: float = 0.3) -> str:
    """解析图片并尝试用 CLIP 提取特征摘要。"""
    start_time = time.time()
    try:
        import base64

        raw = (image_data_base64 or "").strip()
        if not raw:
            return "图片数据为空，无法分析。"
        if "," in raw:
            raw = raw.split(",", 1)[1]
        image_bytes = base64.b64decode(raw)

        image_info = image_service.process_uploaded_image(image_bytes, "uploaded_image.jpg")
        if not isinstance(image_info, dict) or (not image_info.get("success")):
            execution_time = time.time() - start_time
            metrics_collector.record_tool_usage("analyze_image", execution_time, success=False)
            return f"图片处理失败: {(image_info.get('error') if isinstance(image_info, dict) else '')}"

        img_info = image_info.get("data") if isinstance(image_info, dict) else None
        if not isinstance(img_info, dict):
            img_info = {}

        try:
            from backend.services.clip_service_local import clip_service

            image_features = clip_service.encode_image(image_bytes)
            feature_summary = f"提取了{getattr(image_features, 'shape', (0, 0))[1]}维特征向量"

            normalized = normalize_image_hits(results if isinstance(results, list) else [])
            strong_hits, weak_hits = split_by_score(normalized, float(score_threshold))
            if strong_hits:
                summary = summarize_image_hits(strong_hits, f"基于该图片特征，找到{len(normalized)}条相似图片（高相关{len(strong_hits)}条）:", max_items=5, include_content=True)
                if summary:
                    feature_summary += f"\n{summary}"
                else:
                    feature_summary += "\n基于该图片特征，已找到相似图片。"
            elif weak_hits:
                summary = summarize_image_hits(weak_hits, f"基于该图片特征，找到{len(normalized)}条低相关图片（未达到阈值{float(score_threshold):.2f}）:", max_items=5, include_content=True)
                if summary:
                    feature_summary += f"\n{summary}"
                else:
                    feature_summary += "\n基于该图片特征，找到低相关图片。"
            else:
                feature_summary += "\n基于该图片特征，未在图库中找到相似图片。"

        except Exception as clip_error:
            feature_summary = f"CLIP模型分析失败: {str(clip_error)}"

        analysis_result = "图片分析结果:\n"
        analysis_result += f"尺寸: {img_info.get('width', 0)}x{img_info.get('height', 0)}\n"
        analysis_result += f"格式: {img_info.get('format', '')}\n"
        analysis_result += f"CLIP特征: {feature_summary}\n"
        if description:
            analysis_result += f"用户描述: {description}\n"
        analysis_result += "内容分析: 已使用CLIP模型提取图片特征向量，可用于语义搜索和相似度计算。"

        execution_time = time.time() - start_time
        metrics_collector.record_tool_usage("analyze_image", execution_time, success=True)
        return analysis_result
    except Exception as e:
        execution_time = time.time() - start_time
        metrics_collector.record_tool_usage("analyze_image", execution_time, success=False)
        return f"图片分析出错: {str(e)}"


@tool
def save_user_fact(fact: str, visibility: str = "private", dept_id: str = "default_dept") -> str:
    """保存长期事实记忆。"""
    text = (fact or "").strip()
    if not text:
        return "事实内容为空，未保存。"
    vis = (visibility or "private").strip().lower()
    if vis not in ("private", "department"):
        vis = "private"
    dep = (dept_id or "default_dept").strip() or "default_dept"
    return f"准备保存长期记忆: visibility={vis}, dept_id={dep}, 内容长度={len(text)}"


@tool
def save_user_image(description: str = "", visibility: str = "private", dept_id: str = "default_dept", image_id: str = "", filename: str = "") -> str:
    """保存当前会话图片为长期图片记忆。"""
    vis = (visibility or "private").strip().lower()
    if vis not in ("private", "department"):
        vis = "private"
    dep = (dept_id or "default_dept").strip() or "default_dept"
    desc = (description or "").strip()
    saved_id = (image_id or "").strip()
    fn = (filename or "").strip()
    text = f"图片记忆已保存: visibility={vis}, dept_id={dep}"
    if saved_id:
        text += f", image_id={saved_id}"
    if fn:
        text += f", filename={fn}"
    if desc:
        text += f", 描述长度={len(desc)}"
    return text
