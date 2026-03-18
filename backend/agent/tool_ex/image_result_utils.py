from typing import Any, Dict, List, Tuple


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def normalize_image_hits(items: Any) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        image_uri = str(item.get("image_uri") or "").strip()
        filename = str(item.get("filename") or metadata.get("filename") or image_uri or "unknown").strip() or "unknown"
        score = _to_float(item.get("score", item.get("similarity", 0.0)), 0.0)
        content = str(item.get("content") or "").strip()
        normalized.append(
            {
                "id": item.get("id"),
                "image_uri": image_uri,
                "filename": filename,
                "score": score,
                "similarity": score,
                "content": content,
                "metadata": metadata,
            }
        )
    return normalized


def split_by_score(items: List[Dict[str, Any]], threshold: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    strong: List[Dict[str, Any]] = []
    weak: List[Dict[str, Any]] = []
    for item in items:
        score = _to_float(item.get("score", 0.0), 0.0)
        if score >= threshold:
            strong.append(item)
        else:
            weak.append(item)
    return strong, weak


def summarize_image_hits(items: List[Dict[str, Any]], title: str, max_items: int = 5, include_content: bool = True) -> str:
    if not items:
        return ""
    lines: List[str] = []
    for idx, item in enumerate(items[:max_items], 1):
        filename = str(item.get("filename") or item.get("image_uri") or "unknown").strip() or "unknown"
        score = _to_float(item.get("score", item.get("similarity", 0.0)), 0.0)
        content = str(item.get("content") or "").strip()
        if include_content and content:
            lines.append(f"{idx}. {filename} (相似度: {score:.3f}) - {content}")
        else:
            lines.append(f"{idx}. {filename} (相似度: {score:.3f})")
    if not lines:
        return ""
    return f"{title}\n" + "\n".join(lines)
