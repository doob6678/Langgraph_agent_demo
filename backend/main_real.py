import os
import time
import json
import uuid
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, List

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from backend.agent.graph_new_real import agent_graph
from backend.common.error_handler import register_exception_handlers, request_context_middleware
from backend.services.metrics_service import MetricsCollector
from backend.services.milvus_service import MilvusService
from backend.services.search_service import SearchService

# 全局服务实例
metrics_collector = MetricsCollector()
milvus_service = MilvusService()
search_service = SearchService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    print("正在初始化服务... [VERSION: ImageMarkdownBuffer ENABLED]")
    
    # 1. 预加载 Embedding 模型 (避免首次请求卡顿)
    try:
        from backend.agent.node_ex.memory_node import MemoryManagerFactory
        print("[Embedding] Preloading model...")
        manager = MemoryManagerFactory.get_manager()
        if manager.embedding_model:
            manager.embedding_model.load_model()
        image_ready = manager.image_memory.ensure_collection_ready()
        print(f"[ImageMemory] collection='{manager.image_memory._resolve_collection_name()}' ready={image_ready}")
        print("[Embedding] Model preloaded successfully")
    except Exception as e:
        print(f"[Embedding] Preload failed: {e}")

    # 2. 测试Milvus连接
    try:
        milvus_service.test_connection()
        print("[milvus] connection ok")
    except Exception as e:
        print(f"[milvus] connection failed: {e}")
    
    preload_clip = (os.getenv("CLIP_PRELOAD") or "").strip().lower() in ("1", "true", "yes", "y")
    if preload_clip:
        try:
            from backend.services.clip_service_local import clip_service
            model_info = clip_service.get_model_info()
            print(f"[clip] loaded: {model_info}")
        except Exception as e:
            print(f"[clip] load failed: {e}")
    
    yield
    
    # 关闭时清理
    print("正在关闭服务...")

# 创建FastAPI应用
app = FastAPI(
    title="LangGraph Agent API",
    description="基于LangGraph的智能Agent API，支持图片搜索、网页搜索和图片分析",
    version="1.0.0",
    lifespan=lifespan
)

app.middleware("http")(request_context_middleware)
register_exception_handlers(app)

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_frontend_dir = os.path.join(_project_root, "frontend")
# TODO 这个根本没用 挂载静态文件
# app.mount("/static", StaticFiles(directory="backend/static"), name="static")
app.mount("/frontend", StaticFiles(directory=_frontend_dir), name="frontend")
# 图像显示位置
_default_assets_dir = os.path.join(_project_root, "data", "images")
_assets_dir = (os.getenv("ASSETS_DIR") or _default_assets_dir).strip()
if _assets_dir:
    os.makedirs(_assets_dir, exist_ok=True)
    app.mount("/assets", StaticFiles(directory=_assets_dir), name="assets")

# 请求模型
class ChatRequest(BaseModel):
    message: str
    use_rag: bool = True
    use_search: bool = True
    top_k: int = 5

class ChatResponse(BaseModel):
    response: str
    images: List[Dict[str, Any]] = []
    search_results: List[Dict[str, Any]] = []
    timing: Dict[str, float] = {}
    metadata: Dict[str, Any] = {}

def _estimate_tokens(text: str) -> int:
    t = (text or "").strip()
    if not t:
        return 0
    return max(1, int(len(t) / 4))

def _sse_data(obj: Any) -> bytes:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")

def _sse_done() -> bytes:
    return b"data: [DONE]\n\n"

def _openai_chunk(chat_id: str, created: int, model: str, delta: Dict[str, Any], finish_reason: Optional[str] = None) -> Dict[str, Any]:
    return {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }

def _normalize_tool_calls_for_stream(tool_calls: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    if not isinstance(tool_calls, list):
        return normalized
    for i, tc in enumerate(tool_calls):
        if isinstance(tc, dict):
            call_id = (tc.get("id") or f"call_{uuid.uuid4().hex[:24]}").strip()
            fn = tc.get("function") or {}
            name = (fn.get("name") or tc.get("name") or "").strip()
            args = fn.get("arguments") if "arguments" in fn else tc.get("arguments")
        else:
            call_id = (getattr(tc, "id", None) or f"call_{uuid.uuid4().hex[:24]}").strip()
            fn = getattr(tc, "function", None)
            name = (getattr(fn, "name", None) or "").strip()
            args = getattr(fn, "arguments", None)
        if isinstance(args, dict):
            args_str = json.dumps(args, ensure_ascii=False)
        elif args is None:
            args_str = "{}"
        else:
            args_str = str(args)
        if not name:
            name = "unknown"
        normalized.append(
            {
                "index": i,
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": args_str},
            }
        )
    return normalized

def _upload_default_identity_allowed() -> bool:
    if _is_env_true("ALLOW_DEFAULT_UPLOAD", default=False):
        return True
    return _is_env_true("ALLOW_DEFAULT_UPLOAD_IN_TEST", default=False)

def _is_env_true(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return bool(default)
    return raw in ("1", "true", "yes", "y", "on")

def _resolve_identity(user_id: Optional[str], dept_id: Optional[str]) -> tuple[str, str]:
    env_user = (os.getenv("DEFAULT_UPLOAD_USER_ID") or "default_user").strip() or "default_user"
    env_dept = (os.getenv("DEFAULT_UPLOAD_DEPT_ID") or "default_dept").strip() or "default_dept"
    resolved_user = (user_id or "").strip()
    resolved_dept = (dept_id or "").strip()
    if not resolved_user:
        if _is_env_true("USE_ENV_DEFAULT_IDENTITY", default=True):
            resolved_user = env_user
        else:
            resolved_user = f"anon_{uuid.uuid4().hex}"
            print(f"[Session] Generated new anonymous user_id: {resolved_user}")
    if not resolved_dept:
        resolved_dept = env_dept
    return resolved_user, resolved_dept

async def _stream_chat_response(state: Any, request_id: str):
    from backend.agent.stream_ex.graph_stream_impl import stream_chat_graph
    
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    try:
        async for chunk in stream_chat_graph(state, chat_id, created):
            yield chunk
    except Exception as e:
        err = {
            "success": False,
            "done": True,
            "request_id": request_id,
            "message": "流式处理异常",
            "code": "STREAM_ERROR",
            "detail": str(e),
        }
        yield _sse_data(
            {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "doubao-seed-2-0-lite-260215",
                "choices": [],
                "x_error": err,
            }
        )
        yield _sse_data(_openai_chunk(chat_id, created, "doubao-seed-2-0-lite-260215", {}, finish_reason="stop"))
        yield _sse_done()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """主页"""
    try:
        with open(os.path.join(_frontend_dir, "index.html"), "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>前端页面未找到</h1>")

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: Request,
    text: str = Form(...),
    use_rag: bool = Form(True),
    use_search: bool = Form(True),
    top_k: int = Form(5),
    stream: bool = Form(True),
    user_id: Optional[str] = Form(None),
    dept_id: Optional[str] = Form(None),
    visibility: str = Form("private"),
    image: Optional[UploadFile] = File(None)
):
    """聊天API - 支持FormData格式"""
    start_time = time.time()
    
    user_id, dept_id = _resolve_identity(user_id=user_id, dept_id=dept_id)
    
    try:
        # 处理图片数据
        image_data = None
        image_filename = None
        if image:
            image_data = await image.read()
            image_filename = image.filename
            if (user_id == "default_user" or dept_id == "default_dept") and not _upload_default_identity_allowed():
                raise HTTPException(status_code=400, detail="上传保存失败：user_id/dept_id 为只读默认值，请使用可写值。")
        
        # 创建Agent状态
        from backend.agent.graph_new_real import AgentState
        state = AgentState(
            messages=[],
            user_input=text,
            image_data=image_data,
            image_filename=image_filename,
            tool_flags=[bool(use_rag), bool(use_search)],
            top_k=top_k,
            user_id=user_id,
            dept_id=dept_id,
            visibility=visibility
        )

        if bool(stream):
            return StreamingResponse(
                _stream_chat_response(state, str(getattr(request.state, "request_id", "") or uuid.uuid4().hex)),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # 异步调用逻辑
        result = await agent_graph.ainvoke(state)
        # 结果处理需要根据实际 graph 的输出结构调整
        # 假设 result 是一个 dict，包含了 state 的所有字段更新
        # 通常我们关心 'messages' 中的最后一条 AI 回复
        
        # 尝试从 messages 获取
        answer = ""
        if result.get("messages"):
            last_msg = result["messages"][-1]
            if hasattr(last_msg, "content"):
                answer = last_msg.content
            elif isinstance(last_msg, dict):
                answer = last_msg.get("content", "")
        
        # 如果 graph 直接返回 answer 字段 (取决于 graph 定义)
        if not answer and result.get("answer"):
            answer = result.get("answer")
            
        response_data = ChatResponse(
            response=answer or "抱歉，我没有生成回复。",
            images=result.get("images", []),
            search_results=result.get("search_results", []),
            timing=result.get("timing", {}) if isinstance(result.get("timing"), dict) else {},
            metadata=result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {},
        )
        response_data.timing["total_time"] = float(time.time() - start_time)
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"处理请求时出错: {str(e)}\n{traceback.format_exc()}"
        print(f"[error] detail: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)

@app.post("/api/chat_with_image", response_model=ChatResponse)
async def chat_with_image_endpoint(
    request: Request,
    message: str = Form(...),
    image: UploadFile = File(...),
    use_rag: bool = Form(True),
    use_search: bool = Form(True),
    top_k: int = Form(5),
    stream: bool = Form(True),
    user_id: Optional[str] = Form(None),
    dept_id: Optional[str] = Form(None),
    visibility: str = Form("private"),
):
    """带图片的聊天API"""
    start_time = time.time()
    
    user_id, dept_id = _resolve_identity(user_id=user_id, dept_id=dept_id)

    try:
        # 读取图片数据
        image_data = await image.read()
        if (user_id == "default_user" or dept_id == "default_dept") and not _upload_default_identity_allowed():
            raise HTTPException(status_code=400, detail="上传保存失败：user_id/dept_id 为只读默认值，请使用可写值。")
        
        # 创建Agent状态
        from backend.agent.graph_new_real import AgentState
        state = AgentState(
            messages=[],
            user_input=message,
            image_data=image_data,
            image_filename=image.filename,
            tool_flags=[bool(use_rag), bool(use_search)],
            top_k=top_k,
            user_id=user_id,
            dept_id=dept_id,
            visibility=visibility
        )

        if bool(stream):
            return StreamingResponse(
                _stream_chat_response(state, str(getattr(request.state, "request_id", "") or uuid.uuid4().hex)),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # 异步调用逻辑
        result = await agent_graph.ainvoke(state)
        # 结果处理需要根据实际 graph 的输出结构调整
        # 假设 result 是一个 dict，包含了 state 的所有字段更新
        # 通常我们关心 'messages' 中的最后一条 AI 回复
        
        # 尝试从 messages 获取
        answer = ""
        if result.get("messages"):
            last_msg = result["messages"][-1]
            if hasattr(last_msg, "content"):
                answer = last_msg.content
            elif isinstance(last_msg, dict):
                answer = last_msg.get("content", "")
        
        # 如果 graph 直接返回 answer 字段 (取决于 graph 定义)
        if not answer and result.get("answer"):
            answer = result.get("answer")
            
        response_data = ChatResponse(
            response=answer or "抱歉，我没有生成回复。",
            images=result.get("images", []),
            search_results=result.get("search_results", []),
            timing=result.get("timing", {}) if isinstance(result.get("timing"), dict) else {},
            metadata=result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {},
        )
        response_data.timing["total_time"] = float(time.time() - start_time)
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")

@app.get("/api/metrics")
async def get_metrics():
    """获取性能指标"""
    return metrics_collector.get_metrics()


def _to_datetime_text(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            try:
                return datetime.fromtimestamp(int(text)).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                return text
        return text
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(int(value)).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return ""
    return ""


def _to_timestamp(value: Any) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value or "").strip()
    if not text:
        return 0
    if text.isdigit():
        return int(text)
    try:
        dt = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
        return int(dt.timestamp())
    except Exception:
        return 0


async def _query_memory_images(user_id: str, dept_id: str, limit: int, visibility: str = "") -> Dict[str, Any]:
    from backend.agent.node_ex.memory_node import MemoryManagerFactory

    manager = MemoryManagerFactory.get_manager()
    rows = await manager.list_user_images(user_id=user_id, dept_id=dept_id, limit=limit, visibility=visibility)
    images = []
    for row in rows:
        uri = str(row.get("image_uri") or "").strip()
        if uri and not (uri.startswith("http://") or uri.startswith("https://") or uri.startswith("/assets/")):
            uri = f"/assets/{uri.lstrip('/')}"
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        created_at = _to_datetime_text(row.get("created_at") or metadata.get("created_at") or "")
        updated_at = _to_datetime_text(row.get("updated_at") or metadata.get("updated_at") or "")
        images.append(
            {
                "id": row.get("id"),
                "user_id": row.get("user_id"),
                "dept_id": row.get("dept_id"),
                "visibility": row.get("visibility"),
                "image_uri": uri,
                "content": row.get("content"),
                "filename": metadata.get("filename") or metadata.get("stored_filename") or uri,
                "metadata": metadata,
                "created_at": created_at,
                "updated_at": updated_at,
                "created_at_ts": _to_timestamp(created_at),
                "updated_at_ts": _to_timestamp(updated_at),
            }
        )
    return {"items": images, "count": len(images)}

@app.get("/api/memory/images")
async def get_memory_images(
    user_id: Optional[str] = None,
    dept_id: Optional[str] = None,
    limit: int = 50,
    visibility: str = "",
):
    user_id, dept_id = _resolve_identity(user_id=user_id, dept_id=dept_id)
    return await _query_memory_images(user_id=user_id, dept_id=dept_id, limit=limit, visibility=visibility)


@app.get("/api/memory/images/query")
async def query_memory_images(
    user_id: Optional[str] = None,
    dept_id: Optional[str] = None,
    limit: int = 50,
    visibility: str = "",
):
    user_id, dept_id = _resolve_identity(user_id=user_id, dept_id=dept_id)
    return await _query_memory_images(user_id=user_id, dept_id=dept_id, limit=limit, visibility=visibility)

@app.get("/api/memory/facts")
async def get_memory_facts(
    user_id: Optional[str] = None,
    dept_id: Optional[str] = None,
    limit: int = 50,
    visibility: str = "",
    include_image_summary: bool = False,
):
    from backend.agent.node_ex.memory_node import MemoryManagerFactory

    user_id, dept_id = _resolve_identity(user_id=user_id, dept_id=dept_id)
    manager = MemoryManagerFactory.get_manager()
    include_types = None if include_image_summary else ["fact"]
    rows = await manager.list_user_facts(
        user_id=user_id,
        dept_id=dept_id,
        limit=limit,
        visibility=visibility,
        include_types=include_types,
    )
    facts = []
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        facts.append(
            {
                "id": row.get("id"),
                "user_id": row.get("user_id"),
                "dept_id": row.get("dept_id"),
                "visibility": row.get("visibility"),
                "content": row.get("content"),
                "metadata": metadata,
                "created_at": row.get("created_at") or metadata.get("created_at") or 0,
            }
        )
    return {"items": facts, "count": len(facts)}

@app.get("/api/config")
async def get_config():
    from backend.agent import graph_new_real as g

    cfg = g.configure_model()
    return cfg

@app.post("/api/config")
async def set_config(
    base_api_key: Optional[str] = Form(None),
    base_url: Optional[str] = Form(None),
    base_model: Optional[str] = Form(None),
    provider: Optional[str] = Form(None),
):
    from backend.agent import graph_new_real as g

    cfg = g.configure_model(
        api_key=base_api_key,
        base_url=base_url,
        model=base_model,
        provider=provider,
    )
    return cfg

@app.get("/api/health")
async def health_check():
    """健康检查"""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {
            "milvus": False,
            "clip": False
        }
    }
    
    # 检查Milvus连接
    try:
        milvus_service.test_connection()
        health_status["services"]["milvus"] = True
    except:
        pass
    
    try:
        from backend.services.clip_service_local import clip_service
        health_status["services"]["clip"] = bool(clip_service.is_loaded())
    except Exception:
        pass
    
    return health_status

@app.get("/api/images/{image_id}")
async def get_image(image_id: str):
    """获取图片"""
    try:
        # TODO 这里应该实现从Milvus获取图片的逻辑
        # 暂时返回404
        raise HTTPException(status_code=404, detail="图片未找到")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取图片失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # 为了避免端口冲突，尝试使用 8002
    print("启动服务器在 http://127.0.0.1:8002")
    uvicorn.run(app, host="127.0.0.1", port=8002)
