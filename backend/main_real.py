import os
import time
import json
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from backend.agent.graph_new_real import agent_graph
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
    
    # 测试Milvus连接
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

# TODO 这个根本没用 挂载静态文件
# app.mount("/static", StaticFiles(directory="backend/static"), name="static")
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
# 图像显示位置
_default_assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "image_search", "img_marcus"))
_assets_dir = (os.getenv("ASSETS_DIR") or _default_assets_dir).strip()
if _assets_dir and os.path.isdir(_assets_dir):
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

async def _stream_chat_response(state: Any, start_time: float):
    from backend.agent.stream_ex.graph_stream_impl import stream_chat_graph
    
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    
    async for chunk in stream_chat_graph(state, chat_id, created):
        yield chunk

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """主页"""
    try:
        with open("frontend/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>前端页面未找到</h1>")

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    text: str = Form(...),
    use_rag: bool = Form(True),
    use_search: bool = Form(True),
    top_k: int = Form(5),
    stream: bool = Form(True),
    image: Optional[UploadFile] = File(None)
):
    """聊天API - 支持FormData格式"""
    start_time = time.time()
    
    try:
        # 处理图片数据
        image_data = None
        image_filename = None
        if image:
            image_data = await image.read()
            image_filename = image.filename
        
        # 创建Agent状态
        from backend.agent.graph_new_real import AgentState
        state = AgentState(
            messages=[],
            user_input=text,
            image_data=image_data,
            image_filename=image_filename,
            tool_flags=[bool(use_rag), bool(use_search)],
            top_k=top_k
        )

        if bool(stream):
            return StreamingResponse(
                _stream_chat_response(state, start_time),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # 同步调用逻辑
        result = agent_graph.invoke(state)
        response = ChatResponse(
            response=result.get("answer", ""),
            images=result.get("images", []),
            search_results=result.get("search_results", []),
            timing=result.get("timing", {}),
            metadata=result.get("metadata", {}),
        )
        response.timing["total_time"] = float(time.time() - start_time)
        return response
        
    except Exception as e:
        import traceback
        error_detail = f"处理请求时出错: {str(e)}\n{traceback.format_exc()}"
        print(f"[error] detail: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)

@app.post("/api/chat_with_image", response_model=ChatResponse)
async def chat_with_image_endpoint(
    message: str = Form(...),
    image: UploadFile = File(...),
    use_rag: bool = Form(True),
    use_search: bool = Form(True),
    top_k: int = Form(5),
    stream: bool = Form(True),
):
    """带图片的聊天API"""
    start_time = time.time()
    
    try:
        # 读取图片数据
        image_data = await image.read()
        
        # 创建Agent状态
        from backend.agent.graph_new_real import AgentState
        state = AgentState(
            messages=[],
            user_input=message,
            image_data=image_data,
            image_filename=image.filename,
            tool_flags=[bool(use_rag), bool(use_search)],
            top_k=top_k
        )

        if bool(stream):
            return StreamingResponse(
                _stream_chat_response(state, start_time),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        result = agent_graph.invoke(state)
        response = ChatResponse(
            response=result.get("answer", ""),
            images=result.get("images", []),
            search_results=result.get("search_results", []),
            timing=result.get("timing", {}),
            metadata=result.get("metadata", {}),
        )
        response.timing["total_time"] = float(time.time() - start_time)
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")

@app.get("/api/metrics")
async def get_metrics():
    """获取性能指标"""
    return metrics_collector.get_metrics()

@app.get("/api/config")
async def get_config():
    from backend.agent import graph_new_real as g

    cfg = g.configure_ark()
    return cfg

@app.post("/api/config")
async def set_config(
    ark_api_key: Optional[str] = Form(None),
    ark_base_url: Optional[str] = Form(None),
    ark_model: Optional[str] = Form(None),
):
    from backend.agent import graph_new_real as g

    cfg = g.configure_ark(api_key=ark_api_key, base_url=ark_base_url, model=ark_model)
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
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
