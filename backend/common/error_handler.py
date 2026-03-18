import uuid
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

"""
请求 ID 生成函数
逻辑优先级：
先从请求头 x-request-id 读取（外部调用方传入的请求 ID）
若没有，读取 request.state 中存储的请求 ID（中间件已生成）
最后兜底生成一个 UUID4 格式的随机字符串
作用：确保每个请求都有唯一标识，方便日志追踪和问题定位
"""
def request_id_from(request: Request) -> str:
    rid = (request.headers.get("x-request-id") or "").strip()
    if rid:
        return rid
    state_rid = getattr(request.state, "request_id", None)
    if state_rid:
        return str(state_rid)
    return uuid.uuid4().hex


"""
错误响应体构造函数 error_body
字段说明：
success：布尔值，快速判断请求是否成功
request_id：链路追踪核心字段
code：错误类型编码，前端可根据该字段做不同的错误处理逻辑
"""
def error_body(request_id: str, message: str, detail: Any, code: str) -> Dict[str, Any]:
    return {
        "success": False,
        "done": True,
        "request_id": request_id,
        "message": message,
        "code": code,
        "detail": detail,
    }


async def request_context_middleware(request: Request, call_next):
    request_id = request_id_from(request)
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    response.headers["X-Request-Complete"] = "1"
    return response


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def _http_exception_handler(request: Request, exc: HTTPException):
        request_id = request_id_from(request)
        body = error_body(request_id, "请求处理失败", exc.detail, "HTTP_ERROR")
        return JSONResponse(
            status_code=exc.status_code,
            content=body,
            headers={"X-Request-Id": request_id, "X-Request-Complete": "1"},
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(request: Request, exc: RequestValidationError):
        request_id = request_id_from(request)
        body = error_body(request_id, "请求参数校验失败", exc.errors(), "VALIDATION_ERROR")
        return JSONResponse(
            status_code=422,
            content=body,
            headers={"X-Request-Id": request_id, "X-Request-Complete": "1"},
        )

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception):
        request_id = request_id_from(request)
        body = error_body(request_id, "服务内部异常", str(exc), "INTERNAL_ERROR")
        return JSONResponse(
            status_code=500,
            content=body,
            headers={"X-Request-Id": request_id, "X-Request-Complete": "1"},
        )
