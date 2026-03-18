# 全局异常处理设计文档

## 1. 设计目标

- 将异常处理从入口文件中解耦，形成独立的全局异常处理器。
- 后端返回结构化、可机器解析、可直接展示给前端的精确错误消息。
- 无论成功或失败，均为请求输出明确的完成信号，避免前端“等待中假死”。
- 统一非流式（JSON）与流式（SSE）异常协议，降低前端分支复杂度。

## 2. 适用范围

- HTTP 非流式接口（JSONResponse）。
- HTTP 流式接口（text/event-stream）。
- 框架异常（HTTPException、RequestValidationError）与业务未捕获异常（Exception）。

## 3. 统一返回格式设计

### 3.1 非流式 JSON 错误体

```json
{
  "success": false,
  "done": true,
  "request_id": "4f6a7c12f62d49d0ab8ebc40fd4f4692",
  "message": "参数校验失败",
  "code": "VALIDATION_ERROR",
  "detail": {
    "errors": [
      {
        "loc": ["body", "messages", 0, "role"],
        "msg": "field required",
        "type": "value_error.missing"
      }
    ]
  }
}
```

### 3.2 流式 SSE 错误帧

```text
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1700000000,"model":"doubao-seed-2-0-lite-260215","choices":[],"x_error":{"success":false,"done":true,"request_id":"chatcmpl-xxx","message":"流式处理异常","code":"STREAM_ERROR","detail":"具体异常信息"}}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1700000000,"model":"doubao-seed-2-0-lite-260215","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

## 4. 字段定义

| 字段 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| `success` | boolean | 是 | 是否成功，错误场景固定为 `false` |
| `done` | boolean | 是 | 请求是否已完成，错误场景固定为 `true` |
| `request_id` | string | 是 | 请求追踪ID；优先透传 `x-request-id`，否则后端生成 |
| `message` | string | 是 | 面向前端展示的简洁错误信息 |
| `code` | string | 是 | 稳定错误码，供前端策略分支与监控聚合 |
| `detail` | object/string | 否 | 详细错误信息；可用于调试或开发态展示 |
| `x_error` | object | SSE场景必填 | 流式错误载荷容器，结构同上 |

## 5. 错误码设计

| 错误码 | 场景 | HTTP状态码 |
| :--- | :--- | :--- |
| `HTTP_ERROR` | 显式抛出的 HTTPException | 透传异常状态码 |
| `VALIDATION_ERROR` | 请求参数/结构校验失败 | 422 |
| `STREAM_ERROR` | 流式生成过程异常 | 200（SSE通道内错误帧） |
| `INTERNAL_ERROR` | 未捕获系统异常 | 500 |

## 6. 处理流程设计

### 6.1 非流式流程

1. 请求进入中间件，生成/透传 `request_id`。
2. 业务处理抛出异常后进入全局异常处理器。
3. 处理器按异常类型映射 `code/message/detail`。
4. 返回统一 JSON 错误体，`done=true`。
5. 响应头写入 `X-Request-Id` 与 `X-Request-Complete: 1`。

### 6.2 流式流程

1. 请求进入流式接口，初始化 `chat_id` 作为 `request_id`。
2. 正常流式输出 chunk。
3. 一旦发生异常，立即输出 `x_error` 错误帧。
4. 紧接输出 `finish_reason=stop` 的停止帧。
5. 最后输出 `[DONE]`，确保前端立刻结束读取循环。

## 7. 前端消费约定

- 优先读取 `x_error.message` 作为用户提示文案。
- 开发态可展示 `x_error.detail`，生产态可按策略脱敏。
- 收到 `x_error` 后必须停止继续渲染 token。
- 收到 `[DONE]` 或 `done=true` 后必须结束请求生命周期并解锁UI状态。

## 8. 边界与安全策略

- 默认值策略：若未携带 `x-request-id`，后端必须生成UUID作为兜底。
- 空值安全：异常处理中不得假设 `request.state` 一定存在，需安全访问。
- 线程安全：异常处理器保持无状态实现，不持有可变共享上下文。
- 信息安全：`detail` 不返回密钥、连接串、堆栈敏感路径等敏感内容。

## 9. 落地拆分建议

- `backend/common/error_handler.py`：错误体构造、错误码映射、注册函数。
- `backend/main_real.py`：仅保留 `register_exception_handlers(app)` 调用与中间件接线。
- `frontend/js/script.js`：统一解析 `x_error` 与完成信号，单点终止流读取。

## 10. 验收标准

- 非流式异常统一返回 `success=false`、`done=true`、`request_id`、`code`、`message`。
- 流式异常在 1 次错误事件内完成“报错 + 停止 + DONE”闭环。
- 前端在异常场景不再卡住，且可展示可定位的精确错误信息。
