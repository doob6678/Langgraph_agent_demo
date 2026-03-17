# 流式 Markdown 图片渲染卡死问题与解决方案

## 1. 问题描述

在 LLM 的 Server-Sent Events (SSE) 流式传输过程中，Markdown 图片标签 `![alt](url)` 可能被拆分在多个 Chunk 中发送（例如 Chunk 1: `![`，Chunk 2: `alt](url)`）。前端使用的 Markdown 渲染引擎（如 marked.js）在接收到不完整的 Chunk（如 `![`）时，可能会尝试解析但失败，或者在渲染时将不完整的符号作为普通文本展示，导致用户看到类似 `![马库斯](` 的原始字符，甚至导致渲染逻辑卡死。

**症状**：
- 前端显示 `![...` 后停止更新。
- 图片无法正确显示。
- 控制台可能无报错，但渲染流中断。

## 2. 根本原因分析

1.  **LLM 生成的不确定性**：LLM 逐 Token 生成内容，无法保证 Markdown 语法结构的完整性。
2.  **流式传输的特性**：SSE 实时推送数据，前端每收到一个 Chunk 就尝试渲染。
3.  **前端解析器的局限性**：大多数 Markdown 解析器（如 marked.js）设计用于解析完整文档，对流式的不完整片段支持有限，尤其是在复杂的嵌套语法（如图片、链接）上。

## 3. 解决方案：后端语义缓冲 (Semantic Buffering)

既然前端难以处理不完整的语法，解决方案是在后端引入一个**中间件层**，负责拦截并缓冲特定的语法结构，直到其完整后再发送给前端。

### 核心设计：`ImageMarkdownBuffer`

我们设计了一个 `ImageMarkdownBuffer` 类，作为流式输出的过滤器。

#### 状态机逻辑
该缓冲器本质上是一个简单的状态机：
1.  **Normal 状态**：直接透传所有字符。
2.  **Buffering 状态**：当检测到 `![` 时进入此状态，将后续字符存入缓冲区，暂停发送。
3.  **Release 状态**：
    - **成功匹配**：当缓冲区内出现闭合的 `)` 且符合图片语法时，一次性发送完整的 `![alt](url)`。
    - **失败/超时**：如果缓冲区过长（超过阈值）或发现明显不符合图片语法的字符（如换行符），则认为不是图片，原样释放缓冲区内容。

#### 代码实现 (Python)

位置：`backend/agent/stream_ex/image_buffer.py`

```python
class ImageMarkdownBuffer:
    def __init__(self, max_buffer_size: int = 1000):
        self.buffer = ""
        self.max_buffer_size = max_buffer_size

    def process(self, chunk: str) -> str:
        # ... (逻辑实现)
        # 检测 ![
        # 缓冲直到 )
        # 释放完整标签
```

### 集成点

该缓冲器被集成在所有流式输出的路径上：
1.  **普通对话流**：`backend/agent/stream_ex/stream_chat_with_tools.py`
2.  **LangGraph流**：`backend/agent/stream_ex/graph_stream_impl.py`
3.  **工具回退流**：`backend/agent/fallback_ex/tool_fallback_planner.py`

## 4. 复用方法论 (Methodology)

### 原则：后端负责语义完整性 (Backend Ensures Semantic Integrity)

在流式传输（Streaming）场景下，如果传输的数据具有强语法结构（如 JSON, Markdown, Code Block），**不要依赖前端去处理不完整的片段**。后端应承担起“整形”的责任。

**最佳实践**：
1.  **识别边界 (Identify Boundaries)**：确定哪些语法结构被切分会引发前端渲染问题（如 `![...](...)`, ` ```...``` `, `$ ... $`）。
2.  **引入缓冲 (Introduce Buffering)**：在流式生成器（Generator）和网络响应（Response）之间插入一个缓冲层。
3.  **设置超时/逃逸 (Timeout/Escape)**：缓冲必须有最大长度或超时机制，防止因 LLM 生成错误导致缓冲区无限增长，最终阻塞整个流。
4.  **透明性 (Transparency)**：缓冲逻辑应尽量对业务逻辑透明，最好封装为装饰器或中间件类。

## 5. 调试文档与工具归档

为了验证修复效果并防止回归，我们创建了以下测试工具。这些脚本位于 `tests/` 目录下，可直接运行。

### 1. 单元测试：`tests/test_image_buffer.py`
**用途**：验证 `ImageMarkdownBuffer` 类的逻辑是否正确。
**测试场景**：
- 完整的图片标签被切分为多个 Chunk。
- 只有 `![` 但没有后续（非图片）。
- 多个图片标签连续出现。
- 文本中包含类似图片语法的干扰项。

**运行方法**：
```bash
python -m tests.test_image_buffer
```

### 2. 端到端流式验证：`tests/test_backend_stream_image.py`
**用途**：模拟真实客户端请求后端 API，验证流式输出是否包含了完整的图片标签。
**测试逻辑**：
- 发送包含“蓝色帽子”的请求。
- 监听 SSE 流。
- 检查每个 Chunk 是否以 `![` 结尾（如果发生，说明缓冲失败）。
- 打印所有 Chunk 以人工确认。

**运行方法**：
1.  启动后端服务：
    ```bash
    python -m uvicorn backend.main_real:app --host 127.0.0.1 --port 8000
    ```
2.  在另一个终端运行测试：
    ```bash
    python -m tests.test_backend_stream_image
    ```

### 3. 前端调试脚本：`tests/test_image_search.py`
**用途**：验证图搜图功能的后端逻辑（Milvus + CLIP）是否正常工作，确保图片能被检索到。

**运行方法**：
```bash
python -m tests.test_image_search
```

## 6. 总结

通过在后端引入 `ImageMarkdownBuffer`，我们将“处理不完整 Markdown 语法”的复杂性从前端（不可控的渲染引擎）转移到了后端（可控的字符串处理）。这不仅解决了当前的渲染卡死问题，也为未来处理 LaTeX 公式或其他复杂 Markdown 语法提供了通用的架构思路。
