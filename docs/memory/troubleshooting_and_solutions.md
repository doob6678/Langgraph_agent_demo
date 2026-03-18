# 遇到的问题与解决方案记录

在此次系统迭代尤其是记忆功能集成与前后端联调过程中，我们遇到并解决了以下关键问题：

## 1. 记忆上下文未能正确影响大模型回答
- **问题现象**：虽然底层 `memory_recall_node` 成功获取到了记忆数据，但前端问“我的名字是什么？”时，大模型依然回答“不知道”。
- **根本原因**：LangGraph 工作流中使用了**同步**的 `agent_node`，且在处理逻辑中，未将状态（State）中携带的 `memory_context` 动态拼接到 LLM 的 System Prompt 中。
- **解决方案**：
  1. 将同步节点重构为异步节点 `agent_node_async`。
  2. 在构建 Prompt 时，显式提取 `state.memory_context`，以特定格式（如 `【相关记忆上下文】...`）拼接到 System Message 的尾部，强制大模型参考这些信息。

## 2. 前端记忆面板（Memory Panel）始终为空白
- **问题现象**：页面右侧的记忆面板没有任何内容显示，即便后端已经检索到了历史记录。
- **根本原因**：
  1. 后端流式接口 `graph_stream_impl.py` 中，错误地在判断“只有当记忆非空时才发送 `x_memory_event`”，导致新用户的空状态无法通知前端进行初始化清空或渲染。
  2. 存在缩进错误和重复代码块，导致事件被截断或吞掉。
  3. 前端 `memory_ui.js` 中，过度依赖后端的 `contextStr` 字段，当该字段为空，但结构化的 `data.short_term` 有值时，前端没有提供降级（Fallback）的渲染逻辑。
- **解决方案**：
  1. 移除后端的“空数据检查”，**强制每次对话都下发一次 `x_memory_event`**（包括空状态）。
  2. 修复 Python 缩进与冗余代码。
  3. 在前端渲染逻辑中加入降级处理：优先显示 `contextStr`，若无则遍历渲染 `data.short_term` 数组；都没有则显示友好的提示“No recent context”。

## 3. 大模型流式输出（打字机效果）失效
- **问题现象**：提交问题后，前端一片空白，必须等所有处理完成或者干脆无法显示大模型的回答。
- **根本原因**：LangGraph 在使用 `astream_events` 时，如果 Agent 节点是**同步的**，则底层的 `on_chat_model_stream` 事件无法正确冒泡抛出，导致 `yield` 给前端的只有工具调用，没有文本分片（chunks）。
- **解决方案**：将图节点改为全异步（`agent_node_async`），并确保调用 `llm.bind_tools()` 时内部链条支持 `astream` 协议，从而正确派发 `chat.completion.chunk` 事件。

## 4. 浏览器 Network 面板“看不到数据包”的疑惑
- **问题现象**：开发者在浏览器的 Network 面板查看请求时，发现发起请求后一片空白或显示 pending，但页面确实在实时逐字显示内容。
- **根本原因**：这不是 Bug。由于前端采用了 `fetch` 发送 POST 请求并读取流（ReadableStream），Chrome 的 Network 工具在请求未完全结束前（连接未关闭），不会在 "Response" 标签实时显示 SSE 分块数据，这与传统的 GET `EventSource` 请求的 "EventStream" 标签页行为不同。
- **解决方案**：解释 Chrome DevTools 对 fetch stream 的显示机制，并通过前端 `console.log` 和页面实际渲染来验证数据的实时到达。

## 5. 端口占用导致服务启动失败
- **问题现象**：启动 FastAPI 服务时报 `[Errno 10048] error while attempting to bind on address ('127.0.0.1', 8000)`。
- **根本原因**：本地 8000/8001 端口被其他挂起的 Python 进程占用。
- **解决方案**：修改 `backend/main_real.py` 的启动脚本，将默认端口灵活切换（如 8002），或者使用系统命令清理残留进程。

## 6. 前端多次聊天导致记忆串掉/未持久化
- **问题现象**：刷新页面后，上一次聊天的记忆丢失。
- **根本原因**：前端之前没有固定或动态管理 `user_id`，每次请求可能是匿名的。
- **解决方案**：在 `script.js` 的 `FormData` 中统一附加一个测试用 `user_id`（如 `test_user_001`），确保后端的 `MemoryManagerFactory` 能够为该用户持续存取同一份记忆流。
