
# 项目总结

本项目是一个基于 LangGraph 的 Agent 服务端，围绕“图片语义检索 + 联网搜索 +（可选）网页抓取阅读”三类能力，提供一个可直接用 FastAPI 启动的对话 API。核心目标是把多工具编排、默认参数与边界检查、失败回退链路、以及可重复验证的脚本测试固化下来，方便后续扩展更多检索源与工具。

# 技术选型

## 框架与服务

- Web 框架：FastAPI（`backend/main_real.py`）
- Agent 编排：LangGraph（`backend/agent/graph_new_real.py`）
- LLM 接入：OpenAI SDK 兼容 Ark Base URL（`backend/agent/graph_new_real.py`）
- 向量数据库：Milvus（`backend/services/milvus_service.py`）
- 多模态编码：CLIP（ModelScope 模型加载，`backend/services/clip_service_local.py`）
- 联网搜索：优先使用 Ark 内置 `web_search`，不可用时回退到 Metaso MCP 或 DDG/Bing（`backend/services/search_service.py`）
- 网页阅读：使用 Metaso MCP `metaso_web_reader`（`backend/services/search_service.py`、`backend/agent/graph_new_real.py`）

## 依赖管理

- Python 依赖：`requirements.txt`
- 环境变量：`.env`

# 关键问题与解决思路

## 1. 方舟联网搜索未开通/未激活

现象：Ark 调用 `tools: web_search` 返回 `ToolNotOpen` 或类似错误。  
处理：`SearchService.search_web_sync()` 内做模式切换与回退链路，默认 `ark`，失败后按条件回退到 `metaso_mcp` 或本地 `ddg/bing`。
注意，目前应该不开放给个人用户，申请开通不通过

## 2. 搜索结果相关性与可用性

现象：某些搜索源可能返回泛化结果。  
处理：在 `search_service.py` 增加基于 query 的轻量过滤 `_filter_results_by_query()`，并对结果做去重与上限控制（`max_results` 统一约束到 1–50）。

## 3. 网页详情获取（抓取/阅读）

目的：当搜索摘要不足以支撑回答时，需要对指定链接进一步读取正文。  
处理：引入 Metaso MCP 的 `metaso_web_reader`，在 `SearchService.read_webpage_sync()` 封装 URL 校验、超时控制、长度截断等边界保护；在 Agent 增加 `web_read` 工具供模型按需调用。

# 流程设计

## 请求到响应（主链路）

- 用户请求进入 `POST /api/chat`（`backend/main_real.py`）
- 构造 `AgentState`（包含 `use_rag`、`use_search`、`top_k` 等开关）
- LangGraph 运行：
  - `agent_node()`：决定是否调用工具
  - `process_tool_results()`：执行工具（`rag_image_search` / `web_search` / `web_read` / `analyze_image`），写回 `state.images` / `state.search_results` / `state.tool_results`
  - 直到模型给出最终回答
- 返回结构化 JSON：`response`、`images`、`search_results`、`timing`、`metadata`

## 联网搜索回退链路

- `WEB_SEARCH_MODE=ark`：优先 Ark 内置联网搜索；失败且 `WEB_SEARCH_STRICT!=1` 时回退
- `WEB_SEARCH_MODE=metaso`：优先 Metaso MCP；失败回退 DDG/Bing
- `WEB_SEARCH_MODE=ddg` / `bing`：强制使用对应源

# 使用说明

## 1. 启动服务

在项目根目录（`Langgraph_smith`）执行：

`python -m uvicorn backend.main_real:app --host 127.0.0.1 --port 8000`

接口：

- `GET /`：加载 `frontend/index.html`
- `POST /api/chat`：表单字段 `text/use_rag/use_search/top_k`，可选 `image`
- `GET /api/config`：读取当前 Ark 配置与模式（仅进程内）
- `POST /api/config`：设置 Ark 配置与模式（仅进程内）

# 流式输出（默认）

`POST /api/chat` 默认开启 `stream=true`，以 SSE（`text/event-stream`）返回，且单条 `data:` 负载遵循 OpenAI `chat.completion.chunk` 结构（便于前端通用解析与兼容）。

## 1. 事件格式

- 内容增量：`choices[0].delta.content`（字符串片段）
- 工具调用增量：`choices[0].delta.tool_calls`（数组，支持分片拼接 arguments）
- 工具执行事件：`x_tool_event`（用于前端实时展示“调用了哪些工具/耗时/结果预览”）
- 最终汇总：`x_final`（包含最终 `response/images/search_results/timing/metadata`）
- 结束标记：`data: [DONE]`

## 2. 前端展示策略

`frontend/index.html` 使用 `fetch` + `ReadableStream` 逐行解析 SSE：

- 逐步拼接 `delta.content` 为 Markdown 源文本，并用 `requestAnimationFrame` 节流渲染，避免页面卡死
- 实时展示：
  - `tool_calls`：模型计划调用的工具与参数（OpenAI tool_calls 结构）
  - `x_tool_event`：每次工具执行的 `ok/elapsed_s/result_preview`
  - `usage`：`prompt_tokens/completion_tokens/total_tokens`（若上游不提供则后端估算并在最终事件返回）

## 3. 本地验证

已在 `tests/run_all.py` 增加流式端到端校验：

- 基础流式：断言能收到内容增量与最终 `x_final`
- LLM + 工具整合（有有效 `ARK_API_KEY` 且非 `rule_based`）：断言收到 `x_tool_event` 后仍能继续收到内容增量（即“模型结合工具结果后的输出”）

## 2. 环境变量（建议写入 .env）

- `ARK_API_KEY`：方舟 API Key
- `ARK_BASE_URL` / `ARK_API_BASE_URL`：方舟 API Base URL
- `ARK_WEB_SEARCH_MODEL`：用于联网搜索的模型名
- `WEB_SEARCH_MODE`：`ark|metaso|ddg|bing`
- `WEB_SEARCH_STRICT`：`1` 时搜索失败直接抛错，不回退
- `METASO_API_KEY`：Metaso MCP Key
- `METASO_MCP_URL`：默认 `https://metaso.cn/api/mcp`
- `METASO_SEARCH_SCOPE`：默认 `webpage`
- `WEB_READ_MAX_CHARS`：网页读取截断长度
- `WEB_READ_MAX_URLS`：批量抓取的最大 URL 数
- `WEB_READ_CONCURRENCY`：批量抓取并发数
- `WEB_SEARCH_FETCH_ALL`：`1` 时在 Agent web_search 后并行抓取全部链接正文并附加到结果

补充：如果不希望把 Key 写入文件，可通过前端页面的“🔑 大模型配置”面板或直接调用 `POST /api/config` 在进程内设置 `ARK_API_KEY/ARK_BASE_URL/ARK_MODEL`。

## 3. 统一测试与调试入口

已将原先散落的 `debug_*.py` / `test_*.py` 归档合并为：

- `tests/run_all.py`

执行（默认会自动拉起并关闭服务）：

`python tests/run_all.py`

常用参数：

- `python tests/run_all.py --no-server`：不启动服务（用于你已手动启动的情况）
- `python tests/run_all.py --web-search-mode metaso`：指定联网搜索模式
- `python tests/run_all.py --fetch-all`：开启搜索后批量抓取正文
- `python tests/run_all.py --skip-clip`：跳过 CLIP 编码用例
- `python tests/run_all.py --skip-milvus-debug`：跳过 Milvus 直接调试用例

# 目录结构（主文件）

- `backend/main_real.py`：FastAPI 入口（主入口）
- `backend/agent/graph_new_real.py`：LangGraph Agent 图、工具定义、工具执行逻辑
- `backend/services/milvus_service.py`：Milvus 连接与图片检索
- `backend/services/search_service.py`：联网搜索与网页阅读（Ark/Metaso/DDG/Bing）
- `backend/services/clip_service_local.py`：CLIP 编码与模型信息
- `tests/run_all.py`：统一测试与调试入口
