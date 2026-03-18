# LangGraph Smith 多模态记忆 Agent

本项目是一个面向企业级 RAG/Agent 场景的后端与前端一体化示例，核心能力包括：

- 对话式 Agent（流式输出、工具编排）
- 多模态能力（图像分析、图搜图）
- 长短期记忆（会话上下文 + 长期事实 + 图像记忆）
- 联网搜索与网页阅读（多搜索源回退）
- 可观测的工具执行事件流（前端实时展示）

项目目标是提供一套可运行、可扩展、可验证的参考实现，便于后续继续扩展为生产架构。

---

## 1. 项目完整功能（当前已实现）

### 1.1 对话与工具编排

- 基于 FastAPI 提供统一接口，默认支持 SSE 流式响应。
- 基于 LangGraph 状态机实现 Agent 节点、工具执行节点、记忆召回/存储节点的协作。
- 支持工具调用事件实时回传：`x_tool_event`、最终汇总事件 `x_final`。

### 1.2 多模态能力

- 支持用户上传图片参与对话。
- 使用本地 CLIP 模型做图像特征提取。
- 支持以图搜图、图文混合检索结果生成。
- 前端对 Markdown 图片语法做容错修复，降低模型输出格式不稳定带来的渲染失败。

### 1.3 记忆系统（分层存储）

- **短期记忆（Context）**：保存近期对话窗口，支持上下文连续性（MySQL 会话侧）。
- **长期记忆（Long-term）**：结构化事实仅写入 Milvus，单条记忆内容上限 1024 字符。
- **图像记忆（Images）**：独立图像集合存储与查询，支持 ACL 可见性控制。
- 新增并落地记忆分层元数据：`metadata.type`，当前已区分：
  - `fact`：事实层
  - `conversation`：对话摘要层
  - `image_summary`：图像摘要层
- Facts 页面默认仅展示事实层，避免图像摘要重复混入。

### 1.4 Embedding 与多模态模型

- 文本 Embedding：`LocalEmbedding` 基于 ModelScope `sentence_embedding` 管线，默认模型 `damo/nlp_corom_sentence-embedding_chinese-base`。
- 图像 Embedding：`CLIPService` 基于 `CLIPForMultiModalEmbedding`，默认模型 `damo/multi-modal_clip-vit-large-patch14_zh`。
- 模型加载策略：支持本地缓存优先与延迟加载，服务启动阶段可预加载，降低首请求抖动。
- 线程安全与边界兜底：CLIP 服务使用 `RLock` 保护模型推理，`TEXT_MAX_LEN`、分辨率与设备选择均有默认值与边界处理。

### 1.5 搜索与网页阅读

- 联网搜索支持多模式：
  - `ark`
  - `metaso`
  - `ddg`
  - `bing`
- 搜索失败可按策略回退（受 `WEB_SEARCH_STRICT` 控制）。
- 支持网页正文读取与截断、并发控制、异常兜底。

### 1.6 前端体验

- Chat 流式打字机渲染。
- 工具调用过程可视化（执行中/完成/结果预览）。
- 记忆侧边栏三分区：
  - Context
  - Facts
  - Images
- 修复并增强了记忆面板加载逻辑：
  - Facts 与 Images 使用独立缓存身份键，避免互相影响
  - 身份切换自动刷新对应面板

### 1.7 可测试性与回归

- 提供统一测试入口 `tests/run_all.py`。
- 支持自动拉起服务或复用外部运行服务（`--no-server`）。
- 支持按能力开关跳过特定测试（如 CLIP、Milvus 调试）。

---

## 2. 核心架构与分层

### 2.1 技术栈

- Web 框架：FastAPI
- Agent 编排：LangGraph
- 模型接入：OpenAI Compatible API
- 向量数据库：Milvus
- 关系存储：MySQL（短期会话与业务关系数据）
- 多模态模型：CLIP（本地）
- 前端：原生 JS + SSE 流式解析

### 2.2 关键模块

- `backend/main_real.py`：API 入口、配置接口、记忆查询接口
- `backend/agent/memory_ex/*`：短期/长期/图像记忆与统一管理器
- `backend/agent/tool_ex/*`：工具策略与执行上下文
- `backend/agent/stream_ex/*`：SSE 事件流封装与输出
- `backend/common/error_handler.py`：全局异常处理器、中间件 request_id 注入与统一错误体
- `frontend/js/script.js`：聊天流渲染、工具状态展示
- `frontend/js/memory_ui.js`：记忆面板（Context/Facts/Images）

### 2.3 设计模式落地

- **策略模式（Strategy）**：`ToolExecutor` 通过 `tool_name` 路由到 `WebSearchStrategy`、`WebReadStrategy`、`AnalyzeImageStrategy`、`SaveUserFactStrategy` 等实现，新增工具只需新增策略类并注册。
- **门面模式（Facade）**：`MemoryManager` 统一编排 `ShortTermMemory`、`LongTermMemory`、`ImageMemory`，对 Agent 层暴露单一记忆入口，隐藏底层差异与复杂性。
- **工厂 + 单例初始化**：`MemoryManagerFactory` 负责全局管理器和共享 Embedding 的懒加载，避免重复初始化与并发抖动。
- **防御式默认值**：策略执行中统一进行 `top_k`、参数解析、结果预览截断，降低异常输入导致的链路中断概率。

### 2.4 全局异常处理与流式兜底

- 统一异常注册：在 `main_real.py` 中集中注册 `register_exception_handlers(app)` 与 `request_context_middleware`。
- 非流式接口：统一返回 `success/done/request_id/code/message/detail` 结构，便于前端稳定消费。
- 流式接口：SSE 异常时输出 `x_error`，随后发送 `finish_reason=stop` 与 `[DONE]`，保证前端不假死。
- 可观测性：响应头统一写入 `X-Request-Id`、`X-Request-Complete`，便于链路追踪与排障。

---

## 3. 关键接口（当前可用）

### 3.1 对话接口

- `POST /api/chat`
  - 参数：`text/use_rag/use_search/top_k/stream`，可选 `image`
  - 返回：SSE 事件流（默认）或最终结果

### 3.2 记忆接口

- `GET /api/memory/facts`
  - 默认返回 `fact` 层
  - 可通过 `include_image_summary=true` 查看图像摘要层
- `GET /api/memory/images`
- `GET /api/memory/images/query`
- `GET /api/conversations/recent?user_id={user_id}&limit=5`：页面刷新加载最近会话（短期会话存储）

### 3.3 配置接口

- `GET /api/config`
- `POST /api/config`

---

## 4. 分层存储实现说明（最新）

本项目已完成“事实层与图像摘要层拆分”的主流程改造：

- 写入阶段：
  - 普通事实写入 `type=fact`
  - 图像伴随摘要写入 `type=image_summary`
- 查询阶段：
  - Facts 接口默认过滤到 `type=fact`
  - 召回阶段默认排除 `image_summary`，减少上下文噪声
- 展示阶段：
  - Facts 面板同时过滤列表数据与流式事件中的 `image_summary`

这套设计解决了“Facts 与 Images 内容重复展示”的核心问题，并为后续分层检索策略扩展预留了稳定入口。

---

## 5. 本地运行

在项目根目录执行：

```bash
python -m uvicorn backend.main_real:app --host 127.0.0.1 --port 8000
```

访问：

- `http://127.0.0.1:8000/`

---

## 6. 测试与验证

统一测试入口：

```bash
python tests/run_all.py
```

常用参数：

- `--no-server`：复用已启动服务
- `--web-search-mode metaso`：指定搜索模式
- `--fetch-all`：搜索后抓取全部网页正文
- `--skip-clip`：跳过 CLIP 相关用例
- `--skip-milvus-debug`：跳过 Milvus 调试用例

---

## 7. 环境变量（建议）

- `BASE_API_KEY`、`BASE_URL`、`BASE_MODEL`、`BASE_PROVIDER`
- `WEB_SEARCH_MODE`、`WEB_SEARCH_STRICT`
- `METASO_API_KEY`、`METASO_MCP_URL`、`METASO_SEARCH_SCOPE`
- `WEB_READ_MAX_CHARS`、`WEB_READ_MAX_URLS`、`WEB_READ_CONCURRENCY`
- `WEB_SEARCH_FETCH_ALL`
- `IMAGE_MEMORY_COLLECTION`

---

## 8. 待开发问题与演进路线

以下是当前建议优先级最高的待开发事项：

### P0（高优先级，生产阻断）

- [ ] **鉴权与身份体系重构**
  - 现状：前后端仍可通过显式 `user_id/dept_id` 传参驱动数据读取。
  - 风险：存在越权与跨部门数据泄漏风险。
  - 目标：改为 JWT + 网关/中间件注入身份，前端不再直接控制身份字段。

- [ ] **统一部门隔离策略**
  - 现状：多个接口已有 ACL，但策略定义分散。
  - 目标：抽象统一的部门/可见性策略层，避免“接口级漏校验”。

- [ ] **关键路径审计日志**
  - 目标：记录谁在何时读取/写入了哪些记忆对象，满足安全审计需求。

### P1（重要，质量与可维护性）

- [ ] **记忆生命周期管理**
  - 增加 TTL、归档、软删除、批量清理、冷热分层。

- [ ] **分层检索策略可配置化**
  - 支持按场景动态组合 `fact/conversation/image_summary`，并支持权重调度。

- [ ] **前后端契约与接口文档**
  - 补齐 OpenAPI + 示例请求/响应 + 错误码手册。

- [ ] **CI 自动化质量门禁**
  - 增加 PR 级别的单测、静态检查、基础冒烟。

### P2（增强项）

- [ ] **监控与告警**
  - 增加 Prometheus 指标、链路追踪、慢查询观测。

- [ ] **多模型路由与降级**
  - 支持不同模型按任务类型路由，并具备自动降级策略。

- [ ] **管理后台**
  - 记忆检索、纠错、人工标注、策略热更新。

---

## 9. 已知限制

- 部分联网搜索能力依赖外部服务开通状态，不同环境可用性存在差异。
- 本地 CLIP 与向量检索对机器资源有一定要求。
- 当前以“工程样例 + 可运行闭环”为主，距离严格生产形态仍需完成 P0/P1 改造。

---

## 10. 相关文档

- 问题与解决记录：`docs/memory/troubleshooting_and_solutions.md`
- 用户与会话隔离设计：`docs/memory/user_session_isolation.md`
- 设计文档目录：`docs/`
