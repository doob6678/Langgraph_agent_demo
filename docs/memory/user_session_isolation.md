# 用户与会话隔离机制设计 (User & Session Isolation)

在具备记忆增强能力的 Agent 系统中，如何准确识别用户并保证多用户、多会话之间的数据隔离是核心挑战。本项目采用基于 **`user_id`** 和 **`dept_id` (部门ID)** 的双重隔离机制。

## 1. 标识传递与兜底机制

系统通过前端发起的 API 请求携带标识符，以此作为后续所有操作的基础依据。

### 1.1 前端传递
在前端发起聊天请求时（`POST /api/chat`），通过 `FormData` 显式传递 `user_id` 和 `dept_id`。
*   **开发联调状态**：为了方便测试跨刷新、跨会话的记忆持久化，前端 `script.js` 中通常会固定写入一个测试标识：
    ```javascript
    // frontend/js/script.js
    formData.append('user_id', 'test_user_001'); 
    ```

### 1.2 后端兜底与匿名机制
如果前端未传递 `user_id`（例如新访客首次访问），后端的 FastAPI 接口 (`main_real.py`) 以及 LangGraph 的 `AgentState` 初始化逻辑中，会自动生成一个基于 UUID 的匿名 ID。
```python
# backend/main_real.py
if not user_id:
    # 使用 UUID 生成唯一会话ID，防止多匿名用户记忆串扰
    user_id = f"anon_{uuid.uuid4().hex}"
```
这种设计保证了即使是未登录用户，在当前会话窗口内也能拥有连续的短期记忆，但刷新页面后会生成新的匿名 ID，从而表现为“记忆重置”。

## 2. 记忆系统的双重隔离

后端接收到标识后，会通过 `MemoryManagerFactory` 进行分发与隔离处理。

### 2.1 部门隔离 (Department Isolation)
系统原生支持多部门隔离架构：
*   `MemoryManagerFactory` 为每个 `dept_id` 维护独立的 `MemoryManager` 单例实例。
*   在真实业务场景中，不同部门的数据库表前缀、向量库 Collection 或 Partition 可以完全物理/逻辑隔离。

### 2.2 用户隔离 (User Isolation)
在 `MemoryManager` 内部，所有底层存储的增删改查操作都强制绑定 `user_id`。

*   **短期记忆 (`ShortTermMemory`)**：
    存储在关系型数据库（如 SQLite/MySQL）或内存结构中。在查询时，必须附带条件过滤：
    ```python
    # 始终只拉取当前用户的最近记录
    records = db.query(ShortTermMemoryModel).filter(ShortTermMemoryModel.user_id == user_id).limit(K)
    ```
*   **长期事实记忆 (`LongTermMemory` / `ImageMemory`)**：
    存储在向量数据库（如 Milvus）中。存入时，`user_id` 作为 Metadata (Scalar field) 一并存入。
    在执行向量相似度检索时，使用表达式进行强过滤，确保不会召回其他用户的私有记忆：
    ```python
    expr = f"user_id == '{user_id}'"
    results = milvus.search(vector, expr=expr, limit=3)
    ```

## 3. 会话 (Session) 与跨会话记忆的本质

在传统的不带记忆的 LangGraph 或对话系统中，通常依靠 `thread_id` 来维护单个聊天窗口的状态。但在本系统的**记忆增强**架构下：

1.  **淡化单次会话边界**：只要 `user_id` 保持一致，Agent 的 `memory_recall_node` 就会自动拉取该用户最近的**短期对话历史**（Context，如前 10 条）以及相关的**长期知识**（Facts）。
2.  **跨设备/跨刷新连贯性**：这意味着即使用户刷新了浏览器页面（前端丢失了所有本地 DOM 状态），只要重新传入相同的 `user_id`，Agent 依然能从后端数据库和向量库中“回想”起之前的聊天上下文。

## 4. 生产环境建议

目前的机制设计已能跑通核心流程。在未来推向生产环境时，建议进行如下安全加固：
1.  **禁止前端明文传递 `user_id`**：前端只需传递登录凭证（如 JWT Token）。
2.  **API 网关解析**：由 API 网关或后端的鉴权中间件拦截请求，解析 Token 提取出真实的 `user_id` 和 `dept_id`，再注入到后续的业务逻辑和 LangGraph 状态机中，防止用户伪造 ID 越权访问他人记忆。
3.  **统一字段治理**：前后端字段命名统一为 `dept_id`，避免接口联调歧义。
