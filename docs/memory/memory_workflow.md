# 记忆管理工作流与生命周期文档 (Memory Workflow & Lifecycle)

## 1. 概述
本文档详细描述了记忆管理系统在 Agent 运行过程中的完整工作流程，包括记忆的形成、存储、召回、遗忘以及多租户权限控制的实现细节。

---

## 2. 核心工作流程

### 2.1 会话初始化 (Session Initialization)
当用户发起新的请求 (`POST /api/chat`) 时：
1.  **身份鉴权**:
    *   API 网关解析 JWT Token，提取 `user_id` 和 `tenant_id` (部门ID)。
    *   校验用户是否属于该部门。
2.  **状态加载 (State Loading)**:
    *   后端根据 `thread_id` 从 MySQL (`checkpoints` 表) 加载上一次的 `AgentState`。
    *   如果是新会话，初始化空的 `AgentState`，并记录 `user_id` 和 `tenant_id` 到 `chat_threads` 表。

### 2.2 记忆召回 (Memory Recall) - *Before Execution*
在 Agent 执行具体逻辑之前，`MemoryManager` 并行执行以下操作：
1.  **短期上下文 (Short-Term)**:
    *   从 `AgentState` 中提取最近 K 轮对话记录 (`messages`)。
2.  **长期记忆检索 (Long-Term)**:
    *   **语义检索**: 使用当前用户输入 (Query) 生成 Embedding 向量。
    *   **权限过滤**: 在 Milvus 中执行混合查询：
        ```python
        expr = f"tenant_id == {current_tenant} and (visibility == 'department' or (visibility == 'private' and user_id == {current_user}))"
        results = milvus.search(vector, expr=expr, limit=5)
        ```
    *   **重排序 (Rerank)**: (可选) 对检索结果进行相关性重打分。
3.  **上下文注入**:
    *   将检索到的长期记忆片段与短期对话历史合并，注入到 System Prompt 中：
        > "Here are some relevant memories: {long_term_memories}..."

### 2.3 记忆形成与存储 (Memory Formation) - *After Execution*
当 Agent 生成回复后，异步触发记忆更新流程：
1.  **短期记忆更新**:
    *   将最新的 User Message 和 AI Response 追加到 `AgentState`。
    *   **滑动窗口**: 如果消息列表超过 N 条，触发 Summarization (摘要) 机制，将旧消息压缩为摘要保留。
    *   **持久化**: 将更新后的 `AgentState` 保存到 MySQL (`checkpoints` 表)。
2.  **长期记忆提取 (Insight Extraction)**:
    *   **触发条件**: 每 N 轮对话或检测到关键信息（如用户偏好、重要事实）。
    *   **提取过程**: 调用 LLM 分析当前对话：
        > "Extract key facts, user preferences, or important tasks from the conversation."
    *   **存储策略**:
        *   **普通记忆**: 生成向量 -> 存入 Milvus (`agent_long_term_memory`)。
        *   **超长记忆**: 如果内容 > 8192 字符 -> 存入 MySQL (`memory_contents`) -> Milvus 仅存摘要和 ID。
        *   **多模态记忆**: 如果包含图片 -> 提取 CLIP 特征 -> 存入 Milvus (`agent_image_memory`) -> 图片文件存入 OSS。

### 2.4 记忆遗忘与维护 (Forgetting & Maintenance)
1.  **主动遗忘**: 用户显式请求删除某条记忆 -> `DELETE /api/memory/{id}` -> 物理删除 Milvus 和 MySQL 对应记录。
2.  **被动衰减**: (可选) 根据 `timestamp` 计算记忆的时间衰减权重，过旧且不重要的记忆将被定期清理或归档。
3.  **冲突修正**: 当提取的新事实与旧记忆冲突时（如用户更改了居住地），LLM 决策更新旧记忆而非简单的追加。

---

## 3. 时序图 (Sequence Diagram)

```mermaid
sequenceDiagram
    participant U as User
    participant API as API Gateway
    participant MM as Memory Manager
    participant DB as MySQL (State/Content)
    participant Vec as Milvus (Vector)
    participant LLM as LLM Service

    U->>API: Send Message (thread_id)
    API->>MM: Load Context (user_id, tenant_id)
    
    par Parallel Recall
        MM->>DB: Load Short-Term History
        MM->>Vec: Semantic Search (w/ Filter)
    end
    
    MM->>LLM: Generate Response (Prompt + Memories)
    LLM-->>U: Stream Response
    
    Note over MM, Vec: Async Background Process
    MM->>LLM: Analyze for New Memories
    LLM-->>MM: Extracted Facts/Insights
    
    alt Content is Short
        MM->>Vec: Insert Vector + Content
    else Content is Long
        MM->>DB: Insert Full Content
        MM->>Vec: Insert Vector + Summary + DB_ID
    end
    
    MM->>DB: Checkpoint State (Save Thread)
```

## 4. 关键技术实现细节

### 4.1 混合存储策略 (Hybrid Storage)
```python
async def save_long_term_memory(content: str, metadata: dict):
    if len(content) > 8000:
        # 1. 存入 MySQL 长文本表
        mem_id = await mysql.insert_content(content)
        # 2. 生成摘要
        summary = await llm.summarize(content)
        # 3. 存入 Milvus (带标记)
        await milvus.insert(vector, content=summary, has_ext=True, memory_id=mem_id)
    else:
        # 直接存入 Milvus
        await milvus.insert(vector, content=content, has_ext=False)
```

### 4.2 权限过滤表达式 (Permission Expression)
Milvus 查询时的动态表达式构建：
```python
def build_expr(user: User):
    # 基础条件：必须在同一个租户/部门下
    base_expr = f"tenant_id == {user.tenant_id}"
    
    # 可见性条件：
    # 1. 部门公开 (visibility == 'department')
    # 2. 或者是用户自己的私有记忆 (visibility == 'private' && user_id == me)
    # 3. (可选) 全局公开 (visibility == 'public')
    visibility_expr = f"(visibility == 'department' or (visibility == 'private' and user_id == {user.id}))"
    
    return f"{base_expr} and {visibility_expr}"
```
