# API 接口设计文档 - 记忆管理模块

## 1. 基础信息
*   **版本**: v1.0
*   **协议**: RESTful API / JSON
*   **认证**: Bearer Token (JWT)

## 2. 核心接口

### 2.1 会话管理 (Chat & Session)

#### 2.1.1 发起对话
*   **Method**: `POST /api/chat`
*   **Description**: 用户发起新的一轮对话，或继续现有会话。
*   **Request Body**:
    ```json
    {
      "message": "你好，我想查一下昨天的天气",
      "thread_id": "uuid-v4", // 会话 ID (必填)
      "user_id": "user-123",  // 用户 ID (必填，可通过 Token 解析)
      "use_rag": true,        // 是否启用知识库检索
      "use_memory": true      // 是否启用记忆增强
    }
    ```
*   **Response**: Server-Sent Events (SSE) 流式响应。
    *   `event: delta` -> `data: {"content": "..."}`
    *   `event: done`

#### 2.1.2 获取会话历史
*   **Method**: `GET /api/chat/history`
*   **Description**: 获取指定会话的历史消息记录。
*   **Query Params**:
    *   `thread_id`: 会话 ID
    *   `limit`: 返回消息数量限制 (默认 20)
*   **Response**:
    ```json
    {
      "thread_id": "uuid-v4",
      "messages": [
        {"role": "user", "content": "你好", "timestamp": 1715000000},
        {"role": "assistant", "content": "你好！有什么我可以帮你的吗？", "timestamp": 1715000005}
      ]
    }
    ```

### 2.2 记忆管理 (Memory Operations)

#### 2.2.1 检索相关记忆 (Debug/Admin)
*   **Method**: `POST /api/memory/search`
*   **Description**: 根据查询词检索相关的长期记忆片段。
*   **Request Body**:
    ```json
    {
      "query": "上次我提到的那个项目叫什么？",
      "user_id": "user-123",
      "top_k": 3
    }
    ```
*   **Response**:
    ```json
    {
      "results": [
        {"content": "用户正在做一个叫 'LangGraph Demo' 的项目", "score": 0.85, "timestamp": ...},
        {"content": "用户喜欢使用 Python 开发后端", "score": 0.72, "timestamp": ...}
      ]
    }
    ```

#### 2.2.2 存储记忆 (Store Memory)
*   **Method**: `POST /api/memory/add`
*   **Description**: 
    *   **人工模式**: 允许用户或管理员手动向长期记忆库注入信息。
    *   **AI 模式 (工具调用)**: Agent 在对话过程中发现重要信息（如用户偏好、关键事实、待办事项）时，可主动调用此接口（或内部函数）将信息存入长期记忆。
*   **Request Body**:
    ```json
    {
      "content": "用户将在下周三去上海出差",
      "user_id": "user-123",
      "tenant_id": 1001,      // (新增) 归属部门ID
      "visibility": "private", // (新增) 可见性: private/department/public
      "type": "fact",         // fact, preference, todo, etc.
      "source": "ai_extracted" // (新增) 来源: manual/ai_extracted
    }
    ```
*   **Response**: `{"status": "success", "memory_id": "mem-789"}`

#### 2.2.3 删除/遗忘记忆
*   **Method**: `DELETE /api/memory/{memory_id}`
*   **Description**: 删除特定的记忆条目。
*   **Response**: `{"status": "success"}`

## 3. 错误码定义
*   `200`: 成功
*   `400`: 请求参数错误
*   `401`: 未授权 (Invalid Token)
*   `403`: 禁止访问 (非本人数据)
*   `429`: 请求过多 (Rate Limit Exceeded)
*   `500`: 服务器内部错误
