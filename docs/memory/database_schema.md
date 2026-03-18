# 数据库设计文档 (MySQL + Milvus) - 企业级增强版

## 1. 设计理念

本设计针对企业级生产环境进行了优化，核心解决以下问题：
1.  **多租户与数据隔离**：引入 `tenant_id` 和 `visibility` 字段，实现部门间的数据共享与员工间的隐私隔离。
2.  **大文本存储优化**：Milvus 仅存储摘要和中等长度内容，超长文本由 MySQL 或 OSS 接管。
3.  **资产解耦**：摒弃硬编码物理路径，采用 URI 或 Asset ID 实现图片等静态资源的解耦，方便后期迁移至 OSS/S3。

---

## 2. 关系型数据库 (MySQL)

### 2.1 租户表 (`tenants`) - *新增*
用于管理公司、部门或独立组织。

| 字段名 | 类型 | 描述 | 约束 |
| :--- | :--- | :--- | :--- |
| `id` | BIGINT | 租户唯一ID | PK, Auto Inc |
| `name` | VARCHAR(100) | 租户/部门名称 | Not Null |
| `created_at` | DATETIME | 创建时间 | Default NOW() |

### 2.2 用户表 (`users`)
用于存储用户基础信息和鉴权凭证。

| 字段名 | 类型 | 描述 | 约束 |
| :--- | :--- | :--- | :--- |
| `id` | BIGINT | 用户ID | PK, Auto Inc |
| `tenant_id` | BIGINT | 所属租户/部门ID | FK -> tenants.id, Index |
| `username` | VARCHAR(50) | 用户名 | Unique, Not Null |
| `password_hash` | VARCHAR(255) | 密码哈希 | Not Null |
| `role` | VARCHAR(20) | 角色 | 'admin', 'user', 'guest' |

### 2.3 会话表 (`chat_threads`)
存储会话元数据，关联 User 和 LangGraph 的 thread_id。

| 字段名 | 类型 | 描述 | 约束 |
| :--- | :--- | :--- | :--- |
| `thread_id` | CHAR(36) | 会话UUID | PK |
| `user_id` | BIGINT | 归属用户 | FK -> users.id, Index |
| `tenant_id` | BIGINT | 归属部门 | FK -> tenants.id |
| `title` | VARCHAR(100) | 会话标题 | Default 'New Chat' |

### 2.4 记忆长文本拓展表 (`memory_contents`) - *新增*
应对 Milvus 长度限制，用于存储超过 8192 字符的完整长文档或超长对话历史。

| 字段名 | 类型 | 描述 |
| :--- | :--- | :--- |
| `memory_id` | BIGINT | 关联 Milvus 的 memory_id (PK) |
| `full_content` | LONGTEXT | 完整文本内容 |

---

## 3. 向量数据库 (Milvus)

> **核心策略**：对 `tenant_id`, `user_id`, `visibility` 建立 **标量索引 (Scalar Index)**，利用 `Partition Key` (`tenant_id`) 实现物理级别的查询加速与隔离。

### 3.1 长期记忆集合 (`long_term_memories`)
存储从对话中提取的事实、偏好和知识库。

*   **Collection Name**: `agent_long_term_memory`
*   **Partition Key**: `tenant_id` (推荐，实现租户物理隔离)
*   **Metric Type**: `COSINE` (余弦相似度)
*   **Vector Index**: `HNSW`

| 字段名 | 类型 | 维度/长度 | 描述 | 约束/索引 |
| :--- | :--- | :--- | :--- | :--- |
| `memory_id` | Int64 | - | 记忆唯一ID | 主键 (AutoID) |
| `tenant_id` | Int64 | - | 租户/部门ID | **Partition Key**, 标量索引 |
| `user_id` | Int64 | - | 创建者用户ID | 标量索引 |
| `visibility`| VarChar | 20 | 可见性(`private`/`department`/`public`) | 标量索引 |
| `vector` | FloatVector | 1536/768 | 文本语义向量 | 向量索引 |
| `content` | VarChar | 8192 | 核心内容/摘要 (扩容至8K) | - |
| `has_ext` | Bool | - | 是否在 MySQL 中有扩展长文本 | - |
| `type` | VarChar | 32 | 类型 (fact/rule/summary) | 标量索引 |
| `timestamp` | Int64 | - | 记忆形成时间戳 | 标量索引 |

**查询逻辑示例**：
```python
# 查询属于当前部门的公开数据，或者属于自己的私有数据
expr = f"tenant_id == {current_tenant} and (visibility == 'department' or (visibility == 'private' and user_id == {current_user}))"
```

### 3.2 多模态图像集合 (`image_memories`)
存储用户上传或对话中生成的图片特征，结合 CLIP 使用。

*   **Collection Name**: `agent_image_memory`
*   **Partition Key**: `tenant_id`
*   **Metric Type**: `COSINE`

| 字段名 | 类型 | 维度/长度 | 描述 | 约束/索引 |
| :--- | :--- | :--- | :--- | :--- |
| `image_id` | Int64 | - | 图片ID | 主键 (AutoID) |
| `tenant_id` | Int64 | - | 租户/部门ID | **Partition Key**, 标量索引 |
| `user_id` | Int64 | - | 上传者用户ID | 标量索引 |
| `visibility`| VarChar | 20 | 可见性 | 标量索引 |
| `vector` | FloatVector | 512 | CLIP 图像特征向量 | 向量索引 |
| `description` | VarChar | 2048 | 图片描述/OCR文本 | - |
| `image_uri` | VarChar | 512 | **资源标识符 (如 `oss://bucket/img.jpg` 或相对路径)** | 替代原绝对路径 |
| `timestamp` | Int64 | - | 上传时间 | 标量索引 |
