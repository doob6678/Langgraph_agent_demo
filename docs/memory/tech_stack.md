# 技术选型文档 - 记忆管理模块

## 1. 核心框架

### 1.1 语言模型 & Agent
*   **LangChain (v1.2.12) / LangGraph (v1.1.2)**: 提供 Agent 的状态管理、工具调用和流程编排。
*   **OpenAI API (openai v2.28.0)**: 接入 DeepSeek/Doubao/GPT-4 等大模型，作为推理引擎。

### 1.2 后端服务
*   **FastAPI (v0.115.8)**: 高性能异步 Web 框架。
*   **Uvicorn (v0.42.0)**: ASGI 服务器。
*   **Pydantic (v2.12.5)**: 数据校验和序列化。

## 2. 存储与数据库

### 2.1 关系型数据库 (MySQL)
*   **用途**: 持久化存储 Agent 的运行状态 (`checkpoint`)、用户数据、会话记录。
*   **版本**: MySQL 8.0+
*   **驱动**: `SQLAlchemy` + `aiomysql` (需添加到依赖)。

### 2.2 向量数据库 (Milvus)
*   **用途**: 存储长期记忆的向量索引、多模态数据索引。
*   **客户端**: `pymilvus (v2.6.1)`
*   **部署**: Docker / Cloud

### 2.3 缓存与消息队列 (Redis) - [第二阶段]
*   **用途**: 短期会话缓存、API 限流、分布式锁。
*   **客户端**: `redis-py`

## 3. 多模态与辅助工具

### 3.1 嵌入与多模态模型 (CLIP & Embedding)
*   **文本嵌入**: `text-embedding-3-small` 或 `bge-m3`。
*   **图像/文本多模态**: **CLIP (Contrastive Language-Image Pre-training)**
    *   **库**: `transformers (v4.44.2)`, `pillow (v12.1.1)`, `numpy (v2.4.3)`
    *   **用途**: 将图像和文本映射到同一向量空间，实现以文搜图、记忆中的图像检索。
    *   **实现**: 本地加载 CLIP 模型，提取 Image Embedding 存入 Milvus。

### 3.2 鉴权与安全
*   **JWT**: `python-multipart (v0.0.22)` 用于处理表单和上传。
*   **OAuth2**: 标准授权流程。

### 3.3 监控与日志
*   **LangSmith**: (可选)用于 Agent 的链路追踪和调试。
*   **Prometheus + Grafana**: 系统性能监控。
