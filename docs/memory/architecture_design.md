# 架构设计文档 - 记忆管理模块 (Memory Management Module)

## 1. 概述
本项目旨在构建一个具备长短期记忆能力的智能 Agent 系统。通过引入记忆管理模块，解决传统 Agent "聊完即忘" 的问题，实现跨会话的用户偏好记忆、历史事实回溯以及上下文连贯性。

## 2. 核心架构图

```mermaid
graph TD
    User[用户] --> API_Gateway[API 网关 / FastAPI]
    API_Gateway -->|鉴权 & 限流| Agent_Core[Agent 核心 (LangGraph)]
    
    subgraph Memory_System [记忆管理系统]
        Memory_Manager[记忆管理器 (Facade)]
        Short_Term[短期记忆 (Context Window)]
        Long_Term[长期记忆 (Vector Store)]
        Image_Mem[图像记忆 (CLIP)]
    end
    
    Agent_Core -->|读/写| Memory_Manager
    Memory_Manager -->|更新| Short_Term
    Memory_Manager -->|检索/存储| Long_Term
    Memory_Manager -->|特征提取| Image_Mem
    
    Short_Term -->|持久化| SQL_DB[(MySQL)]
    Long_Term -->|向量/摘要| Milvus[(Milvus)]
    Long_Term -->|超长文本| SQL_DB
    Image_Mem -->|向量| Milvus
    Image_Mem -->|文件存储| OSS/FS[OSS/Local]
    SQL_DB -->|缓存| Redis[(Redis)]
```

## 3. 模块职责

### 3.1 记忆管理器 (Memory Manager)
*   **位置**: `backend/agent/memory_ex/memory_manager.py`
*   **职责**: 
    *   作为 Agent 与记忆系统的唯一交互入口。
    *   协调长短期记忆的读写策略。
    *   负责记忆的生命周期管理（创建、更新、遗忘）。

### 3.2 短期记忆 (Short-Term Memory)
*   **位置**: `backend/agent/memory_ex/short_term_memory.py`
*   **职责**:
    *   维护当前会话的上下文窗口（Sliding Window）。
    *   提供历史消息摘要（Summarization）。
    *   利用 `LangGraph Checkpointer` 实现状态持久化。

### 3.3 长期记忆 (Long-Term Memory)
*   **位置**: `backend/agent/memory_ex/long_term_memory.py`
*   **职责**:
    *   存储用户画像、历史事实、重要知识点。
    *   基于语义向量（Embedding）进行相关性检索。
    *   定期整理和压缩历史信息。

## 4. 数据流向

1.  **用户输入**: 用户发送消息 -> API 网关 -> Agent。
2.  **记忆召回**: Agent -> Memory Manager -> (并行) 获取短期上下文 + 检索长期相关记忆。
3.  **决策生成**: Agent 结合系统提示词 + 记忆上下文 + 用户输入 -> LLM -> 生成回复/工具调用。
4.  **记忆存储**: 
    *   Agent 回复 -> Memory Manager。
    *   Memory Manager -> 更新短期窗口 (MySQL/Redis)。
    *   Memory Manager -> (异步) 分析是否包含重要信息 -> 存入长期记忆 (Milvus)。

## 5. 扩展性设计
*   **接口抽象**: 定义 `BaseMemory` 接口，支持未来替换底层存储（如从 Milvus 迁移到 PGVector）。
*   **多用户支持**: 所有记忆操作均绑定 `user_id`，确保数据隔离。
*   **插件化**: 记忆模块可作为独立插件，通过配置开关启用/禁用。
