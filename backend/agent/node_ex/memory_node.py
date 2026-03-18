import logging
import os
import threading
from typing import Dict, Any, List, Optional

from backend.agent.state_ex.agent_state import AgentState
from backend.agent.memory_ex.memory_manager import MemoryManager
from backend.agent.memory_ex.embedding import LocalEmbedding
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)

def _normalize_image_uri(uri: Any) -> str:
    u = str(uri or "").strip()
    if not u:
        return ""
    lu = u.lower()
    if lu.startswith(("http://", "https://", "data:", "/assets/")):
        return u
    if u.startswith("/"):
        return u
    return f"/assets/{u.lstrip('/')}"

class MemoryManagerFactory:
    """
    MemoryManager 工厂类 (单例/享元模式)
    职责：
    1. 管理全局唯一 MemoryManager 实例
    2. 共享底层重资源 (如 Embedding Model)，避免内存爆炸
    3. 线程安全地创建和获取实例
    """
    _instance: Optional[MemoryManager] = None
    _shared_embedding_model: Optional[LocalEmbedding] = None
    _lock = threading.Lock()

    @classmethod
    def get_manager(cls) -> MemoryManager:
        if cls._instance is not None:
            return cls._instance

        with cls._lock:
            if cls._instance is None:
                logger.info("Factory: Creating global MemoryManager")
                if cls._shared_embedding_model is None:
                    logger.info("Factory: Initializing shared Embedding Model...")
                    try:
                        cls._shared_embedding_model = LocalEmbedding()
                        # 显式加载，确保初始化完成
                        cls._shared_embedding_model.load_model()
                    except Exception as e:
                        logger.error(f"Factory: Shared Embedding Model load failed: {e}")
                        cls._shared_embedding_model = None
                
                # 从环境变量读取短期记忆窗口大小
                import os
                stm_window = int(os.getenv("MAX_SHORT_TERM_WINDOW", "10"))
                
                cls._instance = MemoryManager(
                    stm_window_size=stm_window,
                    embedding_model=cls._shared_embedding_model
                )
        
        return cls._instance

def get_memory_manager() -> MemoryManager:
    """
    获取全局 MemoryManager 实例
    """
    return MemoryManagerFactory.get_manager()

async def memory_recall_node(state: AgentState) -> Dict[str, Any]:
    """
    记忆召回节点：
    1. 获取用户 ID 和当前输入
    2. 调用 MemoryManager 检索相关记忆
    3. 格式化记忆为 Context 字符串用于 Prompt
    4. 保留结构化数据用于前端展示
    """
    logger.info("--- 记忆召回节点 ---")
    
    user_id = state.user_id
    
    if not user_id:
        # 理论上 AgentState.__post_init__ 会处理，但为了双重保险
        import uuid
        user_id = f"anon_{uuid.uuid4().hex[:8]}"
        logger.warning(f"MemoryRecallNode 发现 user_id 为空，已生成临时 ID: {user_id}")
    
    # 获取最新的用户消息
    # state.messages 是 List[Any] (可能是 LangChain Message 或 dict)
    last_message = None
    if state.messages:
        last_message = state.messages[-1]
    
    # 兼容处理：如果是 dict，转换为对象或取 content
    query = ""
    if isinstance(last_message, HumanMessage):
        query = last_message.content
    elif isinstance(last_message, dict) and last_message.get("type") == "human":
        query = last_message.get("content", "")
    elif state.user_input:
        query = state.user_input
        
    if not query:
        logger.warning("未找到用户输入，跳过记忆召回")
        return {"memory_context": "", "memory_data": {}}

    logger.info(f"正在检索记忆，Query: {query}, User: {user_id}")
    
    manager = get_memory_manager()
    
    # 调用 MemoryManager 获取记忆
    memories = await manager.recall_context(user_id=user_id, current_query=query)
    
    short_term = memories.get("short_term", [])
    long_term = memories.get("long_term", [])
    image_memory = memories.get("image_memory", [])
    
    # 1. 构造结构化数据 (用于前端展示)
    memory_data = {
        "short_term": [
            {"role": m.get("role"), "content": m.get("content"), "time": m.get("created_at")} 
            for m in short_term
        ],
        "long_term": [
            {"content": m.get("content"), "relevance": m.get("score", 0)} 
            for m in long_term
        ],
        "images": [
            {
                "description": m.get("content"),
                "uri": _normalize_image_uri(m.get("image_uri")),
                "filename": ((m.get("metadata") or {}).get("filename") or ""),
            }
            for m in image_memory
        ]
    }
    
    # 2. 构造 Context 字符串 (用于 Prompt)
    context_parts = []
    
    if short_term:
        context_parts.append("【短期对话历史】:")
        for m in short_term:
            context_parts.append(f"- {m.get('role')}: {m.get('content')}")
            
    if long_term:
        context_parts.append("\n【长期知识回顾】:")
        for m in long_term:
            context_parts.append(f"- [长期记忆] {m.get('content')}")

    if image_memory:
        context_parts.append("\n【相关图片记忆】:")
        for m in image_memory:
            desc = m.get('content')
            uri = m.get('image_uri', '')
            context_parts.append(f"- [图片] {desc} (URI: {uri})")
            
    memory_context = "\n".join(context_parts)
    
    logger.info(f"检索完成，Context 长度: {len(memory_context)}")
    
    # 更新 state
    # 注意：LangGraph node 返回的 dict 会 update 到 state 中
    return {
        "memory_context": memory_context,
        "memory_data": memory_data
    }

async def memory_store_node(state: AgentState) -> Dict[str, Any]:
    """
    记忆存储节点：
    在 Agent 生成回复后，将交互记录存储到记忆中。
    """
    logger.info("--- 记忆存储节点 ---")
    
    user_id = state.user_id
    
    if not user_id:
        logger.error("MemoryStoreNode: user_id 为空，无法存储记忆")
        return {}
    
    # 获取用户输入和 AI 回复
    # 假设 messages 列表最后一条是 AI 回复，倒数第二条是用户输入
    # 或者直接使用 state.user_input 和 state.answer
    
    user_input = state.user_input
    ai_response = state.answer
    
    if not user_input or not ai_response:
        logger.warning("用户输入或 AI 回复为空，跳过记忆存储")
        return {}
        
    logger.info(f"正在存储记忆，User: {user_id}")
    
    manager = get_memory_manager()
    
    # 存储交互
    await manager.store_interaction(
        user_id=user_id,
        user_query=user_input,
        ai_response=ai_response
    )
    
    logger.info("记忆存储完成")
    return {}
