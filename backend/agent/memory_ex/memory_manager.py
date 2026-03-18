from typing import Any, Dict, List, Optional
import logging
import asyncio
from .short_term_memory import ShortTermMemory
from .long_term_memory import LongTermMemory
from .image_memory import ImageMemory
from .embedding import LocalEmbedding

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    记忆管理器 (Facade 模式)
    职责：
    - 作为 Agent 与记忆系统的唯一交互入口。
    - 协调长短期记忆的读写策略。
    - 负责记忆的生命周期管理（创建、更新、遗忘）。
    
    设计考量：
    - 对外隐藏子系统的复杂性（长短期分离、向量检索与普通关系检索的差异）。
    - 异步实现，防止阻塞 Agent 核心流程。
    """

    def __init__(self, stm_window_size: int = 10, embedding_model: Optional[LocalEmbedding] = None):
        """
        初始化记忆管理器。
        """
        logger.info("初始化 MemoryManager (Facade)")
        
        # 初始化共享 Embedding 模型，避免多次加载
        if embedding_model:
            self.embedding_model = embedding_model
        else:
            try:
                self.embedding_model = LocalEmbedding()
            except Exception as e:
                logger.error(f"MemoryManager Embedding 模型初始化失败: {e}")
                self.embedding_model = None

        # 实例化子模块，注入配置
        self.short_term = ShortTermMemory(max_window_size=stm_window_size)
        self.long_term = LongTermMemory(embedding_model=self.embedding_model)
        # ImageMemory uses its own CLIP service, so we don't pass the text embedding model
        self.image_memory = ImageMemory()

    async def recall_context(self, user_id: str, current_query: str, dept_id: str = "default_dept") -> Dict[str, Any]:
        """
        并行（或顺序）获取短期上下文与检索长期相关记忆。
        """
        if not user_id or not current_query:
            raise ValueError("user_id 和 current_query 不能为空")
            
        logger.info(f"开始召回记忆上下文: user={user_id}, query='{current_query}', dept={dept_id}")
        
        # 1. 获取短期记忆（会话上下文）
        stm_context = await self.short_term.get_memory(user_id=user_id, query="")
        
        # 2. 检索长期相关事实与知识
        ltm_context = await self.long_term.get_memory(
            user_id=user_id,
            query=current_query,
            dept_id=dept_id,
            limit=3,
            exclude_types=["image_summary"],
        )
        
        return {
            "short_term": stm_context,
            "long_term": ltm_context,
            "image_memory": []
        }

    async def list_user_images(self, user_id: str, dept_id: str, limit: int = 50, visibility: str = "") -> List[Dict[str, Any]]:
        if not user_id or not dept_id:
            raise ValueError("user_id 和 dept_id 不能为空")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.image_memory.list_images(user_id=user_id, dept_id=dept_id, limit=limit, visibility=visibility),
        )

    async def list_user_facts(
        self,
        user_id: str,
        dept_id: str,
        limit: int = 50,
        visibility: str = "",
        include_types: Optional[List[str]] = None,
        exclude_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        if not user_id or not dept_id:
            raise ValueError("user_id 和 dept_id 不能为空")
        return await self.long_term.list_memories(
            user_id=user_id,
            dept_id=dept_id,
            limit=limit,
            visibility=visibility,
            include_types=include_types,
            exclude_types=exclude_types,
        )

    async def store_interaction(self, user_id: str, user_query: str, ai_response: str, is_important: bool = False) -> None:
        """
        存储交互内容 (一轮对话: 用户提问 + AI回答)。
        将对话更新到短期窗口；
        若对话包含重要信息，可同步/异步存入长期记忆。
        """
        if not user_id or not user_query or not ai_response:
            # 允许 ai_response 为空的情况吗？通常不。
            if not user_query and not ai_response:
                 return
            
        # 1. 更新短期窗口 - 用户提问
        if user_query:
            await self.short_term.add_memory(user_id, user_query, metadata={"role": "user"})
        
        # 2. 更新短期窗口 - AI 回答
        if ai_response:
            await self.short_term.add_memory(user_id, ai_response, metadata={"role": "assistant"})
        
        # 3. 异步分析/按标记存入长期记忆
        # 这里简化处理：如果标记重要，则将 "User: ... \n AI: ..." 存入长期记忆
        if is_important:
            logger.info("检测到重要信息，正在将其注入长期记忆库...")
            combined_content = f"User: {user_query}\nAI: {ai_response}"
            await self.long_term.add_memory(user_id, combined_content, metadata={"source": "interaction", "type": "conversation"})

    async def add_user_fact(
        self,
        user_id: str,
        fact: str,
        dept_id: str = "default_dept",
        visibility: str = "private",
        memory_type: str = "fact",
        source: str = "user_fact",
        metadata_extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        添加关于用户的关键事实/长期记忆。
        visibility: 'private' (default) or 'department'
        """
        m_type = (memory_type or "fact").strip().lower() or "fact"
        m_source = (source or "user_fact").strip().lower() or "user_fact"
        metadata = {"visibility": visibility, "type": m_type, "source": m_source}
        if isinstance(metadata_extra, dict):
            metadata.update(metadata_extra)
        return await self.long_term.add_memory(user_id=user_id, content=fact, dept_id=dept_id, metadata=metadata)

    async def add_image_memory(self, user_id: str, image_bytes: bytes, description: str = "", dept_id: str = "default_dept", visibility: str = "private") -> str:
        """
        存储图片记忆（多模态）。
        """
        metadata = {"visibility": visibility}
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.image_memory.add_image_memory(
            user_id=user_id, dept_id=dept_id, image_bytes=image_bytes, description=description, metadata=metadata, visibility=visibility
        ))

    async def store_image_asset(self, user_id: str, description: str, image_uri: str, dept_id: str = "default_dept", visibility: str = "private") -> str:
        """
        [Deprecated] 存储多模态图像资产记忆 (仅元数据).
        建议使用 add_image_memory 直接存入图片文件和特征。
        """
        metadata = {"image_uri": image_uri, "visibility": visibility}
        # ImageMemory.add_memory is not fully implemented for assets only, keeping this for compatibility if needed
        # But actually add_memory returns a warning string. 
        # We should probably use add_image_memory even here if we could read the URI, but for now let's just leave it or warn.
        logger.warning("store_image_asset is deprecated and may not work as expected.")
        return await self.image_memory.add_memory(user_id, description, metadata)

    async def forget_memory(self, user_id: str, memory_id: str, memory_type: str = "long_term") -> bool:
        """
        主动遗忘记忆（根据类型删除）。
        """
        if memory_type == "long_term":
            return await self.long_term.delete_memory(user_id, memory_id)
        elif memory_type == "short_term":
            return await self.short_term.delete_memory(user_id, memory_id)
        elif memory_type == "image":
            return await self.image_memory.delete_memory(user_id, memory_id)
        else:
            logger.warning(f"未知的记忆类型: {memory_type}")
            return False
