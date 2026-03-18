from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class BaseMemory(ABC):
    """
    记忆模块抽象基类
    定义了记忆模块的基础接口，以支持未来不同存储后端的无缝切换（例如从Milvus切换到PGVector）。
    遵循开闭原则（OCP）和依赖倒置原则（DIP）。
    """

    @abstractmethod
    async def add_memory(self, user_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        添加一条记忆。
        
        Args:
            user_id: 用户唯一标识，用于数据隔离。
            content: 记忆内容。
            metadata: 附加元数据（如时间戳、来源等）。
            
        Returns:
            str: 存储的记忆唯一ID。
        """
        pass

    @abstractmethod
    async def get_memory(self, user_id: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        根据查询条件获取相关记忆。
        
        Args:
            user_id: 用户唯一标识，用于数据隔离。
            query: 查询内容。
            limit: 返回的最大条数。
            
        Returns:
            List[Dict[str, Any]]: 检索到的记忆列表。
        """
        pass

    @abstractmethod
    async def delete_memory(self, user_id: str, memory_id: str) -> bool:
        """
        删除指定记忆。
        
        Args:
            user_id: 用户唯一标识。
            memory_id: 要删除的记忆ID。
            
        Returns:
            bool: 删除是否成功。
        """
        pass
