from typing import Any, Dict, List, Optional
import time
import logging
from sqlalchemy import desc
from .base_memory import BaseMemory
from .database import SessionLocal, engine
from .models import ShortTermMemoryModel, Base

logger = logging.getLogger(__name__)

class ShortTermMemory(BaseMemory):
    """
    短期记忆模块 (持久化版)
    职责：
    - 维护当前会话的上下文窗口（Sliding Window）。
    - 使用 MySQL 持久化存储，防止重启丢失。
    - 自动管理窗口大小，保持最新的 N 条记录。
    """
    
    def __init__(self, max_window_size: int = 10):
        """
        初始化短期记忆
        
        Args:
            max_window_size: 滑动窗口最大保留消息条数。默认10条。
        """
        if max_window_size <= 0:
            raise ValueError("max_window_size 必须大于 0")
            
        self.max_window_size = max_window_size
        
        # 确保表存在
        Base.metadata.create_all(bind=engine)

    async def add_memory(self, user_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        添加短期记忆（即当前会话上下文）。
        """
        if not user_id or not content:
            raise ValueError("user_id 和 content 不能为空")
            
        metadata = metadata or {}
        role = metadata.get("role", "user") # 默认为 user，调用方应传入 role
        # 使用纳秒级时间戳 + 随机数避免 ID 冲突
        import random
        mem_id = f"stm_{int(time.time() * 1000000)}_{random.randint(1000, 9999)}"
        
        db = SessionLocal()
        try:
            # 1. 插入新记录
            new_memory = ShortTermMemoryModel(
                id=mem_id,
                user_id=user_id,
                role=role,
                content=content
            )
            db.add(new_memory)
            db.commit()
            
            # 2. 检查窗口大小并清理旧记录
            # 获取该用户的所有记录数量
            count = db.query(ShortTermMemoryModel).filter(ShortTermMemoryModel.user_id == user_id).count()
            
            if count > self.max_window_size:
                # 删除最旧的 (count - max_window_size) 条记录
                # MySQL delete with limit syntax or subquery
                # 使用子查询找到需要保留的最早一条的时间，删除早于该时间的
                
                # 简单做法：查出所有记录按时间倒序，保留前 N 条，其余删除
                # 但这样在数据量大时效率低。不过短期记忆窗口通常很小（10-50），所以还好。
                # 更高效：查出第 N+1 条记录的 ID/时间，删除之前的。
                
                subquery = db.query(ShortTermMemoryModel.created_at)\
                    .filter(ShortTermMemoryModel.user_id == user_id)\
                    .order_by(desc(ShortTermMemoryModel.created_at))\
                    .limit(self.max_window_size)\
                    .all()
                
                if subquery:
                    oldest_to_keep = subquery[-1][0] # 第 N 条的时间
                    
                    # 删除早于 oldest_to_keep 的记录 (或者 id 不在保留列表中的)
                    # 注意：时间可能重复，最好用 ID。但这里简化处理，直接删除时间更早的。
                    # 为了精确，我们找出要删除的 ID 列表。
                    
                    # 重新查询需要删除的 ID
                    ids_to_keep_query = db.query(ShortTermMemoryModel.id)\
                        .filter(ShortTermMemoryModel.user_id == user_id)\
                        .order_by(desc(ShortTermMemoryModel.created_at))\
                        .limit(self.max_window_size)
                    
                    # Delete where id not in ids_to_keep
                    # MySQL doesn't support DELETE WHERE ID NOT IN (SELECT ... FROM same_table) directly in one query usually without alias trick
                    # So we fetch IDs to keep first.
                    keep_ids = [r[0] for r in ids_to_keep_query.all()]
                    
                    if keep_ids:
                        db.query(ShortTermMemoryModel)\
                            .filter(ShortTermMemoryModel.user_id == user_id)\
                            .filter(ShortTermMemoryModel.id.notin_(keep_ids))\
                            .delete(synchronize_session=False)
                        db.commit()

            return mem_id
        except Exception as e:
            db.rollback()
            logger.error(f"短期记忆写入失败: {e}")
            raise e
        finally:
            db.close()

    async def get_memory(self, user_id: str, query: str = "", limit: int = 5) -> List[Dict[str, Any]]:
        """
        获取当前短期会话记忆。
        """
        if not user_id:
            raise ValueError("user_id 不能为空")
            
        db = SessionLocal()
        try:
            # 获取最近的 limit 条记录，按时间倒序查询，然后反转回正序
            records = db.query(ShortTermMemoryModel)\
                .filter(ShortTermMemoryModel.user_id == user_id)\
                .order_by(desc(ShortTermMemoryModel.created_at))\
                .limit(limit)\
                .all()
            
            # 反转为正序（旧 -> 新）
            records.reverse()
            
            return [
                {
                    "id": r.id,
                    "role": r.role,
                    "content": r.content,
                    "timestamp": r.created_at.timestamp() if r.created_at else 0
                }
                for r in records
            ]
        finally:
            db.close()

    async def delete_memory(self, user_id: str, memory_id: str) -> bool:
        """
        删除某条特定的短期记忆。
        """
        db = SessionLocal()
        try:
            rows = db.query(ShortTermMemoryModel).filter(
                ShortTermMemoryModel.id == memory_id,
                ShortTermMemoryModel.user_id == user_id
            ).delete()
            db.commit()
            return rows > 0
        finally:
            db.close()
        
    async def clear_session(self, user_id: str) -> None:
        """
        清空指定用户的当前会话（用于会话重置）。
        """
        db = SessionLocal()
        try:
            db.query(ShortTermMemoryModel).filter(ShortTermMemoryModel.user_id == user_id).delete()
            db.commit()
        finally:
            db.close()
