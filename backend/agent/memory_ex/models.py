import logging
from sqlalchemy import Column, String, Text, DateTime, inspect, text
from sqlalchemy.sql import func
from .database import Base

logger = logging.getLogger(__name__)

class MemoryContent(Base):
    """
    记忆长文本拓展表
    用于存储超过 Milvus 限制的完整长文档或超长对话历史。
    """
    __tablename__ = "memory_contents"

    # 主键，通常与 Milvus 中存储的 memory_id 对应
    id = Column(String(64), primary_key=True, index=True, comment="记忆唯一标识")
    
    # 用户ID，用于用户级别的隐私隔离
    user_id = Column(String(64), index=True, nullable=False, comment="用户标识")
    
    # 存储超大文本内容 (MySQL 的 TEXT 或 LONGTEXT)
    content = Column(Text, nullable=False, comment="完整的长文本内容")
    
    # 创建时间与更新时间
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), comment="更新时间")

class ShortTermMemoryModel(Base):
    """
    短期记忆表 (会话历史)
    用于持久化存储最近的 N 条对话记录，防止服务重启丢失上下文。
    """
    __tablename__ = "short_term_memories"

    id = Column(String(64), primary_key=True, index=True, comment="记忆唯一标识")
    user_id = Column(String(64), index=True, nullable=False, comment="用户标识")
    role = Column(String(20), nullable=False, comment="角色 (user/assistant)")
    content = Column(Text, nullable=False, comment="对话内容")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True, comment="创建时间")


def migrate_memory_contents_schema(engine) -> None:
    try:
        with engine.begin() as conn:
            inspector = inspect(conn)
            if "memory_contents" not in inspector.get_table_names():
                return

            columns = {c.get("name") for c in inspector.get_columns("memory_contents")}
            if "tenant_id" not in columns:
                return

            idx_rows = conn.execute(
                text(
                    """
                    SELECT INDEX_NAME
                    FROM information_schema.STATISTICS
                    WHERE TABLE_SCHEMA = DATABASE()
                      AND TABLE_NAME = 'memory_contents'
                      AND COLUMN_NAME = 'tenant_id'
                    """
                )
            ).fetchall()
            for row in idx_rows:
                idx_name = str(row[0] or "").strip()
                if not idx_name or idx_name.upper() == "PRIMARY":
                    continue
                safe_name = idx_name.replace("`", "")
                conn.execute(text(f"ALTER TABLE memory_contents DROP INDEX `{safe_name}`"))

            conn.execute(text("ALTER TABLE memory_contents DROP COLUMN tenant_id"))
            logger.info("Schema migration done: removed memory_contents.tenant_id")
    except Exception as e:
        logger.warning(f"Schema migration skipped or failed for memory_contents: {e}")
