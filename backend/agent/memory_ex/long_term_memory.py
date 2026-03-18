from typing import Any, Dict, List, Optional
import time
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

from .base_memory import BaseMemory
from .database import SessionLocal, engine
from .models import MemoryContent, Base, migrate_memory_contents_schema
from .embedding import LocalEmbedding

# 尝试导入 pymilvus
try:
    from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    logging.warning("pymilvus 尚未安装或导入失败，长期记忆将仅支持 MySQL 存储功能或 Mock 模式。")

logger = logging.getLogger(__name__)

# 确保在首次运行时创建数据库表结构
Base.metadata.create_all(bind=engine)
migrate_memory_contents_schema(engine)

class LongTermMemory(BaseMemory):
    """
    长期记忆模块
    职责：
    - 存储用户画像、历史事实、重要知识点。
    - 基于语义向量（Embedding）进行相关性检索（使用Milvus）。
    - 处理超大文本（>8000字符），对接MySQL扩展表 `memory_contents`。
    
    设计考量：
    - 混合存储策略（Hybrid Storage）：长文本存MySQL，摘要/向量存Milvus。
    - 数据隔离：按部门/用户/可见性进行权限隔离。
    """
    
    def __init__(self, embedding_model: Optional[LocalEmbedding] = None):
        """
        初始化长期记忆系统。
        """
        self.embedding_model = embedding_model
        self.milvus_collection = None
        self._milvus_alias = "long_term_memory"
        self._collection_name = "agent_long_term_memory"
        
        # 初始化 Milvus 连接 (按需)
        if MILVUS_AVAILABLE:
            self._init_milvus_connection()
            # 初始化 Embedding 模型
            if self.embedding_model is None:
                try:
                    self.embedding_model = LocalEmbedding()
                except Exception as e:
                    logger.error(f"Embedding 模型加载失败，向量检索功能将不可用: {e}")

    def _init_milvus_connection(self):
        """
        初始化 Milvus 连接并确保 Collection 存在
        """
        load_dotenv()
        milvus_host = os.getenv("MILVUS_HOST", "127.0.0.1")
        milvus_port = os.getenv("MILVUS_PORT", "19530")
        milvus_user = os.getenv("MILVUS_USER", "root")
        milvus_password = os.getenv("MILVUS_PASSWORD", "")
        milvus_db = os.getenv("MILVUS_DB", "default")
        
        self._collection_name = (os.getenv("LONG_TERM_MEMORY_COLLECTION") or "agent_long_term_memory").strip() or "agent_long_term_memory"
        try:
            if not connections.has_connection(self._milvus_alias):
                logger.info(f"正在连接 Milvus ({milvus_host}:{milvus_port}), db: {milvus_db}")
                connections.connect(
                    alias=self._milvus_alias,
                    host=milvus_host,
                    port=milvus_port,
                    user=milvus_user,
                    password=milvus_password,
                    db_name=milvus_db
                )
            
            # 初始化 Collection
            collection_name = self._collection_name
            if utility.has_collection(collection_name, using=self._milvus_alias):
                # 检查 Schema 是否包含 visibility
                existing_coll = Collection(collection_name, using=self._milvus_alias)
                field_names = [f.name for f in existing_coll.schema.fields]
                if (
                    "visibility" not in field_names
                    or "dept_id" not in field_names
                    or "created_at" not in field_names
                    or "updated_at" not in field_names
                    or "tenant_id" in field_names
                ):
                    logger.warning(f"Milvus LongTerm Collection '{collection_name}' Schema 过期 (缺少 visibility 或 dept_id)，正在重建...")
                    utility.drop_collection(collection_name, using=self._milvus_alias)
                else:
                    self.milvus_collection = existing_coll
                    self.milvus_collection.load()
                    logger.info(f"Milvus Collection '{collection_name}' 加载成功")
            
            legacy_collection_name = "agent_long_term_memory_default_tenant"
            if utility.has_collection(legacy_collection_name, using=self._milvus_alias):
                utility.drop_collection(legacy_collection_name, using=self._milvus_alias)
                logger.info(f"已清理历史错误 Collection: {legacy_collection_name}")

            if not utility.has_collection(collection_name, using=self._milvus_alias):
                logger.info(f"Milvus Collection '{collection_name}' 不存在，正在创建...")
                # 定义 Schema
                fields = [
                    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True, description="Primary Key"),
                    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096, description="Text content (short or summary)"),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768, description="Text embedding"), # damo/nlp_corom_sentence-embedding_chinese-base dim is 768
                    FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64, description="User ID"),
                    FieldSchema(name="dept_id", dtype=DataType.VARCHAR, max_length=64, description="Department ID"), # 新增
                    FieldSchema(name="visibility", dtype=DataType.VARCHAR, max_length=20, description="Visibility (private/department)"), # 新增
                    FieldSchema(name="has_ext", dtype=DataType.BOOL, description="Has extended content in MySQL"),
                    FieldSchema(name="metadata", dtype=DataType.JSON, description="Metadata"),
                    FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=32, description="Created At Datetime"),
                    FieldSchema(name="updated_at", dtype=DataType.VARCHAR, max_length=32, description="Updated At Datetime"),
                ]
                schema = CollectionSchema(fields, description="Agent Long Term Memory Collection")
                self.milvus_collection = Collection(collection_name, schema, using=self._milvus_alias)
                
                # 创建索引
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128}
                }
                self.milvus_collection.create_index(field_name="embedding", index_params=index_params)
                self.milvus_collection.load()
                logger.info(f"Milvus Collection '{collection_name}' 创建并加载成功")
                
        except Exception as e:
            logger.error(f"Milvus 连接初始化失败: {str(e)}")
            self.milvus_collection = None

    def _format_datetime(self, value: Optional[Any] = None) -> str:
        if isinstance(value, str):
            text = value.strip()
            if text and not text.isdigit():
                return text
            if text.isdigit():
                try:
                    return datetime.fromtimestamp(int(text)).strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    return ""
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(int(value)).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                return ""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _datetime_to_ts(self, value: Any) -> int:
        text = str(value or "").strip()
        if not text:
            return 0
        if text.isdigit():
            return int(text)
        try:
            return int(datetime.strptime(text, "%Y-%m-%d %H:%M:%S").timestamp())
        except Exception:
            return 0

    async def add_memory(self, user_id: str, content: str, dept_id: str = "default_dept", metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        添加长期记忆。
        实现混合存储策略：当文本超长时存MySQL，短文本/摘要存入Milvus。
        """
        if not user_id or not content:
            raise ValueError("参数 user_id 和 content 不能为空")
            
        metadata = metadata or {}
        created_at = self._format_datetime(metadata.get("created_at"))
        updated_at = self._format_datetime(metadata.get("updated_at") or created_at)
        metadata["created_at"] = created_at
        metadata["updated_at"] = updated_at
        visibility = metadata.get("visibility", "private") # private, department, public
        
        # 生成唯一记忆ID
        mem_id = f"ltm_{int(time.time() * 1000)}"
        has_ext = False
        summary_content = content
        
        try:
            # 模拟超长文本存储逻辑 (这里我们把阈值调小一点以便测试，比如 > 50 个字符就存 MySQL)
            if len(content) > 50:
                logger.info(f"触发长文本存储 (长度 {len(content)}), 写入 MySQL.")
                has_ext = True
                
                # 1. 存入 MySQL 长文本表 (memory_contents)
                db = SessionLocal()
                try:
                    new_mem = MemoryContent(
                        id=mem_id,
                        user_id=user_id,
                        content=content
                    )
                    db.add(new_mem)
                    db.commit()
                except Exception as db_e:
                    db.rollback()
                    logger.error(f"写入 MySQL 失败: {str(db_e)}")
                    raise
                finally:
                    db.close()
                
                # 2. 生成摘要 (此处简化为截取前50个字)
                summary_content = f"[摘要] {content[:50]}..."
            
            # 3. 将摘要存入 Milvus
            if MILVUS_AVAILABLE and self.milvus_collection and self.embedding_model:
                logger.info(f"向 Milvus 写入记忆: {summary_content[:20]}...")
                
                # 生成向量
                embedding_vector = self.embedding_model.embed_query(summary_content)
                if not embedding_vector:
                    logger.warning("向量生成失败，跳过 Milvus 存储")
                    return mem_id
                
                # 构造插入数据
                data = [
                    [mem_id],               # id
                    [summary_content],      # content
                    [embedding_vector],     # embedding
                    [user_id],              # user_id
                    [dept_id],              # dept_id
                    [visibility],           # visibility
                    [has_ext],              # has_ext
                    [metadata],             # metadata
                    [created_at],           # created_at
                    [updated_at],           # updated_at
                ]
                
                self.milvus_collection.insert(data)
                self.milvus_collection.flush() # 强制刷新确保可见
                logger.info(f"Milvus 写入成功: id={mem_id}")
                
        except Exception as e:
            logger.error(f"存储长期记忆失败: {str(e)}")
            raise RuntimeError("存储长期记忆出现系统级异常") from e
            
        return mem_id

    def _normalize_memory_type(self, metadata: Any) -> str:
        if not isinstance(metadata, dict):
            return "fact"
        value = str(metadata.get("type") or "fact").strip().lower()
        return value or "fact"

    def _match_memory_type(self, metadata: Any, include_types: Optional[List[str]] = None, exclude_types: Optional[List[str]] = None) -> bool:
        current_type = self._normalize_memory_type(metadata)
        include_set = {str(v).strip().lower() for v in (include_types or []) if str(v).strip()}
        exclude_set = {str(v).strip().lower() for v in (exclude_types or []) if str(v).strip()}
        if include_set and current_type not in include_set:
            return False
        if exclude_set and current_type in exclude_set:
            return False
        return True

    async def get_memory(
        self,
        user_id: str,
        query: str,
        dept_id: str = "default_dept",
        limit: int = 5,
        include_types: Optional[List[str]] = None,
        exclude_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        根据查询语句（通常会被转为Embedding向量）检索相关长期记忆。
        支持租户和用户级别的数据隔离查询。
        """
        if not user_id or not query:
            raise ValueError("参数 user_id 和 query 不能为空")
            
        logger.info(f"正在检索与 '{query}' 相关的长期记忆, user_id: {user_id}, dept_id: {dept_id}")
        
        results = []
        
        if MILVUS_AVAILABLE and self.milvus_collection and self.embedding_model:
            try:
                # 生成查询向量
                query_vector = self.embedding_model.embed_query(query)
                if not query_vector:
                    logger.warning("查询向量生成失败")
                    return []
                
                # 构建过滤表达式 (租户隔离已经在 collection 层面做了，这里做用户隔离)
                # expr = f"user_id == '{user_id}'"
                # 更新为支持可见性控制 (部门公开可见，或私有且归属本人)
                expr = f"(visibility == 'department' and dept_id == '{dept_id}') or (visibility == 'private' and user_id == '{user_id}')"
                
                # 执行搜索
                search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
                search_results = self.milvus_collection.search(
                    data=[query_vector],
                    anns_field="embedding",
                    param=search_params,
                    limit=limit,
                    expr=expr,
                    output_fields=["content", "has_ext", "metadata", "id", "created_at", "updated_at"]
                )
                
                # 处理结果
                for hits in search_results:
                    for hit in hits:
                        item = {
                            "id": hit.entity.get("id"),
                            "content": hit.entity.get("content"),
                            "metadata": hit.entity.get("metadata"),
                            "score": hit.score,
                            "has_ext": hit.entity.get("has_ext"),
                            "created_at": self._format_datetime(hit.entity.get("created_at")),
                            "updated_at": self._format_datetime(hit.entity.get("updated_at")),
                        }
                        
                        # 如果有扩展内容，去 MySQL 查完整内容
                        if item["has_ext"]:
                            logger.info(f"检测到扩展内容，从 MySQL 加载完整文本: {item['id']}")
                            db = SessionLocal()
                            try:
                                record = db.query(MemoryContent).filter(MemoryContent.id == item['id']).first()
                                if record:
                                    item["content"] = record.content # 替换为完整内容
                                    if not isinstance(item["metadata"], dict):
                                        item["metadata"] = {}
                                    item["metadata"]["source"] = "mysql_full_text"
                            except Exception as db_e:
                                logger.error(f"MySQL 查询失败: {db_e}")
                            finally:
                                db.close()
                                
                        if self._match_memory_type(item.get("metadata"), include_types=include_types, exclude_types=exclude_types):
                            results.append(item)
                        
            except Exception as e:
                logger.error(f"Milvus 检索失败: {e}")
                
        return results

    async def list_memories(
        self,
        user_id: str,
        dept_id: str = "default_dept",
        limit: int = 50,
        visibility: str = "",
        include_types: Optional[List[str]] = None,
        exclude_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        if not user_id:
            raise ValueError("参数 user_id 不能为空")
        if limit < 1:
            limit = 1
        if limit > 200:
            limit = 200

        if not (MILVUS_AVAILABLE and self.milvus_collection):
            return []

        cond = (
            f"(visibility == 'department' and dept_id == '{dept_id}') "
            f"or (visibility == 'private' and user_id == '{user_id}')"
        )
        vis = (visibility or "").strip().lower()
        if vis == "private":
            cond = f"user_id == '{user_id}' and visibility == 'private'"
        elif vis == "department":
            cond = f"dept_id == '{dept_id}' and visibility == 'department'"

        try:
            rows = self.milvus_collection.query(
                expr=cond,
                output_fields=["id", "content", "metadata", "user_id", "dept_id", "visibility", "created_at", "updated_at"],
                limit=limit,
            )
        except Exception as e:
            logger.error(f"Milvus 列表查询失败: {e}")
            return []

        normalized: List[Dict[str, Any]] = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            if self._match_memory_type(meta, include_types=include_types, exclude_types=exclude_types):
                normalized.append(
                    {
                        "id": row.get("id"),
                        "content": row.get("content"),
                        "metadata": meta,
                        "user_id": row.get("user_id"),
                        "dept_id": row.get("dept_id"),
                        "visibility": row.get("visibility"),
                        "created_at": self._format_datetime(row.get("created_at") or meta.get("created_at")),
                        "updated_at": self._format_datetime(row.get("updated_at") or meta.get("updated_at")),
                    }
                )
        normalized.sort(key=lambda x: self._datetime_to_ts(x.get("created_at")), reverse=True)
        return normalized

    async def delete_memory(self, user_id: str, memory_id: str) -> bool:
        """
        主动遗忘机制：删除长期记忆，需同时清理Milvus和MySQL。
        """
        if not user_id or not memory_id:
            return False
            
        logger.info(f"从 MySQL 和 Milvus 删除记忆记录: {memory_id}")
        
        db = SessionLocal()
        try:
            # 1. 清理 MySQL
            record = db.query(MemoryContent).filter(MemoryContent.id == memory_id, MemoryContent.user_id == user_id).first()
            if record:
                db.delete(record)
                db.commit()
                logger.info(f"已从 MySQL 删除记录: {memory_id}")
                
            # 2. 清理 Milvus
            if MILVUS_AVAILABLE and self.milvus_collection:
                expr = f"id == '{memory_id}'"
                self.milvus_collection.delete(expr)
                self.milvus_collection.flush()
                logger.info(f"已从 Milvus 删除记录: {memory_id}")
                
        except Exception as e:
            db.rollback()
            logger.error(f"删除记忆失败: {str(e)}")
            return False
        finally:
            db.close()
            
        return True
