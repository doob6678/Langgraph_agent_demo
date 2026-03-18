from typing import Any, Dict, List, Optional
import time
import logging
import os
import uuid
import re
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

from .base_memory import BaseMemory
from backend.agent.config_ex.memory_config import get_runtime_memory_settings

# 尝试导入 pymilvus
try:
    from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    logging.warning("pymilvus 尚未安装或导入失败，图像记忆将仅支持 Mock 模式。")

# 尝试导入 CLIP 服务
try:
    from backend.services.clip_service_local import clip_service
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logging.warning("CLIP 服务导入失败，图像特征提取功能将不可用。")

logger = logging.getLogger(__name__)

class ImageMemory(BaseMemory):
    """
    图像记忆模块（多模态）
    职责：
    - 存储用户上传或对话中生成的图片特征（CLIP提取的向量）。
    - 将图像元数据及资产解耦存放（URI/Asset ID -> 本地文件系统/OSS）。
    - 实现基于部门/用户的权限隔离。
    """

    def __init__(self):
        """
        初始化图像记忆模块。
        """
        self.milvus_collection = None
        self._collections: Dict[str, Any] = {}
        self._milvus_alias = "image_memory"
        self._vector_dim: Optional[int] = None
        runtime_cfg = get_runtime_memory_settings()
        self._collection_name = str(runtime_cfg.get("image_collection_name") or "agent_image_memory").strip() or "agent_image_memory"

        assets_dir = (os.getenv("ASSETS_DIR") or "").strip()
        project_root = Path(__file__).resolve().parents[3]
        default_assets_dir = project_root / "data" / "images"
        if assets_dir:
            self.storage_root = Path(assets_dir)
        else:
            self.storage_root = default_assets_dir
        self.storage_root.mkdir(parents=True, exist_ok=True)
        
        if MILVUS_AVAILABLE:
            self._init_milvus_connection()

    def _init_milvus_connection(self):
        load_dotenv()
        milvus_host = os.getenv("MILVUS_HOST", "127.0.0.1")
        milvus_port = os.getenv("MILVUS_PORT", "19530")
        milvus_user = os.getenv("MILVUS_USER", "root")
        milvus_password = os.getenv("MILVUS_PASSWORD", "")
        milvus_db = os.getenv("MILVUS_DB", "default")

        try:
            if not connections.has_connection(self._milvus_alias):
                logger.info(f"正在连接 Milvus (Image Memory) ({milvus_host}:{milvus_port})")
                connections.connect(
                    alias=self._milvus_alias,
                    host=milvus_host,
                    port=milvus_port,
                    user=milvus_user,
                    password=milvus_password,
                    db_name=milvus_db
                )
            legacy_collections = [
                "agent_image_memory_default_tenant",
            ]
            for legacy_name in legacy_collections:
                if utility.has_collection(legacy_name, using=self._milvus_alias):
                    utility.drop_collection(legacy_name, using=self._milvus_alias)
                    logger.info(f"已清理历史错误 Image Collection: {legacy_name}")
        except Exception as e:
            logger.error(f"Milvus 连接初始化失败: {e}")
            self._collections = {}
            self.milvus_collection = None

    def _resolve_collection_name(self) -> str:
        return self._collection_name

    def _milvus_escape(self, value: Any) -> str:
        return str(value or "").replace("\\", "\\\\").replace("'", "\\'")

    def _build_acl_expr(self, user_id: str, dept_id: str) -> str:
        if not user_id or not dept_id:
            return ""
        dept = self._milvus_escape(dept_id)
        user = self._milvus_escape(user_id)
        return (
            f"dept_id == '{dept}' and "
            f"(visibility == 'department' or ((visibility == 'private' or visibility == '') and user_id == '{user}'))"
        )

    def ensure_collection_ready(self, embedding_dim: Optional[int] = None) -> bool:
        if embedding_dim is None:
            try:
                model_info = clip_service.get_model_info() if CLIP_AVAILABLE else {}
                embedding_dim = int(model_info.get("embed_dim") or 0) or None
            except Exception:
                embedding_dim = None
        return self._ensure_collection(embedding_dim=embedding_dim) is not None

    def _fit_vector_dim(self, vector: List[float], target_dim: int) -> List[float]:
        if target_dim <= 0:
            return vector
        current_dim = len(vector)
        if current_dim == target_dim:
            return vector
        if current_dim > target_dim:
            return vector[:target_dim]
        return vector + [0.0] * (target_dim - current_dim)

    def _format_datetime(self, ts: Optional[int] = None) -> str:
        value = int(ts if ts is not None else time.time())
        return datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")

    def _normalize_datetime_value(self, value: Any) -> str:
        if isinstance(value, str):
            text = value.strip()
            if text.isdigit():
                try:
                    return datetime.fromtimestamp(int(text)).strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    return text
            return text
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(int(value)).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                return ""
        return ""

    def _ensure_collection(self, embedding_dim: Optional[int] = None) -> Optional[Any]:
        if not MILVUS_AVAILABLE:
            return None
        if embedding_dim is not None:
            try:
                embedding_dim = int(embedding_dim)
            except Exception:
                embedding_dim = None
        collection_name = self._resolve_collection_name()
        cached = self._collections.get(collection_name)
        if cached is not None:
            self.milvus_collection = cached
            return cached
        try:
            if utility.has_collection(collection_name, using=self._milvus_alias):
                collection = Collection(collection_name, using=self._milvus_alias)
                field_names = [f.name for f in collection.schema.fields]
                required_fields = {
                    "id",
                    "dept_id",
                    "user_id",
                    "visibility",
                    "image_uri",
                    "embedding",
                    "content",
                    "metadata",
                    "created_at",
                    "updated_at",
                }
                forbidden_fields = {"tenant_id"}
                has_forbidden = bool(forbidden_fields.intersection(set(field_names)))
                if has_forbidden or not required_fields.issubset(set(field_names)):
                    utility.drop_collection(collection_name, using=self._milvus_alias)
                    collection = None
                else:
                    field_map = {f.name: f for f in collection.schema.fields}
                    created_dtype = getattr(field_map.get("created_at"), "dtype", None)
                    updated_dtype = getattr(field_map.get("updated_at"), "dtype", None)
                    if created_dtype != DataType.VARCHAR or updated_dtype != DataType.VARCHAR:
                        utility.drop_collection(collection_name, using=self._milvus_alias)
                        collection = None
                    else:
                        embedding_field = field_map.get("embedding")
                        existing_dim = None
                        if embedding_field is not None:
                            existing_dim = getattr(embedding_field, "params", {}).get("dim")
                        if existing_dim:
                            self._vector_dim = int(existing_dim)
                        collection.load()
            else:
                collection = None

            if collection is None:
                create_dim = embedding_dim
                if create_dim is None:
                    try:
                        model_info = clip_service.get_model_info() if CLIP_AVAILABLE else {}
                        create_dim = int(model_info.get("embed_dim") or 0)
                    except Exception:
                        create_dim = 0
                if not create_dim or int(create_dim) <= 0:
                    raise ValueError("创建图像集合失败：无法确定 embedding 维度")
                self._vector_dim = int(create_dim)
                fields = [
                    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True, description="Primary Key"),
                    FieldSchema(name="dept_id", dtype=DataType.VARCHAR, max_length=64, description="Department ID"),
                    FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64, description="User ID"),
                    FieldSchema(name="visibility", dtype=DataType.VARCHAR, max_length=20, description="Visibility (private/department)"),
                    FieldSchema(name="image_uri", dtype=DataType.VARCHAR, max_length=1024, description="Image URI/URL"),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=int(create_dim), description="Image CLIP Embedding"),
                    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096, description="Image Description/OCR"),
                    FieldSchema(name="metadata", dtype=DataType.JSON, description="Metadata"),
                    FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=32, description="Created At Datetime"),
                    FieldSchema(name="updated_at", dtype=DataType.VARCHAR, max_length=32, description="Updated At Datetime"),
                ]
                schema = CollectionSchema(fields, description="Agent Image Memory Collection")
                collection = Collection(collection_name, schema, using=self._milvus_alias)
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128}
                }
                collection.create_index(field_name="embedding", index_params=index_params)
                collection.load()

            self._collections[collection_name] = collection
            self.milvus_collection = collection
            return collection
        except Exception as e:
            logger.error(f"初始化图片集合失败({collection_name}): {e}")
            return None

    def add_image_memory(self, user_id: str, dept_id: str, image_bytes: bytes, description: str = "", metadata: Optional[Dict[str, Any]] = None, visibility: str = "private") -> str:
        """
        添加图片记忆（保存文件 + 提取特征 + 落库）
        """
        if not user_id or not dept_id:
            raise ValueError("user_id 和 dept_id 不能为空")
        
        if not image_bytes:
            raise ValueError("image_bytes 不能为空")

        try:
            meta: Dict[str, Any] = metadata if isinstance(metadata, dict) else {}
            timestamp = int(time.time())
            timestamp_dt = self._format_datetime(timestamp)
            unique_id = str(uuid.uuid4())
            raw_name = str(meta.get("filename") or "").strip()

            source_name = Path(raw_name).name if raw_name else "uploaded_image.jpg"
            source_stem = Path(source_name).stem.strip()
            source_ext = Path(source_name).suffix.strip().lower()
            safe_stem = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", source_stem).strip("._-")
            if not safe_stem:
                safe_stem = "uploaded_image"
            if not source_ext or len(source_ext) > 10:
                source_ext = ".jpg"

            filename = f"{safe_stem}_{timestamp}{source_ext}"
            
            relative_path = Path(dept_id) / user_id / filename
            full_path = self.storage_root / relative_path
            full_path.parent.mkdir(parents=True, exist_ok=True)

            if full_path.exists():
                filename = f"{safe_stem}_{timestamp}_{unique_id[:8]}{source_ext}"
                relative_path = Path(dept_id) / user_id / filename
                full_path = self.storage_root / relative_path
            
            with open(full_path, "wb") as f:
                f.write(image_bytes)
            
            logger.info(f"图片已保存至: {full_path}")

            # 3. 提取 CLIP 特征
            if not CLIP_AVAILABLE:
                raise RuntimeError("CLIP 服务不可用，无法提取特征")
            
            embedding = clip_service.encode_image(image_bytes)
            # 确保 embedding 是 list[float]
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()
            if isinstance(embedding[0], list): # 如果是 batch 结果，取第一个
                embedding = embedding[0]
            embedding = [float(x) for x in embedding]
            embedding_dim = len(embedding)

            collection = self._ensure_collection(embedding_dim=embedding_dim)
            if collection is None:
                 raise RuntimeError("Milvus 连接未初始化，无法保存记忆")
            if self._vector_dim and int(self._vector_dim) != int(embedding_dim):
                embedding = self._fit_vector_dim(embedding, int(self._vector_dim))

            # 4. 插入 Milvus
            data = [
                [unique_id],          # id
                [dept_id],            # dept_id
                [user_id],            # user_id
                [visibility],         # visibility
                [str(relative_path).replace("\\", "/")], # image_uri (统一用正斜杠)
                [embedding],          # embedding
                [description],        # content
                [{
                    **meta,
                    "filename": filename,
                    "original_filename": source_name,
                    "stored_filename": filename,
                    "created_at": timestamp_dt,
                    "updated_at": timestamp_dt,
                }],
                [timestamp_dt],       # created_at
                [timestamp_dt],       # updated_at
            ]
            
            collection.insert(data)
            collection.flush()
            logger.info(f"图片记忆已存入 Milvus, ID: {unique_id}")
            
            return unique_id
            
        except Exception as e:
            logger.error(f"添加图片记忆失败: {e}")
            raise e

    def search_images(self, query_vector: Any, top_k: int = 5, user_id: str = "", dept_id: str = "") -> List[Dict[str, Any]]:
        """
        搜索图片记忆（带权限控制）
        """
        expr = self._build_acl_expr(user_id=user_id, dept_id=dept_id)
        if not dept_id or not user_id:
            logger.warning("search_images: 缺少 user_id 或 dept_id，无法构建权限过滤器，拒绝查询。")
            return []

        try:
            # 确保 query_vector 格式正确
            if hasattr(query_vector, "tolist"):
                query_vector = query_vector.tolist()
            if isinstance(query_vector[0], list):
                 query_vector = query_vector[0] # Milvus search expects list of list for vectors, but let's be careful
            query_vector = [float(x) for x in query_vector]
            vector_dim = len(query_vector)
            collection = self._ensure_collection(embedding_dim=vector_dim)
            if collection is None:
                return []
            if self._vector_dim and int(self._vector_dim) != int(vector_dim):
                query_vector = self._fit_vector_dim(query_vector, int(self._vector_dim))
            
            # Milvus search param: data needs to be [[...]] for single vector search
            search_data = [query_vector]

            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            results = collection.search(
                data=search_data,
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["id", "dept_id", "user_id", "visibility", "image_uri", "content", "metadata", "created_at", "updated_at"]
            )
            
            formatted_results = []
            for hits in results:
                for hit in hits:
                    formatted_results.append({
                        "id": hit.entity.get("id"),
                        "dept_id": hit.entity.get("dept_id"),
                        "user_id": hit.entity.get("user_id"),
                        "visibility": hit.entity.get("visibility"),
                        "image_uri": hit.entity.get("image_uri"),
                        "content": hit.entity.get("content"),
                        "score": hit.score,
                        "metadata": hit.entity.get("metadata"),
                        "created_at": self._normalize_datetime_value(hit.entity.get("created_at")),
                        "updated_at": self._normalize_datetime_value(hit.entity.get("updated_at")),
                    })
            
            return formatted_results

        except Exception as e:
            logger.error(f"搜索图片记忆失败: {e}")
            return []

    def list_images(self, user_id: str, dept_id: str, limit: int = 50, visibility: str = "") -> List[Dict[str, Any]]:
        if not user_id or not dept_id:
            return []
        try:
            collection = self._ensure_collection()
            if collection is None:
                return []
            safe_limit = max(1, min(int(limit), 200))
            expr = self._build_acl_expr(user_id=user_id, dept_id=dept_id)
            v = (visibility or "").strip()
            if v:
                expr = f"{expr} and visibility == '{self._milvus_escape(v)}'"
            rows = collection.query(
                expr=expr,
                output_fields=["id", "dept_id", "user_id", "visibility", "image_uri", "content", "metadata", "created_at", "updated_at"],
                limit=safe_limit,
            )
            result: List[Dict[str, Any]] = []
            for row in rows or []:
                if not isinstance(row, dict):
                    continue
                result.append(
                    {
                        "id": row.get("id"),
                        "dept_id": row.get("dept_id"),
                        "user_id": row.get("user_id"),
                        "visibility": row.get("visibility"),
                        "image_uri": row.get("image_uri"),
                        "content": row.get("content"),
                        "metadata": row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
                        "created_at": self._normalize_datetime_value(row.get("created_at")),
                        "updated_at": self._normalize_datetime_value(row.get("updated_at")),
                    }
                )
            return result
        except Exception as e:
            logger.error(f"列出图片记忆失败: {e}")
            return []

    def search_images_by_text(self, query: str, top_k: int = 5, user_id: str = "", dept_id: str = "") -> List[Dict[str, Any]]:
        """
        基于文本搜索图片记忆（带权限控制）
        """
        if not query:
            return []
        
        if not CLIP_AVAILABLE:
             logger.warning("CLIP 服务不可用，无法进行文本搜图。")
             return []

        try:
            # 文本 -> 向量
            text_embedding = clip_service.encode_text(query)
            
            # 使用向量搜索
            return self.search_images(text_embedding, top_k, user_id, dept_id)
            
        except Exception as e:
            logger.error(f"文本搜图失败: {e}")
            return []

    async def get_memory(self, user_id: str, query: str = "", dept_id: str = "", limit: int = 5) -> List[Dict[str, Any]]:
        """
        BaseMemory 接口实现: 文本搜图
        """
        if not query:
            return []
        # Note: search_images_by_text is sync, so we just call it.
        return self.search_images_by_text(query, top_k=limit, user_id=user_id, dept_id=dept_id)

    async def add_memory(self, user_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
         """BaseMemory 接口实现"""
         return "Use add_image_memory instead."

    async def delete_memory(self, user_id: str, memory_id: str) -> bool:
        """
        删除指定记忆 (BaseMemory 接口)
        """
        if not self._collections and self.milvus_collection is None:
            return False
        
        try:
            candidates = list(self._collections.values())
            if self.milvus_collection is not None and self.milvus_collection not in candidates:
                candidates.append(self.milvus_collection)
            for collection in candidates:
                res = collection.query(
                    expr=(
                        f"id == '{self._milvus_escape(memory_id)}' and "
                        f"user_id == '{self._milvus_escape(user_id)}'"
                    ),
                    output_fields=["image_uri"]
                )
                if not res:
                    continue
                collection.delete(
                    expr=(
                        f"id == '{self._milvus_escape(memory_id)}' and "
                        f"user_id == '{self._milvus_escape(user_id)}'"
                    )
                )
                image_uri = (res[0].get("image_uri") or "").strip()
                if image_uri:
                    full_path = self.storage_root / image_uri
                    if full_path.exists():
                        try:
                            os.remove(full_path)
                            logger.info(f"Deleted image file: {full_path}")
                        except OSError as e:
                            logger.error(f"Failed to delete file {full_path}: {e}")
                return True
            logger.warning(f"Delete failed: Memory {memory_id} not found for user {user_id}")
            return False
        except Exception as e:
            logger.error(f"delete_memory failed: {e}")
            return False
