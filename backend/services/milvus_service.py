import os
import time
from typing import List, Dict, Any, Optional
import numpy as np
from pymilvus import Collection, connections, utility

class MilvusService:
    """Milvus向量数据库服务"""
    
    def __init__(self):
        self.host = os.getenv("MILVUS_HOST", "127.0.0.1")
        self.port = int(os.getenv("MILVUS_PORT", "19530"))
        self.user = os.getenv("MILVUS_USER", "root")
        self.password = os.getenv("MILVUS_PASSWORD", "")
        self.db_name = os.getenv("MILVUS_DB", "web_rag_demo_01")
        self.collection_name = os.getenv("MILVUS_COLLECTION", "image_embeddings")
        
        # 连接Milvus
        self.connect()
    
    def connect(self):
        """连接到Milvus数据库"""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                db_name=self.db_name
            )
            print(f"[milvus] connected: {self.host}:{self.port}/{self.db_name}")
        except Exception as e:
            print(f"[milvus] connect failed: {e}")
            raise
    
    def test_connection(self):
        """测试连接"""
        try:
            # 测试连接
            utility.list_collections()
            return True
        except Exception as e:
            print(f"Milvus连接测试失败: {e}")
            return False
    
    def search_images(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """基于向量搜索图片"""
        try:
            collection = Collection(self.collection_name)
            try:
                collection.load()
            except Exception:
                pass

            fields = {f.name: f for f in collection.schema.fields}
            if "vec" in fields:
                vec_field = "vec"
            elif "embedding" in fields:
                vec_field = "embedding"
            else:
                raise RuntimeError(f"collection {self.collection_name} 缺少向量字段 vec/embedding")

            try:
                dim = int((fields[vec_field].params or {}).get("dim") or 0)
            except Exception:
                dim = 0
            
            # 确保向量是一维的float32数组
            if query_vector.ndim > 1:
                query_vector = query_vector.flatten()
            query_vector = query_vector.astype(np.float32)
            if dim > 0 and int(query_vector.shape[0]) != dim:
                raise ValueError(f"查询向量维度不匹配: got={int(query_vector.shape[0])} expected={dim}")

            try:
                k = int(top_k)
            except Exception:
                k = 5
            if k < 1:
                k = 1
            if k > 50:
                k = 50
            
            # 搜索参数
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            # 执行搜索
            results = collection.search(
                data=[query_vector.tolist()],
                anns_field=vec_field,
                param=search_params,
                limit=k,
                output_fields=[x for x in ["filename"] if x in fields]
            )
            
            # 格式化结果
            formatted_results = []
            for hits in results:
                for hit in hits:
                    result = {
                        "filename": hit.entity.get("filename", "") if hasattr(hit, "entity") else "",
                        "score": float(hit.score),
                        "id": hit.id
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"搜索图片失败: {e}")
            return []
    
    def insert_image_embedding(self, filename: str, embedding: np.ndarray, metadata: Dict[str, Any] = None):
        """插入图片向量"""
        try:
            collection = Collection(self.collection_name)
            fields = {f.name: f for f in collection.schema.fields}
            if "vec" in fields and "id" in fields and "filename" in fields:
                raise RuntimeError("当前集合使用(id, filename, vec) schema，请使用离线建库脚本写入")
            if "embedding" not in fields:
                raise RuntimeError("当前集合不支持 insert_image_embedding")
            
            # 确保向量是float32类型
            if embedding.ndim > 1:
                embedding = embedding.flatten()
            embedding = embedding.astype(np.float32)
            
            # 准备数据
            data = [{
                "filename": filename,
                "embedding": embedding.tolist(),
                "metadata": metadata or {}
            }]
            
            # 插入数据
            collection.insert(data)
            print(f"[milvus] inserted embedding: {filename}")
            
        except Exception as e:
            print(f"插入图片向量失败: {e}")
            raise
    
    def search_images_by_text(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """基于文本搜索相关图片"""
        try:
            q = (query or "").strip()
            if not q:
                return []

            from backend.services.clip_service_local import clip_service
            text_embedding = clip_service.encode_text(q)
            
            # 使用向量搜索图片
            results = self.search_images(text_embedding, top_k)
            
            print(f"[milvus] text search done: '{query}' -> {len(results)} results")
            return results
            
        except Exception as e:
            print(f"[milvus] text search failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            collection = Collection(self.collection_name)
            stats = {
                "name": collection.name,
                "num_entities": collection.num_entities,
                "schema": str(collection.schema)
            }
            return stats
        except Exception as e:
            print(f"获取集合统计信息失败: {e}")
            return {}

# 创建全局实例
milvus_service = MilvusService()
