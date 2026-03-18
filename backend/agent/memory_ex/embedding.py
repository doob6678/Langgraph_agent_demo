from typing import List
import logging
import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

logger = logging.getLogger(__name__)

class LocalEmbedding:
    """
    本地 Embedding 模型封装 (基于 ModelScope)
    使用轻量级中文模型: damo/nlp_corom_sentence-embedding_chinese-base
    """
    def __init__(self, model_id: str = None):
        # 优先从环境变量加载配置
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        self.model_id = model_id or os.getenv("EMBEDDING_MODEL_ID", "damo/nlp_corom_sentence-embedding_chinese-base")
        # 兼容 windows 路径
        cache_dir = os.getenv("MODELSCOPE_CACHE_DIR", r"D:\ModelScope\models")
        
        try:
            # 优先检查本地路径，避免每次联网检查更新
            # 将 model_id (如 damo/xxx) 转换为本地路径格式 (damo\xxx)
            local_path = os.path.join(cache_dir, self.model_id.replace("/", os.sep))
            if os.path.exists(local_path):
                logger.info(f"检测到本地模型缓存: {local_path}")
                self.model_to_load = local_path
            else:
                logger.info(f"本地未找到模型，将从 ModelScope 加载: {self.model_id}")
                self.model_to_load = self.model_id
            
            # 不在 init 中加载，延迟到显式调用或首次使用
            self.pipeline = None
            
        except Exception as e:
            logger.error(f"Embedding 配置初始化失败: {str(e)}")
            raise

    def load_model(self):
        """
        显式加载模型（用于启动时预加载）
        """
        if hasattr(self, 'pipeline') and self.pipeline:
            logger.info("Embedding 模型已加载，跳过")
            return

        logger.info(f"正在加载 Embedding 模型: {self.model_to_load} ...")
        self.pipeline = pipeline(Tasks.sentence_embedding, model=self.model_to_load)
        logger.info("Embedding 模型加载成功")

    def embed_query(self, text: str) -> List[float]:
        """
        生成单条文本的向量
        """
        if not hasattr(self, 'pipeline') or not self.pipeline:
            self.load_model()
            
        if not text:
            return []
        
        embeddings = self.embed_documents([text])
        if embeddings:
            return embeddings[0]
        return []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        生成多条文本的向量
        """
        if not texts:
            return []
            
        try:
            # ModelScope pipeline for this model requires input as {'source_sentence': [text1, text2, ...]}
            result = self.pipeline(input={'source_sentence': texts})
            
            # logger.info(f"Embedding raw result type: {type(result)}")
            
            embeddings = []
            if isinstance(result, dict) and 'text_embedding' in result:
                raw_embeddings = result['text_embedding']
                # raw_embeddings is usually a numpy array or list of lists
                if isinstance(raw_embeddings, np.ndarray):
                    embeddings = raw_embeddings.tolist()
                elif isinstance(raw_embeddings, list):
                    embeddings = raw_embeddings
                else:
                    logger.error(f"Unknown embedding format: {type(raw_embeddings)}")
                    
            elif isinstance(result, list):
                # Fallback for other potential return types
                 for item in result:
                     if isinstance(item, dict) and 'text_embedding' in item:
                         emb = item['text_embedding']
                         if isinstance(emb, np.ndarray):
                             embeddings.append(emb.tolist())
                         else:
                             embeddings.append(emb)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"生成向量失败: {str(e)}")
            return []
