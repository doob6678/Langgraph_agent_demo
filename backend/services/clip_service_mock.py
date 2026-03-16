import os
import numpy as np
from typing import List, Optional, Union
from PIL import Image
import io

class CLIPService:
    """模拟CLIP模型服务（用于演示和测试）"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None):
        """初始化CLIP服务（模拟版本）
        
        Args:
            model_name: CLIP模型名称（模拟）
            device: 运行设备（模拟）
        """
        self.model_name = model_name
        self.device = device or "cpu"
        self.embed_dim = 512  # CLIP ViT-B/32的维度
        
        print(f"[模拟] 正在加载CLIP模型 {model_name} 到设备 {self.device}...")
        print(f"[模拟] CLIP模型加载完成")
        
    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """编码文本（模拟版本）
        
        Args:
            text: 要编码的文本或文本列表
            
        Returns:
            文本特征向量，形状为 (n_texts, embed_dim)
        """
        if isinstance(text, str):
            text = [text]
            
        n_texts = len(text)
        
        # 模拟文本编码：基于文本长度和内容的确定性随机向量
        np.random.seed(42)  # 固定种子以确保一致性
        
        text_features = []
        for i, t in enumerate(text):
            # 为每个文本生成基于内容的特征向量
            seed = hash(t) % (2**32)
            np.random.seed(seed)
            
            # 生成基础特征
            features = np.random.randn(self.embed_dim).astype(np.float32)
            
            # 根据文本长度进行微调
            length_factor = min(len(t) / 100.0, 1.0)
            features *= (0.8 + 0.4 * length_factor)
            
            # 归一化
            features = features / np.linalg.norm(features)
            
            text_features.append(features)
        
        return np.array(text_features)
    
    def encode_image(self, image: Union[Image.Image, np.ndarray, bytes]) -> np.ndarray:
        """编码图片（模拟版本）
        
        Args:
            image: PIL图片、numpy数组或图片字节数据
            
        Returns:
            图片特征向量，形状为 (1, embed_dim)
        """
        # 处理不同类型的图片输入
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # 获取图片信息用于生成模拟特征
        width, height = image.size
        
        # 模拟图片编码：基于图片尺寸的确定性随机向量
        seed = (width * 1000 + height) % (2**32)
        np.random.seed(seed)
        
        # 生成基础特征
        features = np.random.randn(self.embed_dim).astype(np.float32)
        
        # 根据图片尺寸进行微调
        size_factor = min((width * height) / (1000 * 1000), 1.0)
        features *= (0.8 + 0.4 * size_factor)
        
        # 归一化
        features = features / np.linalg.norm(features)
        
        return features.reshape(1, -1)
    
    def compute_similarity(self, text_features: np.ndarray, image_features: np.ndarray) -> np.ndarray:
        """计算文本和图片的相似度（模拟版本）
        
        Args:
            text_features: 文本特征向量，形状为 (n_texts, feature_dim)
            image_features: 图片特征向量，形状为 (n_images, feature_dim)
            
        Returns:
            相似度矩阵，形状为 (n_texts, n_images)
        """
        # 计算余弦相似度
        similarity = np.dot(text_features, image_features.T)
        
        # 添加一些噪声使其更真实
        np.random.seed(123)
        noise = np.random.normal(0, 0.05, similarity.shape).astype(np.float32)
        similarity += noise
        
        # 限制在合理范围内
        similarity = np.clip(similarity, -1.0, 1.0)
        
        return similarity
    
    def batch_encode_text(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """批量编码文本（模拟版本）"""
        all_features = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_features = self.encode_text(batch_texts)
            all_features.append(batch_features)
            
        return np.vstack(all_features)
    
    def batch_encode_images(self, images: List[Union[Image.Image, np.ndarray, bytes]], 
                           batch_size: int = 32) -> np.ndarray:
        """批量编码图片（模拟版本）"""
        all_features = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_features = []
            
            for image in batch_images:
                features = self.encode_image(image)
                batch_features.append(features)
                
            batch_features = np.vstack(batch_features)
            all_features.append(batch_features)
            
        return np.vstack(all_features)
    
    def get_model_info(self) -> dict:
        """获取模型信息（模拟版本）"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "input_resolution": 224,
            "embed_dim": self.embed_dim,
            "vision_layers": 12,
            "vision_width": 768,
            "context_length": 77,
            "vocab_size": 49408,
            "note": "这是模拟版本，用于演示和测试"
        }

# 创建全局CLIP服务实例
clip_service = CLIPService(
    model_name=os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32"),
    device=os.getenv("CLIP_DEVICE", None)
)