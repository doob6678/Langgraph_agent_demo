import os
import torch
import clip
import numpy as np
from typing import List, Optional, Union
from PIL import Image
import io

class CLIPService:
    """CLIP模型服务"""
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        """初始化CLIP服务
        
        Args:
            model_name: CLIP模型名称，默认为ViT-B/32
            device: 运行设备，如果为None则自动选择
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"正在加载CLIP模型 {model_name} 到设备 {self.device}...")
        
        # 加载模型
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        
        print(f"CLIP模型加载完成")
        
    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """编码文本
        
        Args:
            text: 要编码的文本或文本列表
            
        Returns:
            文本特征向量，形状为 (n_texts, 512) 对于ViT-B/32模型
        """
        if isinstance(text, str):
            text = [text]
            
        # 文本预处理
        text_tokens = clip.tokenize(text).to(self.device)
        
        # 编码文本
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            
        # 归一化特征向量
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 转换为numpy数组
        return text_features.cpu().numpy()
    
    def encode_image(self, image: Union[Image.Image, np.ndarray, bytes]) -> np.ndarray:
        """编码图片
        
        Args:
            image: PIL图片、numpy数组或图片字节数据
            
        Returns:
            图片特征向量，形状为 (1, 512) 对于ViT-B/32模型
        """
        # 处理不同类型的图片输入
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # 预处理图片
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # 编码图片
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            
        # 归一化特征向量
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # 转换为numpy数组
        return image_features.cpu().numpy()
    
    def compute_similarity(self, text_features: np.ndarray, image_features: np.ndarray) -> np.ndarray:
        """计算文本和图片的相似度
        
        Args:
            text_features: 文本特征向量，形状为 (n_texts, feature_dim)
            image_features: 图片特征向量，形状为 (n_images, feature_dim)
            
        Returns:
            相似度矩阵，形状为 (n_texts, n_images)
        """
        # 计算余弦相似度
        similarity = np.dot(text_features, image_features.T)
        return similarity
    
    def batch_encode_text(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """批量编码文本
        
        Args:
            texts: 文本列表
            batch_size: 批次大小
            
        Returns:
            文本特征向量数组
        """
        all_features = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_features = self.encode_text(batch_texts)
            all_features.append(batch_features)
            
        return np.vstack(all_features)
    
    def batch_encode_images(self, images: List[Union[Image.Image, np.ndarray, bytes]], 
                           batch_size: int = 32) -> np.ndarray:
        """批量编码图片
        
        Args:
            images: 图片列表
            batch_size: 批次大小
            
        Returns:
            图片特征向量数组
        """
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
        """获取模型信息"""
        return {
            "model_name": "ViT-B/32",
            "device": self.device,
            "input_resolution": 224,
            "embed_dim": 512,
            "vision_layers": 12,
            "vision_width": 768,
            "context_length": 77,
            "vocab_size": 49408
        }

# 创建全局CLIP服务实例
clip_service = CLIPService(
    model_name=os.getenv("CLIP_MODEL_NAME", "ViT-B/32"),
    device=os.getenv("CLIP_DEVICE", None)
)