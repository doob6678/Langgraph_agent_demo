import os
import torch
import numpy as np
from typing import List, Union, Optional
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import io

class CLIPService:
    """真正的CLIP模型服务 - 使用Transformers实现"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None):
        """初始化CLIP模型"""
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.embed_dim = 512  # ViT-B/32的维度
        
        self._load_model()
    
    def _load_model(self):
        """加载CLIP模型"""
        try:
            print(f"正在加载CLIP模型: {self.model_name} on {self.device}")
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"CLIP模型加载成功")
        except Exception as e:
            print(f"加载CLIP模型失败: {e}")
            raise
    
    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """编码文本"""
        if isinstance(text, str):
            text = [text]
        
        try:
            # 使用CLIP处理器处理文本
            inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features.float()
                # L2归一化
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 转换为numpy数组
            return text_features.cpu().numpy()
            
        except Exception as e:
            print(f"文本编码失败: {e}")
            raise
    
    def encode_image(self, image_data: Union[bytes, Image.Image, np.ndarray]) -> np.ndarray:
        """编码图片"""
        try:
            if isinstance(image_data, bytes):
                # 从字节数据创建图片
                image = Image.open(io.BytesIO(image_data))
            elif isinstance(image_data, np.ndarray):
                # 从numpy数组创建图片
                image = Image.fromarray(image_data)
            else:
                image = image_data
            
            # 预处理图片
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features.float()
                # L2归一化
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # 转换为numpy数组
            return image_features.cpu().numpy()
            
        except Exception as e:
            print(f"图片编码失败: {e}")
            raise
    
    def compute_similarity(self, text_features: np.ndarray, image_features: np.ndarray) -> np.ndarray:
        """计算文本和图片的相似度"""
        try:
            # 计算余弦相似度
            similarity = np.dot(text_features, image_features.T)
            return similarity
            
        except Exception as e:
            print(f"相似度计算失败: {e}")
            raise
    
    def batch_encode_images(self, images: List[Union[bytes, Image.Image, np.ndarray]]) -> np.ndarray:
        """批量编码图片"""
        try:
            processed_images = []
            
            for image_data in images:
                if isinstance(image_data, bytes):
                    image = Image.open(io.BytesIO(image_data))
                elif isinstance(image_data, np.ndarray):
                    image = Image.fromarray(image_data)
                else:
                    image = image_data
                
                processed_images.append(image)
            
            # 批量处理
            inputs = self.processor(images=processed_images, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                batch_features = self.model.get_image_features(**inputs)
                batch_features = batch_features.float()
                # L2归一化
                batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
            
            return batch_features.cpu().numpy()
            
        except Exception as e:
            print(f"批量图片编码失败: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "embed_dim": self.embed_dim,
            "loaded": self.model is not None
        }

# 创建全局实例
clip_service = CLIPService(
    model_name=os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32"),
    device=os.getenv("CLIP_DEVICE", None)
)