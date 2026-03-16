import io
import base64
from typing import Optional, Dict, Any
from PIL import Image
import numpy as np

class ImageService:
    """图片处理服务"""
    
    def __init__(self):
        self.max_size = (1024, 1024)  # 最大图片尺寸
        self.quality = 85  # JPEG质量
        
    def process_uploaded_image(self, image_data: bytes, filename: str) -> Dict[str, Any]:
        """处理上传的图片"""
        try:
            # 从字节数据创建图片
            image = Image.open(io.BytesIO(image_data))
            
            # 转换为RGB模式（如果不是的话）
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 调整图片大小（如果需要）
            if image.size[0] > self.max_size[0] or image.size[1] > self.max_size[1]:
                image.thumbnail(self.max_size, Image.Resampling.LANCZOS)
            
            # 转换为base64字符串
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=self.quality)
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # 获取图片信息
            info = {
                "filename": filename,
                "size": image.size,
                "mode": image.mode,
                "format": "JPEG",
                "base64": f"data:image/jpeg;base64,{img_base64}",
                "width": image.width,
                "height": image.height
            }
            
            return {
                "success": True,
                "data": info,
                "message": "图片处理成功"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"图片处理失败: {str(e)}",
                "message": "无法处理上传的图片"
            }
    
    def create_image_thumbnail(self, image_data: bytes, max_size: tuple = (300, 300)) -> bytes:
        """创建图片缩略图"""
        try:
            image = Image.open(io.BytesIO(image_data))
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=80)
            return buffered.getvalue()
        except Exception:
            return b""
    
    def get_image_info(self, image_data: bytes) -> Dict[str, Any]:
        """获取图片基本信息"""
        try:
            image = Image.open(io.BytesIO(image_data))
            return {
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "format": image.format,
                "size": len(image_data)
            }
        except Exception as e:
            return {
                "error": f"无法获取图片信息: {str(e)}"
            }
    
    def is_valid_image(self, image_data: bytes) -> bool:
        """检查是否为有效图片"""
        try:
            Image.open(io.BytesIO(image_data))
            return True
        except Exception:
            return False
    
    def convert_to_rgb_array(self, image_data: bytes) -> Optional[np.ndarray]:
        """将图片转换为RGB数组（用于CLIP模型）"""
        try:
            image = Image.open(io.BytesIO(image_data))
            
            # 转换为RGB模式
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 调整大小为CLIP模型输入尺寸（224x224）
            image = image.resize((224, 224), Image.Resampling.LANCZOS)
            
            # 转换为numpy数组
            img_array = np.array(image)
            
            return img_array
            
        except Exception as e:
            print(f"转换为RGB数组失败: {e}")
            return None

# 创建全局实例
image_service = ImageService()