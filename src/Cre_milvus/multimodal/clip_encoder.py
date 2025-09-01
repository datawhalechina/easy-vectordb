"""
CLIP编码器模块

基于CLIP模型实现文本和图像的统一向量编码
"""

import torch
from PIL import Image
try:
    import clip
except ImportError:
    print("Warning: CLIP not installed. Please install with: pip install git+https://github.com/openai/CLIP.git")
    clip = None
import numpy as np
from typing import List, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class CLIPEncoder:
    """
    CLIP编码器，支持文本和图像的向量化
    """
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        """
        初始化CLIP编码器
        
        参数:
            model_name: CLIP模型名称
            device: 计算设备
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        if clip is None:
            raise ImportError("CLIP not available. Please install with: pip install git+https://github.com/openai/CLIP.git")
        
        try:
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            self.model.eval()
            logger.info(f"CLIP模型 {model_name} 加载成功，使用设备: {self.device}")
        except Exception as e:
            logger.error(f"CLIP模型加载失败: {e}")
            raise
    
    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        编码文本为向量
        
        参数:
            texts: 单个文本或文本列表
        
        返回:
            文本向量数组
        """
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            # 对文本进行tokenization
            text_tokens = clip.tokenize(texts).to(self.device)
            
            with torch.no_grad():
                # 获取文本特征
                text_features = self.model.encode_text(text_tokens)
                # L2归一化
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            return text_features.cpu().numpy()
        
        except Exception as e:
            logger.error(f"文本编码失败: {e}")
            raise
    
    def encode_image(self, images: Union[str, Image.Image, List[Union[str, Image.Image]]]) -> np.ndarray:
        """
        编码图像为向量
        
        参数:
            images: 图像路径、PIL图像对象或其列表
        
        返回:
            图像向量数组
        """
        if not isinstance(images, list):
            images = [images]
        
        try:
            processed_images = []
            
            for img in images:
                if isinstance(img, str):
                    # 从路径加载图像
                    pil_image = Image.open(img).convert('RGB')
                elif isinstance(img, Image.Image):
                    pil_image = img.convert('RGB')
                else:
                    raise ValueError(f"不支持的图像类型: {type(img)}")
                
                # 预处理图像
                processed_img = self.preprocess(pil_image)
                processed_images.append(processed_img)
            
            # 批量处理
            image_batch = torch.stack(processed_images).to(self.device)
            
            with torch.no_grad():
                # 获取图像特征
                image_features = self.model.encode_image(image_batch)
                # L2归一化
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy()
        
        except Exception as e:
            logger.error(f"图像编码失败: {e}")
            raise
    
    def compute_similarity(self, text_features: np.ndarray, image_features: np.ndarray) -> np.ndarray:
        """
        计算文本和图像特征之间的相似度
        
        参数:
            text_features: 文本特征向量
            image_features: 图像特征向量
        
        返回:
            相似度矩阵
        """
        # 计算余弦相似度
        similarity = np.dot(text_features, image_features.T)
        return similarity
    
    def text_to_image_search(self, query_text: str, image_features: np.ndarray, 
                           top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        文搜图功能
        
        参数:
            query_text: 查询文本
            image_features: 图像特征库
            top_k: 返回前k个结果
        
        返回:
            (相似度分数, 图像索引)
        """
        # 编码查询文本
        text_features = self.encode_text(query_text)
        
        # 计算相似度
        similarities = self.compute_similarity(text_features, image_features)[0]
        
        # 获取top-k结果
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]
        
        return top_scores, top_indices
    
    def image_to_text_search(self, query_image: Union[str, Image.Image], 
                           text_features: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        图搜文功能
        
        参数:
            query_image: 查询图像
            text_features: 文本特征库
            top_k: 返回前k个结果
        
        返回:
            (相似度分数, 文本索引)
        """
        # 编码查询图像
        image_features = self.encode_image(query_image)
        
        # 计算相似度
        similarities = self.compute_similarity(text_features, image_features)[0]
        
        # 获取top-k结果
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]
        
        return top_scores, top_indices
    
    def get_feature_dimension(self) -> int:
        """
        获取特征向量维度
        
        返回:
            特征维度
        """
        if "ViT-B/32" in self.model_name:
            return 512
        elif "ViT-B/16" in self.model_name:
            return 512
        elif "ViT-L/14" in self.model_name:
            return 768
        else:
            # 通过实际编码获取维度
            dummy_text = self.encode_text("test")
            return dummy_text.shape[1]