"""
多模态处理模块

提供CLIP编码器用于文本和图像的统一向量化处理
"""

try:
    from .clip_encoder import CLIPEncoder
except ImportError:
    # 处理相对导入问题
    from clip_encoder import CLIPEncoder

__all__ = ['CLIPEncoder']