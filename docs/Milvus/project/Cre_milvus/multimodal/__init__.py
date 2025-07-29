"""
多模态处理模块

支持文本和图像的统一向量化处理
"""

try:
    from .clip_encoder import CLIPEncoder
    from .text_processor import TextProcessor
    from .image_processor import ImageProcessor
except ImportError:
    # 处理相对导入问题
    from clip_encoder import CLIPEncoder
    from text_processor import TextProcessor
    from image_processor import ImageProcessor

__all__ = ['CLIPEncoder', 'TextProcessor', 'ImageProcessor']