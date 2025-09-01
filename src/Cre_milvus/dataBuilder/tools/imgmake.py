"""
图像数据处理工具

处理各种格式的图像文件，支持图像向量化和多模态搜索
重构后的版本，充分利用multimodal包的功能，避免代码重复
"""

import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from System.monitor import log_event

# 导入multimodal包的功能
try:
    from multimodal.clip_encoder import CLIPEncoder
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False
    print("Warning: 多模态模块不可用，图像将只保存基本信息")

logger = logging.getLogger(__name__)

# 全局变量，用于缓存编码器实例，避免重复初始化
_clip_encoder_cache = None


def _get_clip_encoder() -> Optional[CLIPEncoder]:
    """
    获取CLIP编码器实例，使用缓存避免重复初始化
    
    返回:
        CLIPEncoder实例或None
    """
    global _clip_encoder_cache
    
    if not MULTIMODAL_AVAILABLE:
        return None
    
    try:
        if _clip_encoder_cache is None:
            _clip_encoder_cache = CLIPEncoder()
            log_event("CLIP编码器初始化成功")
        
        return _clip_encoder_cache
    
    except Exception as e:
        logger.error(f"初始化CLIP编码器失败: {e}")
        return None


def process_img(img_path: str, use_clip: bool = True) -> List[Dict[str, Any]]:
    """
    处理单个图像文件
    
    参数:
        img_path: 图像文件路径
        use_clip: 是否使用CLIP进行向量化
    
    返回:
        包含图像信息的列表
    """
    try:
        log_event(f"开始处理图像: {img_path}")
        
        # 基本图像处理
        result = _process_img_basic(img_path)
        
        # 如果启用CLIP且可用，添加向量化
        if use_clip and result:
            clip_encoder = _get_clip_encoder()
            if clip_encoder:
                try:
                    from PIL import Image
                    image = Image.open(img_path).convert('RGB')
                    image_vector = clip_encoder.encode_image(image)[0]
                    result[0]['embedding'] = image_vector.tolist()
                    result[0]['vector_dim'] = len(image_vector)
                    log_event(f"图像向量化成功: {img_path}, 维度: {len(image_vector)}")
                except Exception as e:
                    logger.warning(f"图像向量化失败 {img_path}: {e}")
                    result[0]['embedding_error'] = str(e)
        
        log_event(f"图像处理完成: {img_path}")
        return result
            
    except Exception as e:
        error_msg = f"处理图像失败: {img_path}\n错误: {str(e)}\n{traceback.format_exc()}"
        log_event(error_msg)
        print(error_msg)
        return []


def _process_img_basic(img_path: str) -> List[Dict[str, Any]]:
    """
    基本图像处理，获取图像基本信息
    
    参数:
        img_path: 图像文件路径
    
    返回:
        包含基本图像信息的列表
    """
    from PIL import Image
    
    try:
        img_path = Path(img_path)
        
        if not img_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {img_path}")
        
        # 支持的图像格式
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
        if img_path.suffix.lower() not in supported_formats:
            raise ValueError(f"不支持的图像格式: {img_path.suffix}")
        
        # 基本图像信息处理
        image = Image.open(img_path).convert('RGB')
        width, height = image.size
        file_size = img_path.stat().st_size
        
        result = {
            'id': img_path.stem,
            'content': str(img_path),
            'source_file': str(img_path),
            'content_type': 'image',
            'image_info': {
                'width': width,
                'height': height,
                'format': image.format or img_path.suffix[1:].upper(),
                'mode': image.mode,
                'size_bytes': file_size,
                'aspect_ratio': round(width / height, 2) if height > 0 else 0
            }
        }
        
        return [result]
        
    except Exception as e:
        logger.error(f"基本图像处理失败 {img_path}: {e}")
        raise


def process_image_directory(directory_path: str, use_clip: bool = True) -> List[Dict[str, Any]]:
    """
    处理目录中的所有图像文件
    
    参数:
        directory_path: 目录路径
        use_clip: 是否使用CLIP进行向量化
    
    返回:
        所有图像信息的列表
    """
    try:
        log_event(f"开始处理图像目录: {directory_path}")
        
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"目录不存在: {directory_path}")
        
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
        all_results = []
        
        # 遍历目录中的所有图像文件
        image_files = []
        for file_path in directory_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_formats:
                image_files.append(file_path)
        
        log_event(f"在目录 {directory_path} 中找到 {len(image_files)} 个图像文件")
        
        # 逐个处理图像
        for img_file in image_files:
            try:
                result = process_img(str(img_file), use_clip)
                all_results.extend(result)
            except Exception as e:
                logger.error(f"处理图像文件失败 {img_file}: {e}")
        
        log_event(f"目录图像处理完成: {directory_path}, 共处理 {len(all_results)} 个图像")
        return all_results
            
    except Exception as e:
        error_msg = f"处理图像目录失败: {directory_path}\n错误: {str(e)}\n{traceback.format_exc()}"
        log_event(error_msg)
        print(error_msg)
        return []


def create_image_thumbnails(image_paths: List[str], thumbnail_size: tuple = (128, 128)) -> Dict[str, str]:
    """
    为图像创建缩略图
    
    参数:
        image_paths: 图像路径列表
        thumbnail_size: 缩略图尺寸
    
    返回:
        原图路径到缩略图路径的映射
    """
    from PIL import Image
    thumbnail_map = {}
    
    for img_path in image_paths:
        try:
            img_path = Path(img_path)
            thumbnail_dir = img_path.parent / 'thumbnails'
            thumbnail_dir.mkdir(exist_ok=True)
            
            thumbnail_path = thumbnail_dir / f"{img_path.stem}_thumb{img_path.suffix}"
            
            if not thumbnail_path.exists():
                image = Image.open(img_path)
                image.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
                image.save(thumbnail_path)
            
            thumbnail_map[str(img_path)] = str(thumbnail_path)
            
        except Exception as e:
            logger.error(f"创建缩略图失败 {img_path}: {e}")
            thumbnail_map[str(img_path)] = str(img_path)  # 使用原图
    
    return thumbnail_map


def get_image_statistics(directory_path: str) -> Dict[str, Any]:
    """
    获取目录中图像的统计信息
    
    参数:
        directory_path: 目录路径
    
    返回:
        统计信息字典
    """
    from PIL import Image
    
    try:
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"目录不存在: {directory_path}")
        
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
        
        stats = {
            'total_images': 0,
            'formats': {},
            'total_size_bytes': 0,
            'total_size_mb': 0,
            'avg_width': 0,
            'avg_height': 0,
            'min_size': {'width': float('inf'), 'height': float('inf')},
            'max_size': {'width': 0, 'height': 0}
        }
        
        widths = []
        heights = []
        
        for file_path in directory_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_formats:
                try:
                    image = Image.open(file_path)
                    width, height = image.size
                    file_size = file_path.stat().st_size
                    
                    stats['total_images'] += 1
                    stats['total_size_bytes'] += file_size
                    
                    format_key = file_path.suffix.lower()
                    stats['formats'][format_key] = stats['formats'].get(format_key, 0) + 1
                    
                    widths.append(width)
                    heights.append(height)
                    
                    # 更新最小最大尺寸
                    if width < stats['min_size']['width']:
                        stats['min_size']['width'] = width
                    if height < stats['min_size']['height']:
                        stats['min_size']['height'] = height
                    if width > stats['max_size']['width']:
                        stats['max_size']['width'] = width
                    if height > stats['max_size']['height']:
                        stats['max_size']['height'] = height
                    
                except Exception as e:
                    logger.error(f"获取图像信息失败 {file_path}: {e}")
        
        if widths:
            stats['avg_width'] = round(sum(widths) / len(widths), 2)
            stats['avg_height'] = round(sum(heights) / len(heights), 2)
        
        stats['total_size_mb'] = round(stats['total_size_bytes'] / (1024 * 1024), 2)
        
        # 处理无图像的情况
        if stats['total_images'] == 0:
            stats['min_size'] = {'width': 0, 'height': 0}
        
        log_event(f"图像统计完成: {directory_path}, 共 {stats['total_images']} 张图像")
        return stats
        
    except Exception as e:
        logger.error(f"获取图像统计信息失败: {e}")
        return {}