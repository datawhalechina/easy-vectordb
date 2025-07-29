"""
图像数据处理工具

处理各种格式的图像文件，支持图像向量化和多模态搜索
"""

import os
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
import logging

from System.monitor import log_event

# 尝试导入CLIP编码器
try:
    from multimodal.clip_encoder import CLIPEncoder
    from multimodal.image_processor import ImageProcessor
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False
    print("Warning: 多模态模块不可用，图像将只保存基本信息")

logger = logging.getLogger(__name__)


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
        img_path = Path(img_path)
        
        if not img_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {img_path}")
        
        # 支持的图像格式
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
        if img_path.suffix.lower() not in supported_formats:
            raise ValueError(f"不支持的图像格式: {img_path.suffix}")
        
        log_event(f"开始处理图像: {img_path}")
        
        # 基本图像信息处理
        try:
            image = Image.open(img_path).convert('RGB')
            width, height = image.size
            file_size = img_path.stat().st_size
            
            result = {
                'id': img_path.stem,
                'content': str(img_path),  # 图像路径作为内容
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
            
            # 如果支持多模态且启用CLIP
            if MULTIMODAL_AVAILABLE and use_clip:
                try:
                    # 初始化CLIP编码器
                    clip_encoder = CLIPEncoder()
                    
                    # 编码图像
                    image_vector = clip_encoder.encode_image(image)[0]
                    result['embedding'] = image_vector.tolist()
                    result['vector_dim'] = len(image_vector)
                    
                    log_event(f"图像向量化成功: {img_path}, 维度: {len(image_vector)}")
                    
                except Exception as e:
                    logger.warning(f"图像向量化失败 {img_path}: {e}")
                    result['embedding_error'] = str(e)
            
            log_event(f"图像处理完成: {img_path}")
            return [result]
            
        except Exception as e:
            logger.error(f"处理图像文件失败 {img_path}: {e}")
            raise
            
    except Exception as e:
        error_msg = f"处理图像失败: {img_path}\n错误: {str(e)}\n{traceback.format_exc()}"
        log_event(error_msg)
        print(error_msg)
        return []


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
        
        # 批量处理图像
        if MULTIMODAL_AVAILABLE and use_clip and image_files:
            try:
                # 初始化处理器
                clip_encoder = CLIPEncoder()
                image_processor = ImageProcessor(clip_encoder)
                
                # 批量处理
                for img_file in image_files:
                    try:
                        result = image_processor.process_file(str(img_file))
                        all_results.append(result)
                    except Exception as e:
                        logger.error(f"批量处理图像失败 {img_file}: {e}")
                        # 降级到基本处理
                        basic_result = process_img(str(img_file), use_clip=False)
                        all_results.extend(basic_result)
                        
            except Exception as e:
                logger.error(f"批量图像处理失败: {e}")
                # 降级到逐个处理
                for img_file in image_files:
                    result = process_img(str(img_file), use_clip=False)
                    all_results.extend(result)
        else:
            # 逐个处理图像
            for img_file in image_files:
                result = process_img(str(img_file), use_clip)
                all_results.extend(result)
        
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
        
        return stats
        
    except Exception as e:
        logger.error(f"获取图像统计信息失败: {e}")
        return {}