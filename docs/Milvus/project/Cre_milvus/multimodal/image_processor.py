"""
图像处理模块

处理各种格式的图像文件，支持图像向量化
"""

import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    图像处理器，支持多种图像格式
    """
    
    def __init__(self, clip_encoder=None):
        """
        初始化图像处理器
        
        参数:
            clip_encoder: CLIP编码器实例
        """
        self.clip_encoder = clip_encoder
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        处理单个图像文件
        
        参数:
            file_path: 图像文件路径
        
        返回:
            处理后的图像信息
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        if file_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"不支持的图像格式: {file_path.suffix}")
        
        try:
            # 加载图像
            image = Image.open(file_path).convert('RGB')
            
            # 获取图像基本信息
            width, height = image.size
            
            # 编码图像（如果有CLIP编码器）
            image_vector = None
            if self.clip_encoder:
                try:
                    image_vector = self.clip_encoder.encode_image(image)[0]  # 取第一个（单张图像）
                except Exception as e:
                    logger.error(f"图像编码失败: {e}")
            
            result = {
                'id': file_path.stem,
                'content': str(file_path),  # 图像路径作为内容
                'source_file': str(file_path),
                'content_type': 'image',
                'image_info': {
                    'width': width,
                    'height': height,
                    'format': image.format,
                    'mode': image.mode,
                    'size_bytes': file_path.stat().st_size
                }
            }
            
            if image_vector is not None:
                result['vector'] = image_vector.tolist()
            
            return result
            
        except Exception as e:
            logger.error(f"处理图像文件失败 {file_path}: {e}")
            raise
    
    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        处理目录中的所有图像文件
        
        参数:
            directory_path: 目录路径
        
        返回:
            所有处理后的图像信息列表
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"目录不存在: {directory_path}")
        
        all_results = []
        
        # 遍历目录中的所有支持的图像文件
        for file_path in directory_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                try:
                    result = self.process_file(str(file_path))
                    all_results.append(result)
                    logger.info(f"成功处理图像: {file_path}")
                except Exception as e:
                    logger.error(f"处理图像失败 {file_path}: {e}")
        
        return all_results
    
    def process_csv_with_images(self, csv_path: str, image_column: str = 'path') -> List[Dict[str, Any]]:
        """
        处理包含图像路径的CSV文件
        
        参数:
            csv_path: CSV文件路径
            image_column: 图像路径列名
        
        返回:
            处理后的图像信息列表
        """
        try:
            df = pd.read_csv(csv_path)
            
            if image_column not in df.columns:
                raise ValueError(f"CSV文件中不存在列: {image_column}")
            
            all_results = []
            csv_dir = Path(csv_path).parent
            
            for idx, row in df.iterrows():
                image_path = row[image_column]
                
                # 处理相对路径
                if not os.path.isabs(image_path):
                    image_path = csv_dir / image_path
                else:
                    image_path = Path(image_path)
                
                try:
                    result = self.process_file(str(image_path))
                    
                    # 添加CSV中的其他信息
                    result['csv_info'] = row.to_dict()
                    result['csv_index'] = idx
                    
                    all_results.append(result)
                    
                except Exception as e:
                    logger.error(f"处理CSV中的图像失败 {image_path}: {e}")
            
            return all_results
            
        except Exception as e:
            logger.error(f"处理CSV文件失败: {e}")
            raise
    
    def create_image_thumbnail(self, image_path: str, size: Tuple[int, int] = (128, 128)) -> str:
        """
        创建图像缩略图
        
        参数:
            image_path: 原始图像路径
            size: 缩略图尺寸
        
        返回:
            缩略图路径
        """
        try:
            image_path = Path(image_path)
            thumbnail_dir = image_path.parent / 'thumbnails'
            thumbnail_dir.mkdir(exist_ok=True)
            
            thumbnail_path = thumbnail_dir / f"{image_path.stem}_thumb{image_path.suffix}"
            
            if not thumbnail_path.exists():
                image = Image.open(image_path)
                image.thumbnail(size, Image.Resampling.LANCZOS)
                image.save(thumbnail_path)
            
            return str(thumbnail_path)
            
        except Exception as e:
            logger.error(f"创建缩略图失败: {e}")
            return str(image_path)  # 返回原图路径
    
    def get_image_stats(self, directory_path: str) -> Dict[str, Any]:
        """
        获取目录中图像的统计信息
        
        参数:
            directory_path: 目录路径
        
        返回:
            统计信息字典
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"目录不存在: {directory_path}")
        
        stats = {
            'total_images': 0,
            'formats': {},
            'total_size_bytes': 0,
            'avg_width': 0,
            'avg_height': 0
        }
        
        widths = []
        heights = []
        
        for file_path in directory_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                try:
                    image = Image.open(file_path)
                    width, height = image.size
                    
                    stats['total_images'] += 1
                    stats['total_size_bytes'] += file_path.stat().st_size
                    
                    format_key = file_path.suffix.lower()
                    stats['formats'][format_key] = stats['formats'].get(format_key, 0) + 1
                    
                    widths.append(width)
                    heights.append(height)
                    
                except Exception as e:
                    logger.error(f"获取图像信息失败 {file_path}: {e}")
        
        if widths:
            stats['avg_width'] = sum(widths) / len(widths)
            stats['avg_height'] = sum(heights) / len(heights)
        
        return stats
    
    def batch_encode_images(self, image_paths: List[str]) -> np.ndarray:
        """
        批量编码图像
        
        参数:
            image_paths: 图像路径列表
        
        返回:
            图像向量数组
        """
        if not self.clip_encoder:
            raise ValueError("需要CLIP编码器进行图像编码")
        
        try:
            return self.clip_encoder.encode_image(image_paths)
        except Exception as e:
            logger.error(f"批量图像编码失败: {e}")
            raise