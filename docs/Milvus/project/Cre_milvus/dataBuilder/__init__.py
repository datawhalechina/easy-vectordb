"""
数据构建器模块

集成了数据分块策略和多模态图像处理功能的统一数据处理模块
"""

from .data import data_process, get_chunking_strategies, batch_process_images

# 导入工具模块
from .tools.csvmake import process_csv
from .tools.mdmake import process_md
from .tools.pdfmake import process_pdf
from .tools.txtmake import process_txt
from .tools.imgmake import process_img, process_image_directory, get_image_statistics

__version__ = "2.0.0"
__author__ = "Cre_milvus Team"

# 模块级别的便捷接口
def process_data(directory_path: str, **kwargs):
    """
    处理数据目录的便捷接口
    
    参数:
        directory_path: 数据目录路径
        **kwargs: 处理参数
            - chunking_strategy: 分块策略 ("traditional", "meta_ppl", "margin_sampling")
            - chunking_params: 分块参数字典
            - enable_multimodal: 是否启用多模态处理
            - url_split: 是否分割URL
    
    返回:
        处理结果列表
    """
    return data_process(
        data_location=directory_path,
        url_split=kwargs.get('url_split', False),
        chunking_strategy=kwargs.get('chunking_strategy', 'traditional'),
        chunking_params=kwargs.get('chunking_params', {}),
        enable_multimodal=kwargs.get('enable_multimodal', True)
    )


def process_images(directory_path: str, use_clip: bool = True):
    """
    处理图像目录的便捷接口
    
    参数:
        directory_path: 图像目录路径
        use_clip: 是否使用CLIP进行向量化
    
    返回:
        图像处理结果列表
    """
    return batch_process_images(directory_path, use_clip)


# 导出主要接口
__all__ = [
    # 主要处理函数
    'data_process',
    'process_data', 
    'process_images',
    
    # 策略相关
    'get_chunking_strategies',
    'batch_process_images',
    
    # 工具函数
    'process_csv',
    'process_md', 
    'process_pdf',
    'process_txt',
    'process_img',
    'process_image_directory',
    'get_image_statistics'
]


# 模块初始化
def _initialize_module():
    """
    模块初始化函数
    """
    try:
        # 检查依赖
        dependencies = []
        
        try:
            from .chunking.chunk_strategies import ChunkingManager
            dependencies.append("✓ 高级分块策略")
        except ImportError:
            dependencies.append("✗ 高级分块策略 (可选)")
        
        try:
            from multimodal.clip_encoder import CLIPEncoder
            dependencies.append("✓ 多模态CLIP编码")
        except ImportError:
            dependencies.append("✗ 多模态CLIP编码 (可选)")
        
        print(f"DataBuilder模块 v{__version__} 已加载")
        print("依赖检查:")
        for dep in dependencies:
            print(f"  {dep}")
        
    except Exception as e:
        print(f"DataBuilder模块初始化警告: {e}")


# 执行初始化
_initialize_module()