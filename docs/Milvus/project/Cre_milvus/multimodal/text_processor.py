"""
文本处理模块

处理各种格式的文本文件，集成文本切分功能
"""

import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
try:
    from ..chunking import ChunkingManager
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from chunking import ChunkingManager

logger = logging.getLogger(__name__)


class TextProcessor:
    """
    文本处理器，支持多种文本格式和切分策略
    """
    
    def __init__(self, chunking_manager: Optional[ChunkingManager] = None):
        """
        初始化文本处理器
        
        参数:
            chunking_manager: 文本切分管理器
        """
        self.chunking_manager = chunking_manager
        self.supported_formats = ['.txt', '.md', '.csv', '.pdf']
    
    def process_file(self, file_path: str, chunking_config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        处理单个文本文件
        
        参数:
            file_path: 文件路径
            chunking_config: 切分配置
        
        返回:
            处理后的文本块列表
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 根据文件扩展名选择处理方法
        extension = file_path.suffix.lower()
        
        if extension == '.txt':
            content = self._process_txt(file_path)
        elif extension == '.md':
            content = self._process_markdown(file_path)
        elif extension == '.csv':
            content = self._process_csv(file_path)
        elif extension == '.pdf':
            content = self._process_pdf(file_path)
        else:
            logger.warning(f"不支持的文件格式: {extension}")
            return []
        
        # 应用文本切分
        if self.chunking_manager and chunking_config:
            chunks = self._apply_chunking(content, chunking_config)
        else:
            chunks = [content]  # 不切分，整个文件作为一个块
        
        # 构建结果
        results = []
        for i, chunk in enumerate(chunks):
            results.append({
                'id': f"{file_path.stem}_{i}",
                'content': chunk,
                'source_file': str(file_path),
                'chunk_index': i,
                'total_chunks': len(chunks),
                'content_type': 'text'
            })
        
        return results
    
    def process_directory(self, directory_path: str, chunking_config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        处理目录中的所有文本文件
        
        参数:
            directory_path: 目录路径
            chunking_config: 切分配置
        
        返回:
            所有处理后的文本块列表
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"目录不存在: {directory_path}")
        
        all_results = []
        
        # 遍历目录中的所有支持的文件
        for file_path in directory_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                try:
                    results = self.process_file(str(file_path), chunking_config)
                    all_results.extend(results)
                    logger.info(f"成功处理文件: {file_path}")
                except Exception as e:
                    logger.error(f"处理文件失败 {file_path}: {e}")
        
        return all_results
    
    def _process_txt(self, file_path: Path) -> str:
        """处理TXT文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # 尝试其他编码
            with open(file_path, 'r', encoding='gbk') as f:
                return f.read()
    
    def _process_markdown(self, file_path: Path) -> str:
        """处理Markdown文件"""
        return self._process_txt(file_path)  # Markdown本质上也是文本文件
    
    def _process_csv(self, file_path: Path) -> str:
        """处理CSV文件"""
        try:
            df = pd.read_csv(file_path)
            # 将CSV转换为文本格式
            content = f"文件: {file_path.name}\n\n"
            content += df.to_string(index=False)
            return content
        except Exception as e:
            logger.error(f"处理CSV文件失败: {e}")
            return ""
    
    def _process_pdf(self, file_path: Path) -> str:
        """处理PDF文件"""
        try:
            import PyPDF2
            content = ""
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    content += page.extract_text() + "\n"
            return content
        except ImportError:
            logger.warning("PyPDF2未安装，无法处理PDF文件")
            return ""
        except Exception as e:
            logger.error(f"处理PDF文件失败: {e}")
            return ""
    
    def _apply_chunking(self, content: str, chunking_config: Dict[str, Any]) -> List[str]:
        """
        应用文本切分
        
        参数:
            content: 原始文本内容
            chunking_config: 切分配置
        
        返回:
            切分后的文本块列表
        """
        if not self.chunking_manager:
            return [content]
        
        strategy = chunking_config.get('strategy', 'traditional')
        
        try:
            chunks = self.chunking_manager.chunk_text(content, strategy, **chunking_config)
            return chunks
        except Exception as e:
            logger.error(f"文本切分失败: {e}")
            return [content]  # 切分失败时返回原始内容
    
    def get_text_stats(self, content: str) -> Dict[str, int]:
        """
        获取文本统计信息
        
        参数:
            content: 文本内容
        
        返回:
            统计信息字典
        """
        return {
            'char_count': len(content),
            'word_count': len(content.split()),
            'line_count': len(content.split('\n')),
            'paragraph_count': len([p for p in content.split('\n\n') if p.strip()])
        }