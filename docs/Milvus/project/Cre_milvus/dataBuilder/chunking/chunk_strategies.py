"""
文本切分策略管理模块

提供统一的切分策略接口和管理功能
"""

from enum import Enum
from typing import List, Dict, Any
try:
    from .meta_chunking import MetaChunking
except ImportError:
    from meta_chunking import MetaChunking


class ChunkingStrategy(Enum):
    """切分策略枚举"""
    TRADITIONAL = "traditional"
    META_PPL = "meta_ppl"
    MARGIN_SAMPLING = "margin_sampling"
    MSP = "msp"
    SEMANTIC = "semantic"


def get_available_strategies() -> List[Dict[str, str]]:
    """
    获取可用的切分策略列表
    
    返回:
        策略列表，包含策略名称和描述
    """
    return [
        {
            "name": ChunkingStrategy.TRADITIONAL.value,
            "display_name": "传统固定切分",
            "description": "基于固定长度和重叠的传统切分方法，简单高效"
        },
        {
            "name": ChunkingStrategy.META_PPL.value,
            "display_name": "PPL困惑度切分",
            "description": "基于语言模型困惑度的智能切分方法，在语义边界处切分"
        },
        {
            "name": ChunkingStrategy.MARGIN_SAMPLING.value,
            "display_name": "边际采样切分(MSP)",
            "description": "基于概率决策的边际采样切分方法，动态调整切分点"
        },
        {
            "name": ChunkingStrategy.MSP.value,
            "display_name": "MSP切分策略",
            "description": "Margin Sampling Partitioning，基于边际概率的高级切分策略"
        },
        {
            "name": ChunkingStrategy.SEMANTIC.value,
            "display_name": "语义切分",
            "description": "基于语义相似度的智能切分，保持语义完整性"
        }
    ]


class ChunkingManager:
    """
    切分管理器，提供统一的切分接口
    """
    
    def __init__(self, model=None, tokenizer=None):
        """
        初始化切分管理器
        
        参数:
            model: 语言模型（用于智能切分策略）
            tokenizer: 分词器
        """
        self.meta_chunking = MetaChunking(model, tokenizer)
    
    def chunk_text(self, text: str, strategy: str, **kwargs) -> List[str]:
        """
        执行文本切分
        
        参数:
            text: 输入文本
            strategy: 切分策略
            **kwargs: 策略特定参数
        
        返回:
            切分后的文本块列表
        """
        if strategy == ChunkingStrategy.TRADITIONAL.value:
            return self._traditional_chunking(text, **kwargs)
        elif strategy == ChunkingStrategy.META_PPL.value:
            return self._ppl_chunking(text, **kwargs)
        elif strategy == ChunkingStrategy.MARGIN_SAMPLING.value:
            return self._margin_sampling_chunking(text, **kwargs)
        elif strategy == ChunkingStrategy.MSP.value:
            return self._msp_chunking(text, **kwargs)
        elif strategy == ChunkingStrategy.SEMANTIC.value:
            return self._semantic_chunking(text, **kwargs)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    def _traditional_chunking(self, text: str, **kwargs) -> List[str]:
        """传统切分"""
        chunk_size = kwargs.get('chunk_size', 512)
        overlap = kwargs.get('overlap', 50)
        return self.meta_chunking.traditional_chunking(text, chunk_size, overlap)
    
    def _ppl_chunking(self, text: str, **kwargs) -> List[str]:
        """PPL困惑度切分"""
        threshold = kwargs.get('threshold', 0.3)
        language = kwargs.get('language', 'zh')
        return self.meta_chunking.ppl_chunking(text, threshold, language)
    
    def _margin_sampling_chunking(self, text: str, **kwargs) -> List[str]:
        """边际采样切分"""
        language = kwargs.get('language', 'zh')
        chunk_length = kwargs.get('chunk_length', 512)
        return self.meta_chunking.margin_sampling_chunking(text, language, chunk_length)
    
    def _msp_chunking(self, text: str, **kwargs) -> List[str]:
        """MSP切分策略"""
        language = kwargs.get('language', 'zh')
        chunk_length = kwargs.get('chunk_length', 512)
        confidence_threshold = kwargs.get('confidence_threshold', 0.7)
        return self.meta_chunking.msp_chunking(text, language, chunk_length, confidence_threshold)
    
    def _semantic_chunking(self, text: str, **kwargs) -> List[str]:
        """语义切分"""
        similarity_threshold = kwargs.get('similarity_threshold', 0.8)
        min_chunk_size = kwargs.get('min_chunk_size', 100)
        max_chunk_size = kwargs.get('max_chunk_size', 1000)
        return self.meta_chunking.semantic_chunking(text, similarity_threshold, min_chunk_size, max_chunk_size)
    
    def get_strategy_config(self, strategy: str) -> Dict[str, Any]:
        """
        获取策略的配置参数
        
        参数:
            strategy: 策略名称
        
        返回:
            配置参数字典
        """
        configs = {
            ChunkingStrategy.TRADITIONAL.value: {
                "chunk_size": {
                    "type": "int",
                    "default": 512,
                    "min": 100,
                    "max": 2048,
                    "description": "文本块大小"
                },
                "overlap": {
                    "type": "int", 
                    "default": 50,
                    "min": 0,
                    "max": 200,
                    "description": "重叠大小"
                }
            },
            ChunkingStrategy.META_PPL.value: {
                "threshold": {
                    "type": "float",
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "description": "困惑度阈值"
                },
                "language": {
                    "type": "select",
                    "default": "zh",
                    "options": ["zh", "en"],
                    "description": "语言类型"
                }
            },
            ChunkingStrategy.MARGIN_SAMPLING.value: {
                "language": {
                    "type": "select",
                    "default": "zh", 
                    "options": ["zh", "en"],
                    "description": "语言类型"
                },
                "chunk_length": {
                    "type": "int",
                    "default": 512,
                    "min": 100,
                    "max": 2048,
                    "description": "最大块长度"
                }
            },
            ChunkingStrategy.MSP.value: {
                "language": {
                    "type": "select",
                    "default": "zh",
                    "options": ["zh", "en"],
                    "description": "语言类型"
                },
                "chunk_length": {
                    "type": "int",
                    "default": 512,
                    "min": 100,
                    "max": 2048,
                    "description": "最大块长度"
                },
                "confidence_threshold": {
                    "type": "float",
                    "default": 0.7,
                    "min": 0.5,
                    "max": 0.95,
                    "step": 0.05,
                    "description": "置信度阈值"
                }
            },
            ChunkingStrategy.SEMANTIC.value: {
                "similarity_threshold": {
                    "type": "float",
                    "default": 0.8,
                    "min": 0.6,
                    "max": 0.95,
                    "step": 0.05,
                    "description": "语义相似度阈值"
                },
                "min_chunk_size": {
                    "type": "int",
                    "default": 100,
                    "min": 50,
                    "max": 500,
                    "description": "最小块大小"
                },
                "max_chunk_size": {
                    "type": "int",
                    "default": 1000,
                    "min": 500,
                    "max": 3000,
                    "description": "最大块大小"
                }
            }
        }
        
        return configs.get(strategy, {})