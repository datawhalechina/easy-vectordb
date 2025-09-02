from enum import Enum
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

try:
    from .meta_chunking import MetaChunking
except ImportError:
    from meta_chunking import MetaChunking


class ChunkingStrategy(Enum):
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
    
    def __init__(self, model=None, tokenizer=None, config=None):
        """
        初始化切分管理器
        
        参数:
            model: 语言模型（用于智能切分策略）
            tokenizer: 分词器
            config: 配置字典，用于自动加载模型或GLM配置
        """
        # 延迟加载模型和分词器
        self._model = model
        self._tokenizer = tokenizer
        self._model_loaded = False
        self.config = config or {}
        
        # 获取LLM配置管理器
        self.llm_config_manager = self._get_llm_config_manager()
        
        # 延迟初始化MetaChunking
        self._meta_chunking = None
    
    def _get_llm_config_manager(self):
        """获取LLM配置管理器"""
        try:
            from .llm_config import create_llm_config_manager
            return create_llm_config_manager()
        except Exception as e:
            logger.warning(f"获取LLM配置管理器失败: {e}")
            return None
    
    def _ensure_model_loaded(self):
        """确保模型已加载（延迟加载）"""
        if self._model_loaded:
            return
        
        try:
            # 如果没有提供模型，尝试从配置加载
            if self._model is None and self._tokenizer is None:
                logger.info("🔄 开始加载模型...")
                self._model, self._tokenizer = self._load_model_from_config(self.config)
                logger.info("✅ 模型加载完成")
            
            # 初始化MetaChunking
            if self._meta_chunking is None:
                self._meta_chunking = MetaChunking(self._model, self._tokenizer, None)
                logger.info("✅ MetaChunking初始化完成")
            
            self._model_loaded = True
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            # 即使加载失败，也标记为已尝试，避免重复尝试
            self._model_loaded = True
    
    @property
    def meta_chunking(self):
        """获取MetaChunking实例（延迟加载）"""
        self._ensure_model_loaded()
        return self._meta_chunking
    
    def _create_llm_client(self):
        """通过LLM配置管理器创建LLM客户端"""
        if not self.llm_config_manager:
            logger.warning("LLM配置管理器不可用，无法创建LLM客户端")
            return None
            
        try:
            # 使用配置管理器创建API客户端
            client = self.llm_config_manager.create_api_client()
            if client:
                logger.info("成功创建LLM API客户端")
                return client
            else:
                logger.debug("未找到有效的LLM配置或创建客户端失败")
                return None
        except Exception as e:
            logger.error(f"创建LLM客户端失败: {e}")
            return None
    
    def _load_model_from_config(self, config):
        """
        从配置加载模型和分词器
        
        参数:
            config: 配置字典
        
        返回:
            (model, tokenizer) 元组，如果加载失败则返回 (None, None)
        """
        try:
            chunking_config = config.get("chunking", {})
            model_config = chunking_config.get("model", {})
            
            # 检查是否启用高级分块
            if not model_config.get("enable_advanced_chunking", False):
                return None, None
            
            model_name = model_config.get("model_name")
            tokenizer_name = model_config.get("tokenizer_name", model_name)
            device = model_config.get("device", "auto")
            
            if not model_name:
                return None, None
            
            # 尝试加载模型 - ModelScope
            try:
                import torch
                import torch
                
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                
                # 设备选择
                if device == "auto":
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map=device if device != "cpu" else None
                )
                
                if device == "cpu":
                    model = model.to(device)
                
                return model, tokenizer
                
            except Exception as e:
                print(f"Warning: Failed to load model {model_name}: {e}")
                return None, None
                
        except Exception as e:
            print(f"Warning: Failed to load model from config: {e}")
            return None, None
    
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
        
        try:
            # 确保MetaChunking有可用的API客户端
            if not self.meta_chunking.api_client:
                self.meta_chunking.api_client = self._create_llm_client()
            
            return self.meta_chunking.ppl_chunking(text, threshold, language)
        except Exception as e:
            logger.error(f"PPL分块失败: {e}")
            # 降级到传统分块
            chunk_size = kwargs.get('chunk_size', 512)
            overlap = kwargs.get('overlap', 50)
            return self.meta_chunking.traditional_chunking(text, chunk_size, overlap)
    
    def _margin_sampling_chunking(self, text: str, **kwargs) -> List[str]:
        """边际采样切分"""
        language = kwargs.get('language', 'zh')
        chunk_length = kwargs.get('chunk_length', 512)
        
        try:
            # 确保MetaChunking有可用的API客户端
            if not self.meta_chunking.api_client:
                self.meta_chunking.api_client = self._create_llm_client()
            
            return self.meta_chunking.margin_sampling_chunking(text, language, chunk_length)
        except Exception as e:
            logger.error(f"边际采样分块失败: {e}")
            # 降级到传统分块
            chunk_size = kwargs.get('chunk_size', 512)
            overlap = kwargs.get('overlap', 50)
            return self.meta_chunking.traditional_chunking(text, chunk_size, overlap)
    
    def _msp_chunking(self, text: str, **kwargs) -> List[str]:
        """MSP切分策略"""
        language = kwargs.get('language', 'zh')
        chunk_length = kwargs.get('chunk_length', 512)
        confidence_threshold = kwargs.get('confidence_threshold', 0.7)
        
        try:
            # 确保MetaChunking有可用的API客户端
            if not self.meta_chunking.api_client:
                self.meta_chunking.api_client = self._create_llm_client()
            
            return self.meta_chunking.msp_chunking(text, language, chunk_length, confidence_threshold)
        except Exception as e:
            logger.error(f"MSP分块失败: {e}")
            # 降级到传统分块
            chunk_size = kwargs.get('chunk_size', 512)
            overlap = kwargs.get('overlap', 50)
            return self.meta_chunking.traditional_chunking(text, chunk_size, overlap)
    
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
    
    def refresh_llm_client(self):
        """刷新LLM客户端（当LLM配置更新时调用）"""
        try:
            new_client = self._create_llm_client()
            meta_chunking = self.meta_chunking  # 触发延迟加载
            if meta_chunking:
                meta_chunking.api_client = new_client
                logger.info("ChunkingManager LLM客户端已刷新")
        except Exception as e:
            logger.error(f"刷新LLM客户端失败: {e}")
    
    def get_llm_status(self) -> Dict[str, Any]:
        """获取LLM状态信息"""
        # 延迟加载检查
        meta_chunking = self.meta_chunking  # 这会触发延迟加载
        
        status = {
            "llm_config_manager_available": self.llm_config_manager is not None,
            "api_client_available": meta_chunking.api_client is not None if meta_chunking else False,
            "local_model_available": (meta_chunking.model is not None and 
                                    meta_chunking.tokenizer is not None) if meta_chunking else False
        }
        
        # LLM配置状态
        if self.llm_config_manager:
            try:
                config_summary = self.llm_config_manager.get_config_summary()
                status["total_configs"] = config_summary.get("total_configs", 0)
                status["available_providers"] = config_summary.get("available_providers", 0)
                
                active_config = config_summary.get("active_config")
                if active_config:
                    status["active_provider"] = active_config.get("provider")
                    status["active_model"] = active_config.get("model")
                    status["llm_configured"] = True
                else:
                    status["llm_configured"] = False
                    
            except Exception as e:
                logger.error(f"获取LLM配置状态失败: {e}")
                status["llm_configured"] = False
        
        # API客户端类型
        if meta_chunking and meta_chunking.api_client:
            try:
                client_type = type(meta_chunking.api_client).__name__
                status["api_client_type"] = client_type
            except:
                status["api_client_type"] = "unknown"
        
        return status