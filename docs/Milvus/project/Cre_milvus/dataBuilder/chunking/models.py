"""
文本分块相关的数据模型

定义分块请求、响应和相关数据结构
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class ChunkingStrategy(str, Enum):
    """分块策略枚举"""
    TRADITIONAL = "traditional"
    META_PPL = "meta_ppl"
    MARGIN_SAMPLING = "margin_sampling"
    MSP = "msp"
    SEMANTIC = "semantic"


class ChunkingProcessRequest(BaseModel):
    """分块处理请求模型"""
    text: str = Field(
        ..., 
        min_length=1, 
        max_length=100000, 
        description="待分割的文本"
    )
    strategy: ChunkingStrategy = Field(
        ..., 
        description="分割策略"
    )
    params: Optional[Dict[str, Any]] = Field(
        default_factory=dict, 
        description="策略参数"
    )
    language: Optional[str] = Field(
        default="zh", 
        description="文本语言",
        pattern="^(zh|en)$"
    )
    enable_metrics: Optional[bool] = Field(
        default=True, 
        description="是否启用性能监控"
    )
    timeout: Optional[int] = Field(
        default=30, 
        ge=5, 
        le=300, 
        description="处理超时时间（秒）"
    )

    @validator('params')
    def validate_params(cls, v, values):
        """验证策略参数"""
        if not isinstance(v, dict):
            raise ValueError("params must be a dictionary")
        
        strategy = values.get('strategy')
        if strategy:
            # 根据策略验证参数
            if strategy == ChunkingStrategy.TRADITIONAL:
                cls._validate_traditional_params(v)
            elif strategy == ChunkingStrategy.META_PPL:
                cls._validate_ppl_params(v)
            elif strategy == ChunkingStrategy.MARGIN_SAMPLING:
                cls._validate_margin_sampling_params(v)
            elif strategy == ChunkingStrategy.MSP:
                cls._validate_msp_params(v)
            elif strategy == ChunkingStrategy.SEMANTIC:
                cls._validate_semantic_params(v)
        
        return v

    @staticmethod
    def _validate_traditional_params(params: Dict[str, Any]):
        """验证传统分块参数"""
        if 'chunk_size' in params:
            chunk_size = params['chunk_size']
            if not isinstance(chunk_size, int) or chunk_size < 100 or chunk_size > 2048:
                raise ValueError("chunk_size must be an integer between 100 and 2048")
        
        if 'overlap' in params:
            overlap = params['overlap']
            if not isinstance(overlap, int) or overlap < 0 or overlap > 200:
                raise ValueError("overlap must be an integer between 0 and 200")

    @staticmethod
    def _validate_ppl_params(params: Dict[str, Any]):
        """验证PPL分块参数"""
        if 'threshold' in params:
            threshold = params['threshold']
            if not isinstance(threshold, (int, float)) or threshold < 0.0 or threshold > 1.0:
                raise ValueError("threshold must be a number between 0.0 and 1.0")

    @staticmethod
    def _validate_margin_sampling_params(params: Dict[str, Any]):
        """验证边际采样分块参数"""
        if 'chunk_length' in params:
            chunk_length = params['chunk_length']
            if not isinstance(chunk_length, int) or chunk_length < 100 or chunk_length > 2048:
                raise ValueError("chunk_length must be an integer between 100 and 2048")

    @staticmethod
    def _validate_msp_params(params: Dict[str, Any]):
        """验证MSP分块参数"""
        if 'chunk_length' in params:
            chunk_length = params['chunk_length']
            if not isinstance(chunk_length, int) or chunk_length < 100 or chunk_length > 2048:
                raise ValueError("chunk_length must be an integer between 100 and 2048")
        
        if 'confidence_threshold' in params:
            confidence_threshold = params['confidence_threshold']
            if not isinstance(confidence_threshold, (int, float)) or confidence_threshold < 0.5 or confidence_threshold > 0.95:
                raise ValueError("confidence_threshold must be a number between 0.5 and 0.95")

    @staticmethod
    def _validate_semantic_params(params: Dict[str, Any]):
        """验证语义分块参数"""
        if 'similarity_threshold' in params:
            similarity_threshold = params['similarity_threshold']
            if not isinstance(similarity_threshold, (int, float)) or similarity_threshold < 0.6 or similarity_threshold > 0.95:
                raise ValueError("similarity_threshold must be a number between 0.6 and 0.95")
        
        if 'min_chunk_size' in params:
            min_chunk_size = params['min_chunk_size']
            if not isinstance(min_chunk_size, int) or min_chunk_size < 50 or min_chunk_size > 500:
                raise ValueError("min_chunk_size must be an integer between 50 and 500")
        
        if 'max_chunk_size' in params:
            max_chunk_size = params['max_chunk_size']
            if not isinstance(max_chunk_size, int) or max_chunk_size < 500 or max_chunk_size > 3000:
                raise ValueError("max_chunk_size must be an integer between 500 and 3000")


class ChunkingMetrics(BaseModel):
    """分块性能指标模型"""
    processing_time: float = Field(description="处理时间（秒）")
    memory_usage: Optional[float] = Field(default=None, description="内存使用量（MB）")
    chunk_count: int = Field(description="分块数量")
    average_chunk_length: float = Field(description="平均分块长度")
    min_chunk_length: int = Field(description="最小分块长度")
    max_chunk_length: int = Field(description="最大分块长度")
    strategy_used: str = Field(description="实际使用的策略")
    fallback_occurred: bool = Field(description="是否发生了策略降级")
    error_count: int = Field(default=0, description="错误数量")
    quality_score: Optional[float] = Field(default=None, description="质量分数")


class ChunkingProcessResponse(BaseModel):
    """分块处理响应模型"""
    success: bool = Field(description="处理是否成功")
    strategy: str = Field(description="请求的策略")
    actual_strategy: str = Field(description="实际使用的策略（可能因降级而不同）")
    params: Dict[str, Any] = Field(description="使用的参数")
    chunk_count: int = Field(description="分块数量")
    chunks: List[str] = Field(description="分块结果")
    metrics: Optional[ChunkingMetrics] = Field(default=None, description="性能指标")
    warnings: Optional[List[str]] = Field(default=None, description="警告信息")
    processing_time: float = Field(description="处理时间（秒）")
    timestamp: str = Field(description="处理时间戳")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ChunkingErrorResponse(BaseModel):
    """分块错误响应模型"""
    success: bool = Field(default=False, description="处理是否成功")
    error_type: str = Field(description="错误类型")
    error_message: str = Field(description="错误信息")
    suggested_action: str = Field(description="建议的解决方案")
    fallback_strategy: Optional[str] = Field(default=None, description="建议的降级策略")
    timestamp: str = Field(description="错误发生时间戳")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TestCase(BaseModel):
    """测试用例模型"""
    name: str = Field(description="测试用例名称")
    text: str = Field(description="测试文本")
    language: str = Field(description="文本语言")
    expected_min_chunks: int = Field(description="期望的最小分块数")
    expected_max_chunks: int = Field(description="期望的最大分块数")
    strategy_params: Dict[str, Any] = Field(description="策略参数")
    quality_threshold: float = Field(description="质量阈值")
    performance_threshold: float = Field(description="性能阈值（秒）")


class StrategyTestResult(BaseModel):
    """策略测试结果模型"""
    strategy: str = Field(description="测试的策略")
    test_cases_run: int = Field(description="运行的测试用例数")
    success_rate: float = Field(description="成功率")
    average_processing_time: float = Field(description="平均处理时间")
    average_quality_score: float = Field(description="平均质量分数")
    error_details: List[str] = Field(description="错误详情")
    performance_metrics: Dict[str, float] = Field(description="性能指标")


class TestReport(BaseModel):
    """测试报告模型"""
    test_id: str = Field(description="测试ID")
    timestamp: datetime = Field(description="测试时间戳")
    total_tests: int = Field(description="总测试数")
    passed_tests: int = Field(description="通过的测试数")
    failed_tests: int = Field(description="失败的测试数")
    strategy_results: Dict[str, StrategyTestResult] = Field(description="各策略测试结果")
    performance_summary: Dict[str, Any] = Field(description="性能摘要")
    quality_summary: Dict[str, Any] = Field(description="质量摘要")
    recommendations: List[str] = Field(description="建议")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PerformanceBenchmark(BaseModel):
    """性能基准测试模型"""
    benchmark_id: str = Field(description="基准测试ID")
    timestamp: datetime = Field(description="测试时间戳")
    strategies_tested: List[str] = Field(description="测试的策略列表")
    test_data_size: int = Field(description="测试数据大小")
    results: Dict[str, Dict[str, float]] = Field(description="各策略的性能结果")
    winner: str = Field(description="性能最佳策略")
    recommendations: List[str] = Field(description="性能优化建议")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ChunkingConfig(BaseModel):
    """分块配置模型"""
    default_strategy: ChunkingStrategy = Field(default=ChunkingStrategy.TRADITIONAL, description="默认策略")
    enable_fallback: bool = Field(default=True, description="是否启用策略降级")
    max_text_length: int = Field(default=100000, description="最大文本长度")
    default_timeout: int = Field(default=30, description="默认超时时间")
    enable_metrics: bool = Field(default=True, description="是否启用性能监控")
    log_level: str = Field(default="INFO", description="日志级别")