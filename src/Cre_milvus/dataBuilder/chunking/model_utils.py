"""
数据模型工具函数

提供数据模型转换、验证和处理的辅助功能
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import time
import psutil
import os
try:
    from .models import (
        ChunkingProcessRequest, ChunkingProcessResponse, ChunkingErrorResponse,
        ChunkingMetrics, ChunkingStrategy
    )
except ImportError:
    from models import (
        ChunkingProcessRequest, ChunkingProcessResponse, ChunkingErrorResponse,
        ChunkingMetrics, ChunkingStrategy
    )


def create_chunking_request(
    text: str,
    strategy: str,
    params: Optional[Dict[str, Any]] = None,
    language: str = "zh",
    enable_metrics: bool = True,
    timeout: int = 30
) -> ChunkingProcessRequest:
    """
    创建分块请求对象
    
    参数:
        text: 待分块的文本
        strategy: 分块策略
        params: 策略参数
        language: 文本语言
        enable_metrics: 是否启用性能监控
        timeout: 超时时间
    
    返回:
        ChunkingProcessRequest对象
    """
    return ChunkingProcessRequest(
        text=text,
        strategy=ChunkingStrategy(strategy),
        params=params or {},
        language=language,
        enable_metrics=enable_metrics,
        timeout=timeout
    )


def create_success_response(
    request: ChunkingProcessRequest,
    chunks: List[str],
    actual_strategy: str,
    processing_time: float,
    warnings: Optional[List[str]] = None,
    metrics: Optional[ChunkingMetrics] = None
) -> ChunkingProcessResponse:
    """
    创建成功响应对象
    
    参数:
        request: 原始请求
        chunks: 分块结果
        actual_strategy: 实际使用的策略
        processing_time: 处理时间
        warnings: 警告信息
        metrics: 性能指标
    
    返回:
        ChunkingProcessResponse对象
    """
    return ChunkingProcessResponse(
        success=True,
        strategy=request.strategy.value,
        actual_strategy=actual_strategy,
        params=request.params,
        chunk_count=len(chunks),
        chunks=chunks,
        metrics=metrics,
        warnings=warnings,
        processing_time=processing_time,
        timestamp=datetime.now().isoformat()
    )


def create_error_response(
    error_type: str,
    error_message: str,
    suggested_action: str,
    fallback_strategy: Optional[str] = None
) -> ChunkingErrorResponse:
    """
    创建错误响应对象
    
    参数:
        error_type: 错误类型
        error_message: 错误信息
        suggested_action: 建议的解决方案
        fallback_strategy: 建议的降级策略
    
    返回:
        ChunkingErrorResponse对象
    """
    return ChunkingErrorResponse(
        error_type=error_type,
        error_message=error_message,
        suggested_action=suggested_action,
        fallback_strategy=fallback_strategy,
        timestamp=datetime.now().isoformat()
    )


def calculate_chunking_metrics(
    chunks: List[str],
    processing_time: float,
    strategy_used: str,
    fallback_occurred: bool = False,
    error_count: int = 0,
    quality_score: Optional[float] = None
) -> ChunkingMetrics:
    """
    计算分块性能指标
    
    参数:
        chunks: 分块结果
        processing_time: 处理时间
        strategy_used: 使用的策略
        fallback_occurred: 是否发生降级
        error_count: 错误数量
        quality_score: 质量分数
    
    返回:
        ChunkingMetrics对象
    """
    if not chunks:
        return ChunkingMetrics(
            processing_time=processing_time,
            chunk_count=0,
            average_chunk_length=0.0,
            min_chunk_length=0,
            max_chunk_length=0,
            strategy_used=strategy_used,
            fallback_occurred=fallback_occurred,
            error_count=error_count,
            quality_score=quality_score
        )
    
    chunk_lengths = [len(chunk) for chunk in chunks]
    
    # 获取内存使用情况
    memory_usage = None
    try:
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
    except Exception:
        pass
    
    return ChunkingMetrics(
        processing_time=processing_time,
        memory_usage=memory_usage,
        chunk_count=len(chunks),
        average_chunk_length=sum(chunk_lengths) / len(chunk_lengths),
        min_chunk_length=min(chunk_lengths),
        max_chunk_length=max(chunk_lengths),
        strategy_used=strategy_used,
        fallback_occurred=fallback_occurred,
        error_count=error_count,
        quality_score=quality_score
    )


def get_default_params(strategy: str) -> Dict[str, Any]:
    """
    获取策略的默认参数
    
    参数:
        strategy: 策略名称
    
    返回:
        默认参数字典
    """
    default_params = {
        ChunkingStrategy.TRADITIONAL.value: {
            "chunk_size": 512,
            "overlap": 50
        },
        ChunkingStrategy.META_PPL.value: {
            "threshold": 0.3,
            "language": "zh"
        },
        ChunkingStrategy.MARGIN_SAMPLING.value: {
            "language": "zh",
            "chunk_length": 512
        },
        ChunkingStrategy.MSP.value: {
            "language": "zh",
            "chunk_length": 512,
            "confidence_threshold": 0.7
        },
        ChunkingStrategy.SEMANTIC.value: {
            "similarity_threshold": 0.8,
            "min_chunk_size": 100,
            "max_chunk_size": 1000
        }
    }
    
    return default_params.get(strategy, {})


def merge_params(strategy: str, user_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并用户参数和默认参数
    
    参数:
        strategy: 策略名称
        user_params: 用户提供的参数
    
    返回:
        合并后的参数字典
    """
    default_params = get_default_params(strategy)
    merged_params = default_params.copy()
    merged_params.update(user_params)
    return merged_params


def validate_text_input(text: str) -> tuple[bool, Optional[str]]:
    """
    验证输入文本
    
    参数:
        text: 输入文本
    
    返回:
        (是否有效, 错误信息)
    """
    if not text:
        return False, "文本不能为空"
    
    if not isinstance(text, str):
        return False, "文本必须是字符串类型"
    
    if len(text.strip()) == 0:
        return False, "文本不能只包含空白字符"
    
    if len(text) > 100000:
        return False, "文本长度不能超过100000个字符"
    
    return True, None


def estimate_processing_time(text: str, strategy: str) -> float:
    """
    估算处理时间
    
    参数:
        text: 输入文本
        strategy: 分块策略
    
    返回:
        估算的处理时间（秒）
    """
    text_length = len(text)
    
    # 基于策略和文本长度的简单估算
    base_time = {
        ChunkingStrategy.TRADITIONAL.value: 0.001,
        ChunkingStrategy.SEMANTIC.value: 0.005,
        ChunkingStrategy.META_PPL.value: 0.02,
        ChunkingStrategy.MARGIN_SAMPLING.value: 0.015,
        ChunkingStrategy.MSP.value: 0.018
    }
    
    strategy_base = base_time.get(strategy, 0.01)
    estimated_time = strategy_base * (text_length / 1000)
    
    return max(0.1, min(estimated_time, 60.0))  # 限制在0.1-60秒之间


class ProcessingTimer:
    """处理时间计时器"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """开始计时"""
        self.start_time = time.time()
    
    def stop(self):
        """停止计时"""
        self.end_time = time.time()
    
    def get_elapsed_time(self) -> float:
        """获取经过的时间"""
        if self.start_time is None:
            return 0.0
        
        end_time = self.end_time or time.time()
        return end_time - self.start_time
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def format_error_message(error: Exception, context: str = "") -> str:
    """
    格式化错误信息
    
    参数:
        error: 异常对象
        context: 错误上下文
    
    返回:
        格式化的错误信息
    """
    error_type = type(error).__name__
    error_message = str(error)
    
    if context:
        return f"{context}: {error_type} - {error_message}"
    else:
        return f"{error_type}: {error_message}"


def get_strategy_display_name(strategy: str) -> str:
    """
    获取策略的显示名称
    
    参数:
        strategy: 策略名称
    
    返回:
        显示名称
    """
    display_names = {
        ChunkingStrategy.TRADITIONAL.value: "传统固定切分",
        ChunkingStrategy.META_PPL.value: "PPL困惑度切分",
        ChunkingStrategy.MARGIN_SAMPLING.value: "边际采样切分",
        ChunkingStrategy.MSP.value: "MSP切分策略",
        ChunkingStrategy.SEMANTIC.value: "语义切分"
    }
    
    return display_names.get(strategy, strategy)