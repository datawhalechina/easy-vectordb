"""
分块错误处理模块

提供统一的错误处理、降级机制和响应格式化功能
"""

import logging
import traceback
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from enum import Enum

try:
    from .models import ChunkingErrorResponse, ChunkingStrategy
    from .model_utils import create_error_response, format_error_message
except ImportError:
    from models import ChunkingErrorResponse, ChunkingStrategy
    from model_utils import create_error_response, format_error_message

logger = logging.getLogger(__name__)


class ErrorType(str, Enum):
    """错误类型枚举"""
    VALIDATION_ERROR = "ValidationError"
    DEPENDENCY_MISSING = "DependencyMissingError"
    PROCESSING_TIMEOUT = "ProcessingTimeoutError"
    STRATEGY_NOT_AVAILABLE = "StrategyNotAvailableError"
    PARAMETER_ERROR = "ParameterError"
    INTERNAL_ERROR = "InternalError"
    SERVICE_UNAVAILABLE = "ServiceUnavailable"
    TEXT_TOO_LONG = "TextTooLongError"
    ENCODING_ERROR = "EncodingError"
    MODEL_ERROR = "ModelError"
    API_ERROR = "APIError"


class ChunkingErrorHandler:
    """分块错误处理器"""
    
    def __init__(self):
        self.error_count = 0
        self.error_history = []
        self.fallback_chain = {
            ChunkingStrategy.META_PPL.value: [
                ChunkingStrategy.SEMANTIC.value,
                ChunkingStrategy.TRADITIONAL.value
            ],
            ChunkingStrategy.MSP.value: [
                ChunkingStrategy.MARGIN_SAMPLING.value,
                ChunkingStrategy.SEMANTIC.value,
                ChunkingStrategy.TRADITIONAL.value
            ],
            ChunkingStrategy.MARGIN_SAMPLING.value: [
                ChunkingStrategy.SEMANTIC.value,
                ChunkingStrategy.TRADITIONAL.value
            ],
            ChunkingStrategy.SEMANTIC.value: [
                ChunkingStrategy.TRADITIONAL.value
            ],
            ChunkingStrategy.TRADITIONAL.value: []  # 最终降级目标
        }
    
    def handle_dependency_error(
        self, 
        strategy: str, 
        missing_deps: List[str]
    ) -> ChunkingErrorResponse:
        """
        处理依赖缺失错误
        
        参数:
            strategy: 失败的策略
            missing_deps: 缺失的依赖列表
        
        返回:
            错误响应对象
        """
        self._record_error(ErrorType.DEPENDENCY_MISSING, strategy)
        
        fallback_strategy = self.suggest_fallback_strategy(strategy)
        
        # 生成安装建议
        install_suggestions = self._generate_install_suggestions(missing_deps)
        
        error_message = f"策略 '{strategy}' 所需依赖缺失: {', '.join(missing_deps)}"
        suggested_action = f"请安装缺失的依赖: {'; '.join(install_suggestions)}"
        
        if fallback_strategy:
            suggested_action += f" 或使用降级策略: {fallback_strategy}"
        
        return create_error_response(
            error_type=ErrorType.DEPENDENCY_MISSING.value,
            error_message=error_message,
            suggested_action=suggested_action,
            fallback_strategy=fallback_strategy
        )
    
    def handle_timeout_error(
        self, 
        strategy: str, 
        timeout: int,
        actual_time: float = None
    ) -> ChunkingErrorResponse:
        """
        处理超时错误
        
        参数:
            strategy: 超时的策略
            timeout: 设置的超时时间
            actual_time: 实际处理时间
        
        返回:
            错误响应对象
        """
        self._record_error(ErrorType.PROCESSING_TIMEOUT, strategy)
        
        fallback_strategy = self.suggest_fallback_strategy(strategy)
        
        error_message = f"策略 '{strategy}' 处理超时 (限制: {timeout}s"
        if actual_time:
            error_message += f", 实际: {actual_time:.2f}s"
        error_message += ")"
        
        suggested_action = "请尝试以下解决方案: 1) 增加超时时间 2) 减少文本长度 3) 调整策略参数"
        
        if fallback_strategy:
            suggested_action += f" 4) 使用更快的策略: {fallback_strategy}"
        
        return create_error_response(
            error_type=ErrorType.PROCESSING_TIMEOUT.value,
            error_message=error_message,
            suggested_action=suggested_action,
            fallback_strategy=fallback_strategy
        )
    
    def handle_parameter_error(
        self, 
        strategy: str, 
        invalid_params: Dict[str, Any],
        validation_errors: List[str] = None
    ) -> ChunkingErrorResponse:
        """
        处理参数错误
        
        参数:
            strategy: 策略名称
            invalid_params: 无效的参数
            validation_errors: 验证错误列表
        
        返回:
            错误响应对象
        """
        self._record_error(ErrorType.PARAMETER_ERROR, strategy)
        
        error_message = f"策略 '{strategy}' 参数验证失败"
        
        if validation_errors:
            error_message += f": {'; '.join(validation_errors)}"
        elif invalid_params:
            param_names = list(invalid_params.keys())
            error_message += f": 无效参数 {', '.join(param_names)}"
        
        # 生成参数建议
        suggested_action = self._generate_parameter_suggestions(strategy, invalid_params)
        
        return create_error_response(
            error_type=ErrorType.PARAMETER_ERROR.value,
            error_message=error_message,
            suggested_action=suggested_action
        )
    
    def handle_text_validation_error(
        self, 
        text: str, 
        validation_error: str
    ) -> ChunkingErrorResponse:
        """
        处理文本验证错误
        
        参数:
            text: 输入文本
            validation_error: 验证错误信息
        
        返回:
            错误响应对象
        """
        self._record_error(ErrorType.VALIDATION_ERROR, "text_validation")
        
        error_message = f"输入文本验证失败: {validation_error}"
        
        # 根据错误类型提供建议
        if "长度" in validation_error or "length" in validation_error.lower():
            suggested_action = "请检查文本长度，确保在1-100000字符范围内"
        elif "空" in validation_error or "empty" in validation_error.lower():
            suggested_action = "请提供非空的文本内容"
        elif "格式" in validation_error or "format" in validation_error.lower():
            suggested_action = "请确保文本为有效的字符串格式"
        else:
            suggested_action = "请检查输入文本的格式和内容"
        
        return create_error_response(
            error_type=ErrorType.VALIDATION_ERROR.value,
            error_message=error_message,
            suggested_action=suggested_action
        )
    
    def handle_strategy_unavailable_error(
        self, 
        strategy: str, 
        reason: str = None
    ) -> ChunkingErrorResponse:
        """
        处理策略不可用错误
        
        参数:
            strategy: 不可用的策略
            reason: 不可用的原因
        
        返回:
            错误响应对象
        """
        self._record_error(ErrorType.STRATEGY_NOT_AVAILABLE, strategy)
        
        fallback_strategy = self.suggest_fallback_strategy(strategy)
        
        error_message = f"策略 '{strategy}' 当前不可用"
        if reason:
            error_message += f": {reason}"
        
        suggested_action = "请检查策略依赖或使用其他可用策略"
        if fallback_strategy:
            suggested_action += f"，建议使用: {fallback_strategy}"
        
        return create_error_response(
            error_type=ErrorType.STRATEGY_NOT_AVAILABLE.value,
            error_message=error_message,
            suggested_action=suggested_action,
            fallback_strategy=fallback_strategy
        )
    
    def handle_internal_error(
        self, 
        error: Exception, 
        context: str = "",
        strategy: str = None
    ) -> ChunkingErrorResponse:
        """
        处理内部错误
        
        参数:
            error: 异常对象
            context: 错误上下文
            strategy: 相关策略
        
        返回:
            错误响应对象
        """
        self._record_error(ErrorType.INTERNAL_ERROR, strategy or "unknown")
        
        error_message = format_error_message(error, context)
        
        # 记录详细错误信息到日志
        logger.error(f"内部错误: {error_message}")
        logger.error(f"错误堆栈: {traceback.format_exc()}")
        
        fallback_strategy = None
        if strategy:
            fallback_strategy = self.suggest_fallback_strategy(strategy)
        
        suggested_action = "请稍后重试"
        if fallback_strategy:
            suggested_action += f"或使用其他策略: {fallback_strategy}"
        suggested_action += "。如问题持续，请联系管理员"
        
        return create_error_response(
            error_type=ErrorType.INTERNAL_ERROR.value,
            error_message="服务器内部错误",  # 不暴露详细错误信息给用户
            suggested_action=suggested_action,
            fallback_strategy=fallback_strategy
        )
    
    def suggest_fallback_strategy(self, failed_strategy: str) -> Optional[str]:
        """
        建议降级策略
        
        参数:
            failed_strategy: 失败的策略
        
        返回:
            建议的降级策略，如果没有则返回None
        """
        fallbacks = self.fallback_chain.get(failed_strategy, [])
        return fallbacks[0] if fallbacks else None
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        获取错误统计信息
        
        返回:
            错误统计字典
        """
        error_types = {}
        strategy_errors = {}
        
        for error_record in self.error_history:
            error_type = error_record["error_type"]
            strategy = error_record["strategy"]
            
            error_types[error_type] = error_types.get(error_type, 0) + 1
            strategy_errors[strategy] = strategy_errors.get(strategy, 0) + 1
        
        return {
            "total_errors": self.error_count,
            "error_types": error_types,
            "strategy_errors": strategy_errors,
            "recent_errors": self.error_history[-10:] if self.error_history else []
        }
    
    def _record_error(self, error_type: ErrorType, strategy: str):
        """记录错误"""
        self.error_count += 1
        
        error_record = {
            "error_type": error_type.value,
            "strategy": strategy,
            "timestamp": datetime.now().isoformat(),
            "error_id": self.error_count
        }
        
        self.error_history.append(error_record)
        
        # 保持历史记录在合理范围内
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-500:]
    
    def _generate_install_suggestions(self, missing_deps: List[str]) -> List[str]:
        """生成安装建议"""
        suggestions = []
        
        for dep in missing_deps:
            if dep == "torch":
                suggestions.append("pip install torch")
            elif dep == "nltk":
                suggestions.append("pip install nltk")
            elif dep == "jieba":
                suggestions.append("pip install jieba")
            elif dep == "perplexity_chunking模块":
                suggestions.append("确保perplexity_chunking.py文件存在")
            else:
                suggestions.append(f"安装 {dep}")
        
        return suggestions
    
    def _generate_parameter_suggestions(
        self, 
        strategy: str, 
        invalid_params: Dict[str, Any]
    ) -> str:
        """生成参数建议"""
        suggestions = []
        
        if strategy == ChunkingStrategy.TRADITIONAL.value:
            if "chunk_size" in invalid_params:
                suggestions.append("chunk_size应在100-2048范围内")
            if "overlap" in invalid_params:
                suggestions.append("overlap应在0-200范围内")
        
        elif strategy == ChunkingStrategy.META_PPL.value:
            if "threshold" in invalid_params:
                suggestions.append("threshold应在0.0-1.0范围内")
            if "language" in invalid_params:
                suggestions.append("language应为'zh'或'en'")
        
        elif strategy == ChunkingStrategy.SEMANTIC.value:
            if "similarity_threshold" in invalid_params:
                suggestions.append("similarity_threshold应在0.6-0.95范围内")
            if "min_chunk_size" in invalid_params:
                suggestions.append("min_chunk_size应在50-500范围内")
            if "max_chunk_size" in invalid_params:
                suggestions.append("max_chunk_size应在500-3000范围内")
        
        if not suggestions:
            suggestions.append("请参考API文档检查参数格式和范围")
        
        return "参数建议: " + "; ".join(suggestions)


    def get_error_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取错误历史"""
        return self.error_history[-limit:]
    
    def clear_error_history(self) -> None:
        """清除错误历史"""
        self.error_history.clear()
        self.error_count = 0
        logger.info("错误历史已清除")


class ResponseFormatter:
    """响应格式化器"""
    
    @staticmethod
    def format_success_response(response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化成功响应
        
        参数:
            response_data: 响应数据
        
        返回:
            格式化后的响应
        """
        formatted_response = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            **response_data
        }
        
        # 确保处理时间格式化
        if "processing_time" in formatted_response:
            processing_time = formatted_response["processing_time"]
            formatted_response["processing_time_formatted"] = f"{processing_time:.3f}s"
        
        # 格式化指标
        if "metrics" in formatted_response and formatted_response["metrics"]:
            metrics = formatted_response["metrics"]
            if hasattr(metrics, 'dict'):
                formatted_response["metrics"] = metrics.dict()
        
        return formatted_response
    
    @staticmethod
    def format_error_response(error_response: ChunkingErrorResponse) -> Dict[str, Any]:
        """
        格式化错误响应
        
        参数:
            error_response: 错误响应对象
        
        返回:
            格式化后的错误响应
        """
        if hasattr(error_response, 'dict'):
            return error_response.dict()
        else:
            return {
                "success": False,
                "error_type": getattr(error_response, 'error_type', 'UnknownError'),
                "error_message": getattr(error_response, 'error_message', 'Unknown error'),
                "suggested_action": getattr(error_response, 'suggested_action', 'Please try again'),
                "timestamp": datetime.now().isoformat()
            }
    
    @staticmethod
    def format_validation_response(
        is_valid: bool, 
        errors: List[str] = None
    ) -> Dict[str, Any]:
        """
        格式化验证响应
        
        参数:
            is_valid: 是否有效
            errors: 错误列表
        
        返回:
            验证响应
        """
        return {
            "valid": is_valid,
            "errors": errors or [],
            "timestamp": datetime.now().isoformat()
        }


# 全局错误处理器实例
global_error_handler = ChunkingErrorHandler()