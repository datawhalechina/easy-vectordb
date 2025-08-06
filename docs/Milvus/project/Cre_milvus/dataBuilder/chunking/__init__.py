# """
# 数据分块模块

# 提供多种文本分块策略，用于数据处理流程
# """

# try:
#     from .meta_chunking import MetaChunking
#     from .chunk_strategies import ChunkingStrategy, ChunkingManager, get_available_strategies
#     from .models import (
#         ChunkingProcessRequest, ChunkingProcessResponse, ChunkingErrorResponse,
#         ChunkingMetrics, TestCase, StrategyTestResult, TestReport,
#         PerformanceBenchmark, ChunkingConfig
#     )
#     from .model_utils import (
#         create_chunking_request, create_success_response, create_error_response,
#         calculate_chunking_metrics, get_default_params, merge_params,
#         validate_text_input, estimate_processing_time, ProcessingTimer,
#         format_error_message, get_strategy_display_name
#     )
#     from .error_handler import (
#         ChunkingErrorHandler, ResponseFormatter, ErrorType, global_error_handler
#     )
# except ImportError:
#     from meta_chunking import MetaChunking
#     from chunk_strategies import ChunkingStrategy, ChunkingManager, get_available_strategies
#     from models import (
#         ChunkingProcessRequest, ChunkingProcessResponse, ChunkingErrorResponse,
#         ChunkingMetrics, TestCase, StrategyTestResult, TestReport,
#         PerformanceBenchmark, ChunkingConfig
#     )
#     from model_utils import (
#         create_chunking_request, create_success_response, create_error_response,
#         calculate_chunking_metrics, get_default_params, merge_params,
#         validate_text_input, estimate_processing_time, ProcessingTimer,
#         format_error_message, get_strategy_display_name
#     )
#     from error_handler import (
#         ChunkingErrorHandler, ResponseFormatter, ErrorType, global_error_handler
#     )

# __all__ = [
#     'MetaChunking', 'ChunkingStrategy', 'ChunkingManager', 'get_available_strategies',
#     'ChunkingProcessRequest', 'ChunkingProcessResponse', 'ChunkingErrorResponse',
#     'ChunkingMetrics', 'TestCase', 'StrategyTestResult', 'TestReport',
#     'PerformanceBenchmark', 'ChunkingConfig',
#     'create_chunking_request', 'create_success_response', 'create_error_response',
#     'calculate_chunking_metrics', 'get_default_params', 'merge_params',
#     'validate_text_input', 'estimate_processing_time', 'ProcessingTimer',
#     'format_error_message', 'get_strategy_display_name',
#     'ChunkingErrorHandler', 'ResponseFormatter', 'ErrorType', 'global_error_handler'
# ]
try:
    from .perplexity_chunking import Chunking
    PERPLEXITY_AVAILABLE = True
except ImportError:
    PERPLEXITY_AVAILABLE = False