# import sys
# import os

# project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# sys.path.insert(0, project_root)

# try:
#     from Cre_milvus.dataBuilder.chunking.models import (
#         ChunkingProcessRequest, ChunkingProcessResponse, ChunkingErrorResponse,
#         ChunkingMetrics, ChunkingStrategy
#     )
#     from Cre_milvus.dataBuilder.chunking.model_utils import (
#         create_chunking_request, create_success_response, create_error_response,
#         calculate_chunking_metrics, validate_text_input, get_default_params
#     )
# except ImportError as e:
#     print(f"导入错误: {e}")
#     print("尝试本地导入...")
    
    
#     current_dir = os.path.dirname(__file__)
#     sys.path.insert(0, current_dir)
    
#     from models import (
#         ChunkingProcessRequest, ChunkingProcessResponse, ChunkingErrorResponse,
#         ChunkingMetrics, ChunkingStrategy
#     )
#     import model_utils
#     create_chunking_request = model_utils.create_chunking_request
#     create_success_response = model_utils.create_success_response
#     create_error_response = model_utils.create_error_response
#     calculate_chunking_metrics = model_utils.calculate_chunking_metrics
#     validate_text_input = model_utils.validate_text_input
#     get_default_params = model_utils.get_default_params


# def test_chunking_request_creation():
#     """测试分块请求创建"""
#     print("测试分块请求创建...")
    
#     # 测试有效请求
#     try:
#         request = create_chunking_request(
#             text="这是一个测试文本，用于验证分块功能。",
#             strategy="traditional",
#             params={"chunk_size": 512, "overlap": 50},
#             language="zh"
#         )
#         print(f"✅ 成功创建请求: {request.strategy}")
#     except Exception as e:
#         print(f"❌ 创建请求失败: {e}")
    
#     # 测试参数验证
#     try:
#         request = ChunkingProcessRequest(
#             text="测试文本",
#             strategy=ChunkingStrategy.TRADITIONAL,
#             params={"chunk_size": 5000}  # 超出范围
#         )
#         print("❌ 参数验证失败 - 应该抛出异常")
#     except Exception as e:
#         print(f"✅ 参数验证正常: {e}")


# def test_response_creation():
#     """测试响应创建"""
#     print("\n测试响应创建...")
    
#     # 创建测试请求
#     request = create_chunking_request(
#         text="测试文本",
#         strategy="traditional"
#     )
    
#     # 测试成功响应
#     chunks = ["测试", "文本"]
#     metrics = calculate_chunking_metrics(
#         chunks=chunks,
#         processing_time=0.5,
#         strategy_used="traditional"
#     )
    
#     response = create_success_response(
#         request=request,
#         chunks=chunks,
#         actual_strategy="traditional",
#         processing_time=0.5,
#         metrics=metrics
#     )
    
#     print(f"✅ 成功响应创建: {response.chunk_count} 个分块")
    
#     # 测试错误响应
#     error_response = create_error_response(
#         error_type="ValidationError",
#         error_message="参数验证失败",
#         suggested_action="请检查输入参数"
#     )
    
#     print(f"✅ 错误响应创建: {error_response.error_type}")


# def test_text_validation():
#     """测试文本验证"""
#     print("\n测试文本验证...")
    
#     test_cases = [
#         ("正常文本", True),
#         ("", False),
#         ("   ", False),
#         ("a" * 100001, False),  # 超长文本
#         (None, False)
#     ]
    
#     for text, expected in test_cases:
#         try:
#             is_valid, error_msg = validate_text_input(text)
#             if is_valid == expected:
#                 print(f"✅ 文本验证正确: '{text[:20]}...' -> {is_valid}")
#             else:
#                 print(f"❌ 文本验证错误: '{text[:20]}...' -> {is_valid}, 期望: {expected}")
#         except Exception as e:
#             if not expected:
#                 print(f"✅ 文本验证正确（异常）: '{text}' -> {e}")
#             else:
#                 print(f"❌ 文本验证错误（异常）: '{text}' -> {e}")


# def test_default_params():
#     """测试默认参数"""
#     print("\n测试默认参数...")
    
#     strategies = [
#         ChunkingStrategy.TRADITIONAL.value,
#         ChunkingStrategy.META_PPL.value,
#         ChunkingStrategy.MARGIN_SAMPLING.value,
#         ChunkingStrategy.MSP.value,
#         ChunkingStrategy.SEMANTIC.value
#     ]
    
#     for strategy in strategies:
#         params = get_default_params(strategy)
#         print(f"✅ {strategy} 默认参数: {params}")


# def test_metrics_calculation():
#     """测试指标计算"""
#     print("\n测试指标计算...")
    
#     chunks = ["这是第一个分块", "这是第二个分块", "这是第三个分块"]
#     metrics = calculate_chunking_metrics(
#         chunks=chunks,
#         processing_time=1.5,
#         strategy_used="traditional",
#         fallback_occurred=False,
#         quality_score=0.85
#     )
    
#     print(f"✅ 指标计算完成:")
#     print(f"   - 分块数量: {metrics.chunk_count}")
#     print(f"   - 平均长度: {metrics.average_chunk_length:.2f}")
#     print(f"   - 处理时间: {metrics.processing_time}s")
#     print(f"   - 质量分数: {metrics.quality_score}")


# if __name__ == "__main__":
#     print("开始测试数据模型...")
    
#     test_chunking_request_creation()
#     test_response_creation()
#     test_text_validation()
#     test_default_params()
#     test_metrics_calculation()
    
#     print("\n数据模型测试完成！")