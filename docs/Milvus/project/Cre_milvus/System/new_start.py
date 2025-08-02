"""
新的启动模块
使用预连接架构，避免连接阻塞问题
"""

import logging
import os
import time
from typing import Dict, Any, List
from .connection_initializer import get_initializer, is_initialized

logger = logging.getLogger(__name__)

def validate_data_list(data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """验证和清理数据列表"""
    if not data_list:
        return []
    
    valid_data = []
    for i, data in enumerate(data_list):
        if not isinstance(data, dict):
            logger.warning(f"跳过无效数据项 {i}: 不是字典格式")
            continue
        
        if "embedding" not in data or not data["embedding"]:
            logger.warning(f"跳过无效数据项 {i}: 缺少embedding")
            continue
        
        if "content" not in data or not data["content"]:
            logger.warning(f"跳过无效数据项 {i}: 缺少content")
            continue
        
        if "id" not in data:
            data["id"] = i  # 自动分配ID
        
        valid_data.append(data)
    
    logger.info(f"数据验证完成: {len(valid_data)}/{len(data_list)} 条有效")
    return valid_data

def process_data_with_config(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """根据配置处理数据"""
    try:
        data_config = config.get("data", {})
        data_location = data_config.get("data_location", "./data/upload")
        
        if not os.path.exists(data_location):
            raise FileNotFoundError(f"数据目录不存在: {data_location}")
        
        logger.info(f"开始处理数据目录: {data_location}")
        
        # 导入数据处理模块
        from dataBuilder.data import data_process
        
        # 获取配置参数
        system_config = config.get("system", {})
        chunking_config = config.get("chunking", {})
        multimodal_config = config.get("multimodal", {})
        
        chunking_params = {
            "chunk_length": chunking_config.get("chunk_length", 512),
            "ppl_threshold": chunking_config.get("ppl_threshold", 0.3),
            "confidence_threshold": chunking_config.get("confidence_threshold", 0.7),
            "similarity_threshold": chunking_config.get("similarity_threshold", 0.8),
            "overlap": chunking_config.get("overlap", 50),
            "language": chunking_config.get("language", "zh")
        }
        
        # 处理数据
        start_time = time.time()
        data_list = data_process(
            data_location=data_location,
            url_split=system_config.get("url_split", False),
            chunking_strategy=chunking_config.get("strategy", "traditional"),
            chunking_params=chunking_params,
            enable_multimodal=multimodal_config.get("enable_image", False)
        )
        end_time = time.time()
        
        logger.info(f"数据处理完成，耗时: {end_time - start_time:.2f}秒")
        logger.info(f"处理结果: {len(data_list) if data_list else 0} 条数据")
        
        return data_list or []
        
    except Exception as e:
        logger.error(f"数据处理失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return []

def fast_vector_database_build(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    快速向量数据库构建
    使用预连接架构，避免连接阻塞
    """
    try:
        logger.info("=" * 60)
        logger.info("开始快速向量数据库构建")
        logger.info("=" * 60)
        
        # 1. 检查连接状态
        if not is_initialized():
            return {
                "status": "error",
                "msg": "系统未初始化，请先调用startup_initialize()"
            }
        
        logger.info("✅ 连接状态检查通过")
        
        # 2. 处理数据
        logger.info("开始数据处理...")
        data_list = process_data_with_config(config)
        
        if not data_list:
            return {
                "status": "error",
                "msg": "数据处理失败或无有效数据"
            }
        
        # 3. 验证数据
        valid_data = validate_data_list(data_list)
        if not valid_data:
            return {
                "status": "error",
                "msg": "没有有效的数据可以插入"
            }
        
        logger.info(f"有效数据量: {len(valid_data)}")
        
        # 4. 快速插入数据
        logger.info("开始快速数据插入...")
        
        # 获取配置
        milvus_config = config.get("milvus", {})
        system_config = config.get("system", {})
        
        collection_name = milvus_config.get("collection_name", "default_collection")
        index_name = milvus_config.get("index_name", "IVF_FLAT")
        replica_num = milvus_config.get("replica_num", 1)
        url_split = system_config.get("url_split", False)
        insert_mode = system_config.get("insert_mode", "覆盖（删除原有数据）")
        
        # 构建索引参数
        from IndexParamBuilder.indexparam import indexParamBuilder
        index_param = indexParamBuilder(milvus_config.get("index_device", "cpu"), index_name)
        
        # 使用快速插入
        from milvusBuilder.fast_insert import fast_milvus_insert
        
        host = milvus_config.get("host", "127.0.0.1")
        port = int(milvus_config.get("port", "19530"))
        
        start_time = time.time()
        result = fast_milvus_insert(
            collection_name=collection_name,
            index_param=index_param,
            replica_num=replica_num,
            data_list=valid_data,
            url_split=url_split,
            insert_mode=insert_mode,
            milvus_host=host,
            milvus_port=port
        )
        end_time = time.time()
        
        logger.info(f"数据插入完成，耗时: {end_time - start_time:.2f}秒")
        
        if result.get("status") == "success":
            final_result = {
                "status": "success",
                "message": "快速向量数据库构建完成",
                "processed_files": len(valid_data),
                "insert_result": result,
                "total_time": end_time - start_time
            }
            logger.info("🎉 快速向量数据库构建成功!")
            return final_result
        else:
            logger.error(f"数据插入失败: {result}")
            return {
                "status": "error",
                "msg": f"数据插入失败: {result.get('msg', '未知错误')}"
            }
        
    except Exception as e:
        logger.error(f"快速向量数据库构建失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return {
            "status": "error",
            "msg": str(e)
        }

def fast_vector_database_build_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    从配置构建快速向量数据库
    这是新的入口函数，替代原来的Cre_VectorDataBaseStart_from_config
    """
    try:
        logger.info("使用新架构进行向量数据库构建")
        return fast_vector_database_build(config)
        
    except Exception as e:
        logger.error(f"配置解析失败: {e}")
        return {
            "status": "error",
            "msg": str(e)
        }

def get_connection_status() -> Dict[str, Any]:
    """获取连接状态"""
    try:
        from milvusBuilder.persistent_connection import check_milvus_connection_status
        
        initializer = get_initializer()
        init_status = initializer.get_status()
        milvus_status = check_milvus_connection_status()
        
        return {
            "initializer_status": init_status,
            "milvus_connection": milvus_status,
            "overall_ready": is_initialized()
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "overall_ready": False
        }