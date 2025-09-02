"""
快速数据插入模块
使用预建立的持久化连接进行数据插入
"""

import logging
from typing import List, Dict, Any, Optional
from pymilvus import Collection, utility, FieldSchema, CollectionSchema, DataType
from .persistent_connection import get_milvus_connection

logger = logging.getLogger(__name__)

def fast_milvus_insert(
    collection_name: str,
    index_param: Dict[str, Any],
    replica_num: int,
    data_list: List[Dict[str, Any]],
    url_split: bool = False,
    insert_mode: str = "覆盖（删除原有数据）",
    milvus_host: str = None,
    milvus_port: int = None
) -> Dict[str, Any]:
    """
    快速Milvus数据插入 - 使用预建立的连接避免阻塞
    """
    try:
        logger.info(f"🚀 开始快速数据插入: {collection_name}")
        logger.info(f"数据量: {len(data_list)}, 插入模式: {insert_mode}")
        
        # 如果提供了连接参数，更新连接
        if milvus_host and milvus_port:
            logger.info(f"🔄 更新Milvus连接: {milvus_host}:{milvus_port}")
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(__file__)))
            from simple_milvus import update_milvus_connection
            use_lite = milvus_host.endswith('.db')
            success = update_milvus_connection(milvus_host, milvus_port, use_lite)
            if not success:
                raise ConnectionError(f"无法连接到Milvus: {milvus_host}:{milvus_port}")
        
        # 获取连接别名
        connection_alias = get_milvus_connection()
        if not connection_alias:
            raise ConnectionError("无法获取Milvus连接别名，请检查连接状态")
        
        logger.info(f"✅ 使用预建立的连接: {connection_alias}")
        
        # 验证数据
        if not data_list:
            return {"status": "error", "msg": "数据列表为空"}
        
        # 检测向量维度
        embedding_dim = 1024  # 默认维度
        for data in data_list:
            if "embedding" in data and data["embedding"]:
                embedding_dim = len(data["embedding"])
                break
        
        logger.info(f"检测到向量维度: {embedding_dim}")
        
        # 定义集合字段
        if url_split:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
                FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=1024)
            ]
        else:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim)
            ]
        
        # 创建集合schema
        schema = CollectionSchema(fields, collection_name, enable_dynamic_field=True)
        
        # 处理集合
        if insert_mode == "覆盖（删除原有数据）":
            # 删除现有集合
            if utility.has_collection(collection_name, using=connection_alias):
                utility.drop_collection(collection_name, using=connection_alias)
                logger.info(f"已删除现有集合: {collection_name}")
            
            # 创建新集合
            collection = Collection(name=collection_name, schema=schema, using=connection_alias)
            logger.info(f"创建新集合: {collection_name}")
            
        else:  # 追加模式
            if utility.has_collection(collection_name, using=connection_alias):
                collection = Collection(name=collection_name, using=connection_alias)
                logger.info(f"使用现有集合: {collection_name}")
            else:
                collection = Collection(name=collection_name, schema=schema, using=connection_alias)
                logger.info(f"创建新集合: {collection_name}")
        
        # 准备插入数据
        logger.info("准备插入数据...")
        
        # 确保数据字段对齐
        for data in data_list:
            if url_split:
                if "url" not in data:
                    data["url"] = ""
            else:
                if "url" in data:
                    del data["url"]
        
        # 准备实体数据
        if url_split:
            entities = [
                [d["id"] for d in data_list],
                [d["content"] for d in data_list],
                [d["embedding"] for d in data_list],
                [d.get("url", "") for d in data_list]
            ]
        else:
            entities = [
                [d["id"] for d in data_list],
                [d["content"] for d in data_list],
                [d["embedding"] for d in data_list]
            ]
        
        # 插入数据
        logger.info("开始插入数据...")
        insert_result = collection.insert(entities)
        logger.info(f"数据插入完成，插入数量: {insert_result.insert_count}")
        
        # 刷新数据
        collection.flush()
        logger.info("数据刷新完成")
        
        # 创建索引
        logger.info("开始创建索引...")
        collection.create_index(field_name="embedding", index_params=index_param)
        logger.info("索引创建完成")
        
        # 加载集合
        logger.info("开始加载集合...")
        collection.load(replica_number=replica_num)
        logger.info("集合加载完成")
        
        result = {
            "status": "success",
            "msg": f"成功插入 {len(data_list)} 条数据",
            "insert_count": insert_result.insert_count,
            "collection_name": collection_name
        }
        
        logger.info(f"✅ 快速插入完成: {result}")
        return result
        
    except Exception as e:
        logger.error(f"❌ 快速插入失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        
        return {
            "status": "error",
            "msg": str(e),
            "collection_name": collection_name
        }

def check_collection_status(collection_name: str) -> Dict[str, Any]:
    """检查集合状态"""
    try:
        connection_alias = get_milvus_connection()
        if not connection_alias:
            return {"status": "error", "msg": "无连接"}
        
        if utility.has_collection(collection_name, using=connection_alias):
            collection = Collection(name=collection_name, using=connection_alias)
            return {
                "status": "success",
                "exists": True,
                "num_entities": collection.num_entities,
                "is_loaded": utility.load_state(collection_name, using=connection_alias)
            }
        else:
            return {
                "status": "success", 
                "exists": False
            }
            
    except Exception as e:
        return {"status": "error", "msg": str(e)}

def list_collections() -> List[str]:
    """列出所有集合"""
    try:
        connection_alias = get_milvus_connection()
        if not connection_alias:
            return []
        
        return utility.list_collections(using=connection_alias)
        
    except Exception as e:
        logger.error(f"列出集合失败: {e}")
        return []