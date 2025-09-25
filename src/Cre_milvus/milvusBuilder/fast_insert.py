"""
快速数据插入模块
使用默认连接管理，避免重复/冲突的连接管理。
"""

import logging
from System.monitor import log_event
from typing import List, Dict, Any
from pymilvus import Collection, utility, FieldSchema, CollectionSchema, DataType, connections
import time
from start_simple import connect_milvus
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('back.log', encoding='utf-8')
    ]
)
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
    快速Milvus数据插入 - 使用连接管理器确保连接稳定性
    """
    try:
        log_event(f"🚀 开始快速数据插入: {collection_name}")
        log_event(f"数据量: {len(data_list)}, 插入模式: {insert_mode},milvus_host:{milvus_host},milvus_port:{milvus_port}")

        # 直接连接 Milvus，只用默认 alias
        # connections.connect(host=milvus_host, port=int(milvus_port) if milvus_port is not None else None)
        # connections.connect()
        connection_alias = "default"
        log_event("开始检查集合是否存在")
        # 检查集合是否存在
        # if utility.has_collection(collection_name):
        #     if insert_mode in ("overwrite", "覆盖", "覆盖（删除原有数据）"):
        #         log_event(f"删除现有集合: {collection_name}")
        #         utility.drop_collection(collection_name)
        #         log_event(f"已删除现有集合: {collection_name}")
        #     else:
        #         log_event(f"使用现有集合: {collection_name}")

        # 检测向量维度
        log_event("向量维度检测")
        embedding_dim = 1024
        for data in data_list:
            if "embedding" in data and data["embedding"]:
                embedding_dim = len(data["embedding"])
                break

        # 定义集合字段
        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim)
        ]
        if url_split:
            fields.append(FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=1024))

        schema = CollectionSchema(fields, enable_dynamic_field=True)

        # 创建集合，启用 mmap
        log_event(f"开始创建集合，是否存在milvus连接{connections.has_connection(alias="default")}")
        connections.connect("default")
        if not utility.has_collection(collection_name):
            log_event(f"创建新集合: {collection_name}")
            collection = Collection(
                name=collection_name,
                schema=schema,
                properties={"mmap_enabled": True},
                timeout=30
            )
            log_event(f"创建新集合完成: {collection_name}")
        else:
            collection = Collection(name=collection_name)

        # 数据准备
        for data in data_list:
            if url_split and "url" not in data:
                data["url"] = ""
            elif not url_split and "url" in data:
                del data["url"]

        if url_split:
            entities = [
                [d["id"] for d in data_list],
                [d["content"] for d in data_list],
                [d["embedding"] for d in data_list],
                [d["url"] for d in data_list]
            ]
        else:
            entities = [
                [d["id"] for d in data_list],
                [d["content"] for d in data_list],
                [d["embedding"] for d in data_list]
            ]

        log_event("开始插入数据...")
        insert_result = collection.insert(entities, timeout=30)
        log_event(f"数据插入完成，插入数量: {insert_result.insert_count}")

        collection.flush()
        log_event("数据刷新完成")

        # 创建索引
        log_event("开始创建索引...")
        print("开始创建索引")
        collection.create_index(field_name="embedding", index_params=index_param, timeout=60)
        
        # 关键修复：等待索引构建完成
        max_wait_time = 300  # 5分钟超时
        wait_interval = 2    # 每2秒检查一次
        
        # 等待索引构建完成
        start_time = time.time()
        index_built = False
        while time.time() - start_time < max_wait_time:
            index_info = collection.index()
            if index_info:
                log_event(f"索引构建进度: {index_info}")
                index_built = True
                break
            log_event("等待索引构建...")
            time.sleep(wait_interval)
        
        if not index_built:
            raise Exception("索引构建超时")
        
        log_event("索引创建完成")

        # 加载集合
        log_event("开始加载集合...")
        print("开始加载集合")
        collection.load(replica_number=replica_num)
        
        # 等待集合加载完成（默认连接）
        utility.wait_for_loading_complete(collection_name, using=connection_alias, timeout=300)
        log_event("集合加载完成确认")
        
        # 验证集合状态
        load_state = utility.load_state(collection_name, using=connection_alias)
        if load_state != "Loaded":
            raise Exception(f"集合加载失败，当前状态: {load_state}")
        
        log_event(f"集合状态确认: {load_state}")

        return {
            "status": "success",
            "msg": f"成功插入 {len(data_list)} 条数据",
            "insert_count": insert_result.insert_count,
            "collection_name": collection_name
        }

    except Exception as e:
        logger.error(f"❌ 快速插入失败: {e}")
        # 记录更详细的错误信息
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return {
            "status": "error",
            "msg": str(e),
            "collection_name": collection_name
        }

def _perform_insert_with_connection(
    connection_alias: str,
    collection_name: str,
    index_param: Dict[str, Any],
    replica_num: int,
    data_list: List[Dict[str, Any]],
    url_split: bool,
    insert_mode: str
) -> Dict[str, Any]:
    """
    使用指定连接执行实际的插入操作
    """
    try:
        # 验证数据
        if not data_list:
            return {"status": "error", "msg": "数据列表为空"}
        
        # 检测向量维度
        embedding_dim = 1024  # 默认维度
        for data in data_list:
            if "embedding" in data and data["embedding"]:
                embedding_dim = len(data["embedding"])
                break
        
        log_event(f"检测到向量维度: {embedding_dim}")
        
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
        schema = CollectionSchema(fields, enable_dynamic_field=True)    
        # 判断集合是否存在的逻辑
        collection_exists = utility.has_collection(collection_name, using=connection_alias)
        if insert_mode in ("overwrite", "覆盖", "覆盖（删除原有数据）"):
            if collection_exists:
                log_event(f"删除现有集合: {collection_name}")
                utility.drop_collection(collection_name, using=connection_alias)
                log_event(f"已删除现有集合: {collection_name}")
            
            # 无论集合之前是否存在，overwrite模式都要创建新集合
            log_event(f"创建新集合: {collection_name}")
            print(f"创建新集合: {collection_name}")
            print(f"使用的连接别名: {connection_alias}")
            log_event(f"使用的连接别名: {connection_alias}")
            
            # 验证连接别名是否有效
            # 强化连接检测（新增10秒超时）
            try:
                connections.get_connection(connection_alias).wait_for_connected(timeout=10)
                utility.list_collections(using=connection_alias)
                log_event(f"✅ 连接别名验证通过: {connection_alias}")
            except Exception as e:
                logger.error(f"❌ 连接检测失败: {e}")
                raise RuntimeError(f"无法连接到Milvus服务: {e}")
            
            # 设置全局操作超时（新增）
            connections.set_timeout(15, using=connection_alias)
            collection = Collection(
                name=collection_name,
                schema=schema,
                using=connection_alias,
                timeout=15
            )
        
            log_event(f"创建新集合完成: {collection_name}")
            
            # 添加元数据同步验证
            max_retries = 3
            for attempt in range(max_retries):
                if utility.has_collection(collection_name, using=connection_alias):
                    log_event(f"✅ 集合元数据同步成功(第{attempt+1}次验证)")
                    break
                logger.warning(f"集合元数据未同步(第{attempt+1}次重试)")
                time.sleep(0.5)
            else:
                raise Exception("集合创建后元数据同步超时")
            
            print(f"创建新集合完成: {collection_name}")
        else:  # 追加模式
            if collection_exists:
                log_event(f"使用现有集合: {collection_name}")
                collection = Collection(name=collection_name, using=connection_alias)
                log_event(f"连接到现有集合: {collection_name}")
                existing_schema = collection.schema
                # 简单比较字段名和类型（可根据需要细化）
                if str(existing_schema) != str(schema):
                    raise ValueError(f"现有集合的 schema 与目标 schema 不一致，请使用覆盖模式或手动处理。")
            else:
                log_event(f"创建新集合: {collection_name}")
                collection = Collection(name=collection_name, schema=schema, using=connection_alias)
                log_event(f"创建新集合完成: {collection_name}")
        
        # 准备插入数据
        log_event("准备插入数据...")
        print("准备插入数据...")
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
        log_event("开始插入数据...")
        print("开始插入数据")
        insert_result = collection.insert(entities,timeout=30)
        log_event(f"数据插入完成，插入数量: {insert_result.insert_count}")
        
        # 刷新数据
        collection.flush()
        print("数据刷新完成")
        log_event("数据刷新完成")
        
        # 创建索引
        log_event("开始创建索引...")
        print("开始创建索引")
        collection.create_index(field_name="embedding", index_params=index_param, timeout=60)
        
        # 关键修复：等待索引构建完成
        max_wait_time = 300  # 5分钟超时
        wait_interval = 2    # 每2秒检查一次
        
        # 等待索引构建完成
        start_time = time.time()
        index_built = False
        while time.time() - start_time < max_wait_time:
            index_info = collection.index()
            if index_info:
                log_event(f"索引构建进度: {index_info}")
                index_built = True
                break
            log_event("等待索引构建...")
            time.sleep(wait_interval)
        
        if not index_built:
            raise Exception("索引构建超时")
        
        log_event("索引创建完成")

        # 加载集合
        log_event("开始加载集合...")
        print("开始加载集合")
        collection.load(replica_number=replica_num)
        
        # 等待集合加载完成
        utility.wait_for_loading_complete(collection_name, using=connection_alias, timeout=300)
        log_event("集合加载完成确认")
        
        # 验证集合状态
        load_state = utility.load_state(collection_name, using=connection_alias)
        if load_state != "Loaded":
            raise Exception(f"集合加载失败，当前状态: {load_state}")
        
        log_event(f"集合状态确认: {load_state}")
        
        result = {
            "status": "success",
            "msg": f"成功插入 {len(data_list)} 条数据",
            "insert_count": insert_result.insert_count,
            "collection_name": collection_name
        }
        
        log_event(f"✅ 快速插入完成: {result}")
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
        # 由于我们使用默认连接，这里直接使用 "default" 作为连接标识
        connection_alias = "default"
        from pymilvus import utility, Collection
        import threading
        
        # 带超时的集合检查
        def check_with_timeout():
            result = [None]
            exception = [None]
            
            def check():
                try:
                    exists = utility.has_collection(collection_name, using=connection_alias)
                    if exists:
                        collection = Collection(name=collection_name, using=connection_alias)
                        result[0] = {
                            "status": "success",
                            "exists": True,
                            "num_entities": collection.num_entities,
                            "is_loaded": utility.load_state(collection_name, using=connection_alias)
                        }
                    else:
                        result[0] = {
                            "status": "success", 
                            "exists": False
                        }
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=check)
            thread.daemon = True
            thread.start()
            thread.join(10)  # 10秒超时
            
            if thread.is_alive():
                return {"status": "error", "msg": "检查集合状态超时"}
            
            if exception[0]:
                raise exception[0]
            
            return result[0]
        
        return check_with_timeout()
            
    except Exception as e:
        return {"status": "error", "msg": str(e)}

def list_collections() -> List[str]:
    """列出所有集合"""
    try:
        # 由于我们使用默认连接，这里直接使用 "default" 作为连接标识
        connection_alias = "default"
        from pymilvus import utility
        
        return utility.list_collections(using=connection_alias)
        
    except Exception as e:
        logger.error(f"列出集合失败: {e}")
        return []