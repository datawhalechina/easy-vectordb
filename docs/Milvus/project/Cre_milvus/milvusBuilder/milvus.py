from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import logging
import asyncio
import concurrent.futures
import time
import threading
import uuid
from .lazy_connection import get_lazy_connection

def test_milvus_connection(host, port, timeout=5):
    """测试Milvus连接是否可用"""
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, int(port)))
        sock.close()
        return result == 0
    except Exception as e:
        logging.error(f"连接测试失败: {e}")
        return False

_connection_lock = threading.Lock()

def cleanup_all_connections():
    """清理所有现有连接"""
    try:
        existing_connections = connections.list_connections()
        for conn_alias in existing_connections:
            try:
                connections.disconnect(conn_alias)
                logging.info(f"已断开连接: {conn_alias}")
            except Exception as e:
                logging.warning(f"断开连接 {conn_alias} 失败: {e}")
        time.sleep(1)  # 等待连接完全关闭
    except Exception as e:
        logging.warning(f"清理连接时出错: {e}")

def milvus_connect_with_retry(alias, host, port, max_retries=3, timeout=10):
    """带重试机制的Milvus连接，使用线程锁防止并发冲突"""
    with _connection_lock:  # 使用锁确保连接操作的原子性
        for attempt in range(max_retries):
            try:
                logging.info(f"尝试连接Milvus (第{attempt + 1}次): {host}:{port}")
                
                # 先测试网络连接
                if not test_milvus_connection(host, port, timeout=5):
                    raise ConnectionError(f"无法连接到 {host}:{port}")
                
                cleanup_all_connections()
                
                logging.info("尝试建立新连接")
                connections.connect(
                    alias=alias,
                    host=host,
                    port=int(port),
                    timeout=timeout
                )
                
                try:
                    collections = utility.list_collections()
                    server_version = utility.get_server_version()
                    logging.info(f"连接成功，服务器版本: {server_version}, 现有集合: {collections}")
                    return True
                except Exception as verify_error:
                    logging.error(f"连接验证失败: {verify_error}")
                    try:
                        connections.disconnect(alias)
                    except:
                        pass
                    raise verify_error
                
            except Exception as e:
                logging.warning(f"连接尝试 {attempt + 1} 失败: {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    logging.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    raise ConnectionError(f"经过 {max_retries} 次尝试后仍无法连接到Milvus: {e}")

async def milvus_connect_insert_async(CollectionName, IndexParam, ReplicaNum, dataList, url_split, Milvus_host, Milvus_port, insert_mode="覆盖（删除原有数据）"):
    """异步版本的Milvus连接和插入"""
    def sync_operation():
        return milvus_connect_insert(CollectionName, IndexParam, ReplicaNum, dataList, url_split, Milvus_host, Milvus_port, insert_mode)
    
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(executor, sync_operation)
    return result

def milvus_connect_insert(CollectionName, IndexParam, ReplicaNum, dataList, url_split, Milvus_host, Milvus_port,insert_mode="覆盖（删除原有数据）"):
    """使用简化的连接管理进行数据插入 - 避免阻塞"""
    
    try:
        logging.info("🚀 进入Insert模块：使用简化连接管理")
        logging.info(f"目标服务器: {Milvus_host}:{Milvus_port}")
        
        # 使用简化的连接管理
        from simple_milvus import get_milvus_connection, connect_milvus
        
        conn = get_milvus_connection()
        
        # 检查是否需要建立新连接
        use_lite = Milvus_host.endswith('.db')
        if not conn.is_connected() or conn.host != Milvus_host or conn.port != int(Milvus_port):
            logging.info(f"🔄 建立新连接: {Milvus_host}:{Milvus_port} (Lite: {use_lite})")
            success = connect_milvus(Milvus_host, int(Milvus_port), use_lite)
            if not success:
                raise ConnectionError(f"无法连接到Milvus: {Milvus_host}:{Milvus_port}")
        
        connection_alias = conn.get_connection_alias()
        if not connection_alias:
            raise ConnectionError("无法获取连接别名")
        
        logging.info(f"✅ 使用简化连接: {connection_alias}")
        
        logging.info(f"📋 现有集合: {utility.list_collections(using=connection_alias)}")
        if not dataList:
            logging.warning("数据列表为空，无法处理")
            return {"status": "fail", "msg": "数据列表为空"}
            
        logging.info(f"📊 准备处理 {len(dataList)} 条数据")
        embedding_dim = 1024  # 默认维度
        if dataList and len(dataList) > 0:
            first_embedding = dataList[0].get("embedding", [])
            if isinstance(first_embedding, list):  
                embedding_dim = len(first_embedding)
                logging.info(f"🔍 检测到向量维度: {embedding_dim}")
        
        collection_name = CollectionName
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
        
        # 创建带动态字段支持的 schema
        schema = CollectionSchema(fields, collection_name, enable_dynamic_field=True)
        
        # 添加动态字段支持验证
        if utility.has_collection(collection_name, using=connection_alias) and insert_mode != "覆盖（删除原有数据）":
            existing_schema = Collection(collection_name, using=connection_alias).schema
            if not existing_schema.enable_dynamic_field:
                raise ValueError("无法追加到不支持动态字段的集合，请先删除原有集合")
        
        # 根据插入模式决定是否删除原有 collection
        if insert_mode == "覆盖（删除原有数据）":
            if utility.has_collection(collection_name, using=connection_alias):
                utility.drop_collection(collection_name, using=connection_alias)
                logging.info(f"🗑️ 已删除现有集合: {collection_name}")
            collection = Collection(name=collection_name, schema=schema, using=connection_alias)
            logging.info(f"🆕 创建新集合: {collection_name}")
        else:  # 追加模式
            if utility.has_collection(collection_name, using=connection_alias):
                existing_col = Collection(collection_name, using=connection_alias)
                if existing_col.schema.fields != schema.fields:  # 精确比较字段
                    raise ValueError(f"集合{collection_name}已存在但字段不匹配")
                logging.info(f"📝 使用现有集合: {collection_name}")
            if not utility.has_collection(collection_name, using=connection_alias):
                collection = Collection(name=collection_name, schema=schema, using=connection_alias)
                logging.info(f"🆕 创建新集合: {collection_name}")
            else:
                collection = Collection(name=collection_name, using=connection_alias)
        
        # 确保数据字段对齐
        for data in dataList:
            if url_split:
                if "url" not in data or not isinstance(data["url"], str):
                    data["url"] = ""
            else:
                if "url" in data:
                    del data["url"]

        # 检查 embedding
        embeddings = [data["embedding"] for data in dataList if "embedding" in data and data["embedding"] is not None]
        if not embeddings:
            logging.error("❌ 数据中没有可用的embedding，无法入库")
            return {"status": "fail", "msg": "数据中没有可用的embedding，无法入库。"}
        
        # 准备插入数据 - 字段顺序必须与 schema 完全一致
        if url_split:
            # 字段顺序: id, content, embedding, url
            entities = [
                [d["id"] for d in dataList],
                [d["content"] for d in dataList],
                [d["embedding"] for d in dataList],
                [d.get("url", "") for d in dataList]
            ]
        else:
            # 字段顺序: id, content, embedding
            entities = [
                [d["id"] for d in dataList],
                [d["content"] for d in dataList],
                [d["embedding"] for d in dataList]
            ]
        
        # 插入数据到 Milvus
        logging.info("📥 开始插入数据...")
        insert_result = collection.insert(entities)
        
        # 刷新确保数据写入
        logging.info("💾 刷新数据...")
        collection.flush()
        
        # 创建索引
        logging.info("🔗 创建索引...")
        collection.create_index(field_name="embedding", index_params=IndexParam)
        
        # 加载集合到内存
        logging.info("🚀 加载集合到内存...")
        collection.load(replica_number=ReplicaNum)
        
        logging.info(f"✅ 成功插入 {insert_result.insert_count} 条数据")
        logging.info(f"🔧 索引参数: {IndexParam}")
        
        return {
            "status": "success",
            "msg": f"成功插入 {len(dataList)} 条数据",
            "insert_count": insert_result.insert_count
        }
        
    except Exception as e:
        logging.error(f"操作失败: {str(e)}", exc_info=True)
        logging.error(f"错误发生时数据列表长度: {len(dataList) if dataList else 0}")
        if 'dataList' in locals() and dataList:
            logging.error(f"错误发生时第一条数据: {dataList[0]}")
        
        # 在出现错误时，记录错误信息
        logging.error("Milvus操作出现错误，连接将自动清理")
            
        return {
            "status": "error",
            "msg": str(e)
        }