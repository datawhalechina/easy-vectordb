from pymilvus import Collection, connections, utility, MilvusClient, DataType
import logging as logger
import os
from Search.embedding import embedder as _embedder

def milvus_search(embedding, topK, host, port, collection_name, connection_alias=None):
    try:
        # 使用 MilvusClient 连接（优先无凭证，失败再尝试默认token）
        client = None
        uri = f"http://{host}:{port}"
        token_env = os.environ.get("MILVUS_TOKEN")
        try:
            client = MilvusClient(uri=uri)
        except Exception:
            client = MilvusClient(uri=uri, token=token_env or "root:Milvus")
        
        logger.info(f"✅ 使用 MilvusClient 连接成功: {host}:{port}")
        print(f"✅ 使用 MilvusClient 连接成功: {host}:{port}")
        
        # 检查集合是否存在，如果不存在则创建（根据实际 embedding 维度创建）
        if not client.has_collection(collection_name):
            logger.warning(f"⚠️ 集合 {collection_name} 不存在，正在创建...")
            print(f"⚠️ 集合 {collection_name} 不存在，正在创建...")
            
            # 创建 schema
            schema = client.create_schema(
                auto_id=False,
                enable_dynamic_field=True,
            )
            
            # 计算向量维度：优先用当前查询向量，否则从模型获取
            try:
                vector_dim = len(embedding) if embedding else _embedder.get_embedding_dimension()
            except Exception:
                vector_dim = 1024
            
            # 添加字段到 schema（根据实际需求调整字段）
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=vector_dim)
            schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="url", datatype=DataType.VARCHAR, max_length=1024)
            
            # 创建集合
            client.create_collection(
                collection_name=collection_name,
                schema=schema
            )
            
            # 为向量字段创建索引
            index_params = client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128}
            )
            client.create_index(
                collection_name=collection_name,
                index_params=index_params
            )
            
            logger.info(f"✅ 集合 {collection_name} 创建成功并已创建索引")
            print(f"✅ 集合 {collection_name} 创建成功并已创建索引")
        
        # 加载集合以进行搜索
        try:
            client.load_collection(collection_name=collection_name)
            logger.info(f"✅ 集合 {collection_name} 加载成功")
            print(f"✅ 集合 {collection_name} 加载成功")
        except Exception as load_error:
            logger.warning(f"⚠️ 集合 {collection_name} 加载失败: {str(load_error)}")
            print(f"⚠️ 集合 {collection_name} 加载失败: {str(load_error)}")
        
        expr = ""
        output_fields = ["id", "content", "embedding", "url"]
        
        # 使用 MilvusClient 进行搜索
        search_results = client.search(
            collection_name=collection_name,
            data=[embedding],
            anns_field="embedding",
            search_params={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=topK,
            output_fields=output_fields
        )
        
        # 格式化搜索结果
        formatted_results = []
        for result in search_results:
            for hit in result:
                item = {
                    "id": hit.get("id", ""),
                    "content": hit.get("content", ""),
                    "url": hit.get("url", ""),
                    "distance": hit.get("distance", 0),
                    "embedding": hit.get("embedding", [])
                }
                formatted_results.append(item)
        
        logger.info(f"找到 {len(formatted_results)} 条搜索结果")
        return formatted_results
    
    except Exception as e:
        print(f"搜索失败: {str(e)}")
        logger.error(f"搜索失败: {str(e)}")
        return []