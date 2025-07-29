from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import logging

def milvus_connect_insert(CollectionName, IndexParam, ReplicaNum, dataList, url_split, Milvus_host, Milvus_port,insert_mode="覆盖（删除原有数据）"):
    try:
        # 连接到 Milvus
        connections.connect(
            alias="default", 
            host=Milvus_host, 
            port=Milvus_port
        )
        
        # 动态获取向量维度
        embedding_dim = 1024  # 默认维度
        if dataList and len(dataList) > 0:
            first_embedding = dataList[0].get("embedding", [])
            if isinstance(first_embedding, list):  # 添加类型检查
                embedding_dim = len(first_embedding)
                logging.info(f"检测到向量维度: {embedding_dim}")
        
        # 检查并创建 collection
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
        if utility.has_collection(collection_name) and insert_mode != "覆盖（删除原有数据）":
            existing_schema = Collection(collection_name).schema
            if not existing_schema.enable_dynamic_field:
                raise ValueError("无法追加到不支持动态字段的集合，请先删除原有集合")
        
        # 根据插入模式决定是否删除原有 collection
        if insert_mode == "覆盖（删除原有数据）":
            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)
            collection = Collection(name=collection_name, schema=schema)
        else:  # 追加模式
            if not utility.has_collection(collection_name):
                collection = Collection(name=collection_name, schema=schema)
            else:
                collection = Collection(name=collection_name)
        
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
            logging.error("No embeddings found in dataList，无法入库。")
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
        insert_result = collection.insert(entities)
        
        # 刷新确保数据写入
        collection.flush()
        
        # 创建索引
        collection.create_index(field_name="embedding", index_params=IndexParam)
        
        # 加载集合到内存
        collection.load(replica_number=ReplicaNum)
        
        return {
            "status": "success",
            "msg": f"成功插入 {len(dataList)} 条数据",
            "insert_count": insert_result.insert_count
        }
        
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        return {
            "status": "error",
            "msg": str(e)
        }