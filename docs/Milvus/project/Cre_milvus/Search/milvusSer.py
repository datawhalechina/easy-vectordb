from pymilvus import Collection, connections, utility
import logging as logger

def milvus_search(embedding, topK, host, port, collection_name, connection_alias=None):
    """
    使用指定的连接别名进行搜索
    如果没有提供connection_alias，则使用旧的连接方式（兼容性）
    """
    try:
        if connection_alias is None:
            # 兼容旧的连接方式
            connections.disconnect("default")
            connections.connect(alias="default", host=host, port=port)
            connection_alias = "default"
            logger.info(f"已连接到 Milvus: {host}:{port}")
        else:
            logger.info(f"使用现有连接 [{connection_alias}] 进行搜索")
        
        # 使用基础查询代替搜索
        expr = ""
        output_fields = ["id", "content","embedding","url"]
        
        # 获取所有ID (仅用于诊断)
        all_ids = Collection(collection_name, using=connection_alias).query(
            expr=expr,
            output_fields=["id"],
            limit=100
        )
        logger.info(f"集合中有 {len(all_ids)} 条记录")
        
        # 执行ANN搜索的替代方法
        results = Collection(collection_name, using=connection_alias).search(
            data=[embedding],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=topK,
            output_fields=output_fields,
            # 使用旧版API避免Hit对象问题
            _async=False,
            _callback=None
        )
        
        # 基础结果处理（避免使用Hit对象）
        search_results = []
        for result in results:
            for hit in result:
                item = {
                    "id": hit.id,
                    "content": hit.entity.get("content", ""),
                    "url": hit.entity.get("url", ""),
                    "distance": hit.distance,
                    "embedding": hit.entity.get("embedding", [])
                }
                search_results.append(item)
        
        logger.info(f"找到 {len(search_results)} 条搜索结果")
        return search_results
    
    except Exception as e:
        print(f"搜索失败: {str(e)}")        
        # 获取详细集合信息
        try:
            coll_info = {
                "exists": utility.has_collection(collection_name, using=connection_alias),
                "entities": Collection(collection_name, using=connection_alias).num_entities if utility.has_collection(collection_name, using=connection_alias) else 0
            }
            print(f"集合 {collection_name} 信息: {coll_info}")
            # 检查向量维度
            if coll_info["exists"]:
                schema = Collection(collection_name, using=connection_alias).schema
                for field in schema.fields:
                    if field.name == "embedding":
                        dim = field.params.get("dim", "未知")
                        print(f"集合 {collection_name} 的向量维度: {dim}")
        except Exception as diag_error:
            print(f"获取集合信息失败: {str(diag_error)}")
        return []