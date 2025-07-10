from pymilvus import Collection, connections
import logging as logger

def milvus_search(embedding, topK, host, port, collection_name):
    try:
                
        # 创建新连接
        connections.disconnect("default")
        connections.connect(alias="default", host=host, port=port)
        logger.info(f"已连接到 Milvus: {host}:{port}")
        
        
        # 使用基础查询代替搜索
        expr = ""
        output_fields = ["id", "content","embedding","url"]
        
        # 获取所有ID (仅用于诊断)
        all_ids = Collection(collection_name).query(
            expr=expr,
            output_fields=["id"],
            limit=100
        )
        logger.info(f"集合中有 {len(all_ids)} 条记录")
        
        # 执行ANN搜索的替代方法
        results = Collection(collection_name).search(
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
            for i in range(len(result.ids)):
                item = {
                    "id": result.ids[i],
                    "distance": result.distances[i],
                }
                
                # 手动获取内容字段
                entity_data = Collection(collection_name).query(
                    expr=f"id == {result.ids[i]}",
                    output_fields=output_fields
                )
                
                if entity_data:
                    item["content"] = entity_data[0].get("content", "")
                    item["embedding"] = entity_data[0].get("embedding", None)
                    item["url"] = entity_data[0].get("url", "")
                else:
                    item["content"] = ""
                    item["embedding"] = None
                    item["url"] = ""
                
                
                search_results.append(item)
        
        logger.info(f"找到 {len(search_results)} 条搜索结果")
        return search_results
    
    except Exception as e:
        print(f"搜索失败: {str(e)}")        
        # 获取详细集合信息
        try:
            from pymilvus import utility
            coll_info = {
                "exists": utility.has_collection(collection_name),
                "entities": Collection(collection_name).num_entities if utility.has_collection(collection_name) else 0
            }
            print(f"集合 {collection_name} 信息: {coll_info}")
            # 检查向量维度
            if coll_info["exists"]:
                schema = Collection(collection_name).schema
                for field in schema.fields:
                    if field.name == "embedding":
                        dim = field.params.get("dim", "未知")
                        print(f"集合 {collection_name} 的向量维度: {dim}")
        except Exception as diag_error:
            print(f"获取集合信息失败: {str(diag_error)}")
        return []