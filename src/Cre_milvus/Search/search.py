from pymilvus import Collection
from Search.embedding import embedder  # 用 embedder 替换 EmbeddingGenerator
from System.monitor import log_event
from Search.milvusSer import milvus_search
from pymilvus import Collection, connections

def search(CollectionName, question, topK, url_split, host, port):
    """搜索函数 - 使用统一的连接别名"""
    try:
        # 获取统一的连接别名
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        # 由于我们使用默认连接，这里直接使用 "default" 作为连接标识
        connection_alias = "default"
        log_event(f"使用连接别名进行搜索: {connection_alias}")
        
        # 生成embedding
        embedding = embedder.get_embedding(question)
        log_event("embedding结果为：{}".format(embedding))
        
        # 执行搜索
        milvus_list = milvus_search(embedding, topK, host, port, CollectionName, connection_alias) 
        
        if not milvus_list:
            log_event("milvus_list结果为空")
        else:
            log_event(f"搜索返回 {len(milvus_list)} 条结果")
            
        return milvus_list
        
    except Exception as e:
        log_event(f"搜索失败: {e}")
        return []