from pymilvus import Collection
from Search.embedding import embedder  # 用 embedder 替换 EmbeddingGenerator
from System.monitor import log_event
from Search.milvusSer import milvus_search
from pymilvus import Collection, connections
from milvusBuilder.lazy_connection import get_lazy_connection

def search(CollectionName, question, topK, url_split, host, port):
    """使用延迟连接进行搜索"""
    lazy_connection = get_lazy_connection()
    
    try:
        # 使用延迟连接的上下文管理器
        with lazy_connection.get_connection(host, port) as connection_alias:
            results = None
            def _do_search():
                collection = Collection(CollectionName, using=connection_alias)
                embedding = embedder.get_embedding(question)  # 用 embedder
                log_event("embedding结果为：{}".format(embedding))
                milvus_list = milvus_search(embedding, topK, host, port, CollectionName, connection_alias) 
                if not milvus_list:
                    log_event("milvus_list结果为空")
                return milvus_list

            results = _do_search()
            return results
    except Exception as e:
        log_event(f"搜索失败: {e}")
        return []