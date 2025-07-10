from pymilvus import Collection
from Search.embedding import embedder  # 用 embedder 替换 EmbeddingGenerator
from System.monitor import log_event
from Search.milvusSer import milvus_search
from pymilvus import Collection, connections

def search(CollectionName, question, topK, url_split,host, port):
    if not connections.has_connection("default"):
        connections.connect(alias="default", host=host, port=port)
        
    results = None
    def _do_search():
        collection = Collection(CollectionName)
        embedding = embedder.get_embedding(question)  # 用 embedder
        log_event("embedding结果为：{}".format(embedding))
        milvus_list = milvus_search(embedding, topK,host, port, CollectionName) 
        if not milvus_list:
            log_event("milvus_list结果为空")
        return milvus_list

    results = _do_search()
    return results