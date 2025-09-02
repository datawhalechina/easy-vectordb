import json
import numpy as np
from redisvl import index
from redisvl.query import VectorQuery
from System.eval import score_redis_recall

# 定义一个函数，用于归一化嵌入向量
# 为什么要归一化呢：
# 归一化嵌入向量可以确保所有向量具有相同的长度，从而使得它们在计算相似度时具有可比性。这样可以避免向量长度差异对相似度计算结果的影响，提高相似度计算的准确性，并显著提高计算效率，可以去看看计算式。
def normalize_embedding(embedding):
    # 将嵌入向量转换为numpy数组，数据类型为float32
    arr = np.array(embedding, dtype=np.float32)
    # 计算数组的范数
    norm = np.linalg.norm(arr)
    # 如果范数大于0，则将数组除以范数，并转换为列表返回；否则直接将数组转换为列表返回
    return (arr / norm).tolist() if norm > 0 else arr.tolist()

def cache_embedding_result(redis_client, result_list,url_split):
    """
    result_list: 列表，每个元素为dict，包含id、content、embedding等字段
    """
    # 创建Index对象，用于存储embedding结果
    idx = index(redis_client, name="qa_embedding")
    # 遍历result_list中的每个元素
    for item in result_list:
        # 创建doc字典，用于存储每个元素的id、content、embedding和distance字段
        if url_split:
            doc = {
                "id": item.get("id"),
                "content": item.get("content", ""),
                "embedding": normalize_embedding(item.get("embedding", [])),
                "distance": item.get("distance", 0),
                "url": item.get("url", "")
            }
        else:
            doc = {
                "id": item.get("id"),
                "content": item.get("content", ""),
                "embedding": normalize_embedding(item.get("embedding", [])),
                "distance": item.get("distance", 0),
            }
        # 将doc字典添加到Index对象中
        idx.add_document(doc)
def search_similar_embedding(redis_client, embedding, top_k=1, score_threshold=0.80):
    idx = index(redis_client, name="qa_embedding")
    # 做归一化处理
    embedding = normalize_embedding(embedding)
    q = VectorQuery("embedding", embedding, top_k=top_k)
    # 虽然已经进行了top_k,但是还是要进行再一次的审核过滤
    docs = idx.query(q)
    resList = score_redis_recall(embedding,docs)
    # resList里面有score字段
    if resList:
        # 可以根据score排序或筛选
        scored_results = [r for r in resList if r["score"] > score_threshold]
        return scored_results
    return None