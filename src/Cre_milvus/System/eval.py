import numpy as np
from pymilvus import Collection

def evaluate_data_quality(txt):
    """
    简单的向量质量评估函数。（一条文本数据）的质量如何，如果质量高就返回True，
    否则返回False。
    """
    return True

def insert_with_quality_check(collection, dataList):
    high_quality = []
    low_quality = []

    for data in dataList:
        if evaluate_data_quality(data["content"]):
            high_quality.append(data)
        else:
            low_quality.append(data)

    status1 = collection.insert(high_quality)
    # 低质量数据插入到另一个集合中
    # 由于我们使用默认连接，这里直接使用 "default" 作为连接标识
    connection_alias = "default"
    
    if connection_alias:
        logger.info(f"创建Collection对象: name=low_quality_data, using={connection_alias}")
        print(f"创建Collection对象: name=low_quality_data, using={connection_alias}")
        
        # 验证连接是否有效
        try:
            utility.list_collections(using=connection_alias)
            logger.info(f"✅ 连接验证通过: {connection_alias}")
            print(f"✅ 连接验证通过: {connection_alias}")
        except Exception as e:
            logger.error(f"❌ 连接验证失败: {connection_alias}, 错误: {e}")
            print(f"❌ 连接验证失败: {connection_alias}, 错误: {e}")
        
        low_quality_collection = Collection("low_quality_data", using=connection_alias)
        status2 = low_quality_collection.insert(low_quality)
    else:
        logger.error("无法获取Milvus连接")
        status2 = None
    return status1, status2


def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    v1 = np.array(vec1, dtype=np.float32)
    v2 = np.array(vec2, dtype=np.float32)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def score_redis_recall(query_embedding, recall_results, score_key="embedding"):
    """
    对redis召回的每条数据，计算与query_embedding的相似度分数。
    :param query_embedding: 查询的embedding向量
    :param recall_results: redis召回的结果（list，每个元素为dict，需包含embedding字段）
    :param score_key: 召回结果中embedding字段名
    :return: 带有score字段的结果列表
    """
    scored = []
    for item in recall_results:
        item_embedding = item.get(score_key)
        if item_embedding is not None:
            score = cosine_similarity(query_embedding, item_embedding)
            item = dict(item) 
            item["score"] = score
            scored.append(item)
    return scored