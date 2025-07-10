from System.init import init_milvus, init_es, init_redis
from dataBuilder.data import data_process
from milvusBuilder.milvus import milvus_connect_insert
from IndexParamBuilder.indexparam import indexParamBuilder
from reorder.reo_clu import reorder_clusters

from Search.search import search
from System.monitor import log_event
from System.Retry import retry_function
from Search.embedding import embedder
import hdbscan
from sklearn.cluster import KMeans
import numpy as np
import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config = load_config()

def Cre_VectorDataBaseStart_from_config(config):
    milvus_cfg = config["milvus"]
    data_cfg = config["data"]
    sys_cfg = config["system"]

    return Cre_VectorDataBaseStart(
        C_G_Choic=milvus_cfg["index_device"],
        IP=milvus_cfg["host"],
        Port=milvus_cfg["port"],
        VectorName=milvus_cfg["vector_name"],
        CollectionName=milvus_cfg["collection_name"],
        IndexName=milvus_cfg["index_name"],
        ReplicaNum=milvus_cfg["replica_num"],

        Data_Location=data_cfg["data_location"],
        url_split=sys_cfg["url_split"],
        insert_mode=sys_cfg["insert_mode"]
    )

def Cre_VectorDataBaseStart(
    C_G_Choic, IP, Port, VectorName, CollectionName,
    IndexName, ReplicaNum, Data_Location, url_split,insert_mode
):
    """
    构建向量数据库并插入数据，参数全部由配置文件自动读取。
    """
    # 初始化连接
    init_milvus(VectorName, IP, Port)
    log_event("开始数据处理")
    # 数据处理加重试
    dataList = retry_function(
        lambda: data_process(
            data_location=Data_Location,
            url_split=url_split
        )
    )()
    log_event(f"数据处理完成，数据量：{len(dataList)}")

    # 构建索引参数
    indexParam = indexParamBuilder(C_G_Choic, IndexName)

    # 连接Milvus并插入数据
    log_event("开始连接Milvus并插入数据")
    def milvus_insert():
        # 数据质量评估与分流插入
        status = milvus_connect_insert(
           CollectionName, indexParam, ReplicaNum, dataList, url_split,IP, Port,insert_mode
        )
    
        return status
    
    milvus_insert()
    log_event("Milvus插入流程完成")

def Cre_Search(config, question):
    """
    从配置文件读取参数，执行检索、聚类和重排序。
    """
    # 读取配置
    milvus_cfg = config["milvus"]
    search_cfg = config.get("search", {})
    CollectionName = milvus_cfg["collection_name"]
    host = milvus_cfg["host"]
    port = milvus_cfg["port"]
    topK = search_cfg.get("topK", 10)
    ColChoice = search_cfg.get("col_choice", "hdbscan")
    reorder_strategy = search_cfg.get("reorder_strategy", "distance")

    log_event(f"开始检索: {question}")
    
    data_cfg = config["system"]
    url_split= data_cfg["url_split"],

    responseList = search(CollectionName, question, topK,url_split,host, port)
    print(f"Search results: {len(responseList)}")
    if not responseList:
        return {"message": "No results found", "clusters": []}

    # 提取搜索结果中的向量和 ID
    embeddings = [result["embedding"] for result in responseList]
    ids = [result["id"] for result in responseList]

    # 转换为 NumPy 数组
    embeddings = np.array(embeddings)

# 下面是进行搜索结果重排序的部分，对于召回的大批量的数据进行聚类，和重排序
    # 根据选择的聚类算法进行聚类
    if ColChoice.lower() == "hdbscan":
        clusterer = hdbscan.HDBSCAN(min_samples=3, min_cluster_size=2)
        labels = clusterer.fit_predict(embeddings)
    elif ColChoice.lower() == "kmeans":
        k = min(len(embeddings), 5)  # 设置聚类数量，最多为5
        clusterer = KMeans(n_clusters=k, random_state=42)
        labels = clusterer.fit_predict(embeddings)
    else:
        raise ValueError(f"Unsupported clustering algorithm: {ColChoice}")
    
    labels = labels.astype(int).tolist()  # 转换为Python整数列表
    
    # 将聚类结果
    clustered_results = {}
    for idx, label in enumerate(labels):
        py_label = int(label)  # 转换为Python整数
        
        if py_label not in clustered_results:
            clustered_results[py_label] = []
        
        # 创建结果项
        result_item = {
            "id": ids[idx],
            "embedding": embeddings[idx].tolist(),
            "content": responseList[idx]["content"],
            "distance": responseList[idx]["distance"]
        }
        
        # 如果存在URL字段，则添加它
        if "url" in responseList[idx]:
            result_item["url"] = responseList[idx]["url"]
        
        clustered_results[py_label].append(result_item)
    
    # 对聚类结果进行重排序
    query_vector = embedder.get_embedding(question) 
    sorted_clusters = reorder_clusters(clustered_results, query_vector, strategy=reorder_strategy)
    log_event(f"检索完成，返回结果数：{len(sorted_clusters)}")
    
    final_clusters = []
    for cluster in sorted_clusters:
        # 将cluster_id转换为整数
        cluster_id = int(cluster["cluster_id"])
        
        # 创建新的集群表示
        final_clusters.append({
            "cluster_id": cluster_id,
            "documents": cluster["documents"]
        })
    
    # 返回重排序后的结果
    return {
        "message": "Search, clustering, and reordering completed",
        "clusters": sorted_clusters  # 已经是列表结构
    }