# from System.init import  init_es, init_redis
from dataBuilder.data import data_process
from milvusBuilder.fast_insert import fast_milvus_insert
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
import os

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def Cre_VectorDataBaseStart_from_config(config):
    try:
        milvus_cfg = config.get("milvus", {})
        data_cfg = config.get("data", {})
        sys_cfg = config.get("system", {})
        chunking_cfg = config.get("chunking", {})
        multimodal_cfg = config.get("multimodal", {})

        log_event(f"开始向量化存储，配置: {config}")

        return Cre_VectorDataBaseStart(
            C_G_Choic=milvus_cfg.get("index_device", "cpu"),
            IP=milvus_cfg.get("host", "127.0.0.1"),
            Port=milvus_cfg.get("port", "19530"),
            VectorName=milvus_cfg.get("vector_name", "default"),
            CollectionName=milvus_cfg.get("collection_name", "Test_one"),
            IndexName=milvus_cfg.get("index_name", "IVF_FLAT"),
            ReplicaNum=milvus_cfg.get("replica_num", 1),

            Data_Location=data_cfg.get("data_location", "./data/upload"),
            url_split=sys_cfg.get("url_split", False),
            insert_mode=sys_cfg.get("insert_mode", "覆盖（删除原有数据）"),
            
            # 新增参数
            chunking_strategy=chunking_cfg.get("strategy", "traditional"),
            chunking_params={
                "chunk_length": chunking_cfg.get("chunk_length", 512),
                "ppl_threshold": chunking_cfg.get("ppl_threshold", 0.3),
                "confidence_threshold": chunking_cfg.get("confidence_threshold", 0.7),
                "similarity_threshold": chunking_cfg.get("similarity_threshold", 0.8),
                "overlap": chunking_cfg.get("overlap", 50),  # 新增缺失参数
                "language": chunking_cfg.get("language", "zh")
            },
            enable_multimodal=multimodal_cfg.get("enable_image", False)
        )
    except Exception as e:
        log_event(f"配置解析失败: {e}")
        raise

def Cre_VectorDataBaseStart(
    C_G_Choic, IP, Port, VectorName, CollectionName,
    IndexName, ReplicaNum, Data_Location, url_split, insert_mode,
    chunking_strategy="traditional", chunking_params=None, enable_multimodal=False
):
    """
    构建向量数据库并插入数据，参数全部由配置文件自动读取。
    """
    try:
        # 检查数据目录
        if not os.path.exists(Data_Location):
            raise FileNotFoundError(f"数据目录不存在: {Data_Location}")
        
        log_event(f"开始数据处理，目录: {Data_Location}")
        log_event(f"分块策略: {chunking_strategy}, 参数: {chunking_params}")
        
        # 数据处理加重试
        def process_data():
            return data_process(
                data_location=Data_Location,
                url_split=url_split,
                chunking_strategy=chunking_strategy,
                chunking_params=chunking_params or {
                    "chunk_length": 512,
                    "ppl_threshold": 0.3,
                    "confidence_threshold": 0.7,
                    "similarity_threshold": 0.8,
                    "overlap": 50,
                    "language": "zh"
                },
                enable_multimodal=enable_multimodal
            )
        
        dataList = retry_function(process_data)()
        
        if not dataList:
            raise ValueError("数据处理结果为空，请检查数据目录和文件格式")
        
        # log_event(f"数据处理完成，数据量：{len(dataList)}")
        
        # 验证数据质量
        valid_data = []
        for i, data in enumerate(dataList):
            if not isinstance(data, dict):
                log_event(f"跳过无效数据项 {i}: 不是字典格式")
                continue
            if "embedding" not in data or not data["embedding"]:
                log_event(f"跳过无效数据项 {i}: 缺少embedding")
                continue
            if "content" not in data or not data["content"]:
                log_event(f"跳过无效数据项 {i}: 缺少content")
                continue
            valid_data.append(data)
        
        if not valid_data:
            raise ValueError("没有有效的数据可以插入")
        
        log_event(f"有效数据量：{len(valid_data)}")

        # 构建索引参数
        log_event(f"构建索引参数: {IndexName}")
        indexParam = indexParamBuilder(C_G_Choic, IndexName)
        # log_event(f"索引参数详细: {indexParam}")
        log_event(f"第一条数据embedding长度: {len(valid_data[0]['embedding']) if valid_data else 0}")
        # 连接Milvus并插入数据
        # log_event(f"开始连接Milvus并插入数据,IP:{IP},Port:{Port},CollectionName:{CollectionName}")
        log_event(f"数据量: {len(valid_data)}, 插入模式: {insert_mode}")
        
        log_event(f"开始插入数据")
        status = fast_milvus_insert(
                collection_name=CollectionName,
                index_param=indexParam,
                replica_num=ReplicaNum,
                data_list=valid_data,
                url_split=url_split,
                insert_mode=insert_mode,
                milvus_host=IP,
                milvus_port=Port
        )
        log_event(f"Milvus连接状态:{status}")
        
        return {
            "status": "success",
            "message": "向量化存储完成",
            "processed_files": len(valid_data),
            "milvus_result": status
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        log_event(f"向量化存储失败: {e}\n详细错误: {error_details}")
        raise Exception(f"向量化存储失败: {str(e)}")

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
    url_split= data_cfg["url_split"]

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