import numpy as np

def reorder_clusters(clustered_results, query_vector, strategy="distance"):
    """
    对聚类结果和搜索结果进行重排序，返回排序后的集群列表。
    """
    query_vector = np.array(query_vector)
    
    def compute_cluster_center(cluster):
        embeddings = np.array([result["embedding"] for result in cluster])
        return np.mean(embeddings, axis=0)
    
    # 将集群转换为列表
    cluster_list = []
    for cluster_label, cluster_docs in clustered_results.items():
        # 首先对集群内的文档排序
        if strategy == "distance":
            sorted_docs = sorted(cluster_docs, key=lambda x: x["distance"])
        else:
            sorted_docs = cluster_docs  # 其他策略不改变集群内顺序
        
        cluster_list.append({
            "cluster_id": int(cluster_label),  # 转换为Python原生整数
            "documents": sorted_docs
        })
    
    # 对集群排序
    if strategy == "distance":
        # 按最小距离排序（距离越小越相关）
        cluster_list.sort(key=lambda c: min(d["distance"] for d in c["documents"]))
    
    elif strategy == "cluster_size":
        # 按集群大小排序（文档越多越相关）
        cluster_list.sort(key=lambda c: len(c["documents"]), reverse=True)
    
    elif strategy == "cluster_center":
        # 计算每个集群的中心
        cluster_centers = {}
        for cluster in cluster_list:
            cluster_centers[cluster["cluster_id"]] = compute_cluster_center(
                cluster["documents"]
            )
        
        # 按集群中心与查询向量的距离排序
        cluster_list.sort(
            key=lambda c: np.linalg.norm(
                cluster_centers[c["cluster_id"]] - query_vector
            )
        )
    
    else:
        raise ValueError(f"Unsupported reorder strategy: {strategy}")
    
    return cluster_list