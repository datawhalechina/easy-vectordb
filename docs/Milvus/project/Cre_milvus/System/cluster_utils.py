from sklearn.cluster import KMeans
import numpy as np

def cluster_embeddings(embeddings, n_clusters=10):
    """
    对向量做KMeans聚类。仅分析/召回后用，不入库存。
    """
    if not embeddings:
        return []
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(np.array(embeddings))
    return labels