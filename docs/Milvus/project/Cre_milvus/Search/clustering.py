"""
搜索结果聚类服务模块

提供对搜索结果进行聚类分析和智能排序的功能
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """搜索结果数据结构"""
    id: str
    content: str
    url: Optional[str]
    distance: float
    embedding: List[float]
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class Cluster:
    """聚类结果数据结构"""
    cluster_id: int
    documents: List[SearchResult]
    centroid: Optional[List[float]]
    size: int
    avg_distance: float
    
    def __post_init__(self):
        self.size = len(self.documents)
        if self.documents:
            self.avg_distance = sum(doc.distance for doc in self.documents) / len(self.documents)

@dataclass
class ClusteredSearchResponse:
    """聚类搜索响应数据结构"""
    query: str
    clusters: List[Cluster]
    total_results: int
    clustering_method: str
    execution_time: float


class ClusteringService:
    """搜索结果聚类服务"""
    
    def __init__(self):
        self.hdbscan_available = self._check_hdbscan()
        self.sklearn_available = self._check_sklearn()
        
    def _check_hdbscan(self) -> bool:
        """检查HDBSCAN是否可用"""
        try:
            import hdbscan
            return True
        except ImportError:
            logger.warning("HDBSCAN不可用，将使用备用聚类方法")
            return False
    
    def _check_sklearn(self) -> bool:
        """检查sklearn是否可用"""
        try:
            from sklearn.cluster import KMeans
            return True
        except ImportError:
            logger.warning("sklearn不可用，将使用简单聚类方法")
            return False
    
    def cluster_search_results(self, results: List[Dict[str, Any]], method: str = "hdbscan") -> List[Cluster]:
        """
        对搜索结果进行聚类
        
        Args:
            results: 搜索结果列表
            method: 聚类方法 ("hdbscan", "kmeans", "distance")
            
        Returns:
            聚类结果列表
        """
        if not results:
            return []
        
        # 转换为SearchResult对象
        search_results = []
        embeddings = []
        
        for result in results:
            search_result = SearchResult(
                id=str(result.get('id', '')),
                content=result.get('content', ''),
                url=result.get('url'),
                distance=float(result.get('distance', 0.0)),
                embedding=result.get('embedding', []),
                metadata=result.get('metadata', {})
            )
            search_results.append(search_result)
            
            # 提取embedding用于聚类
            if search_result.embedding:
                embeddings.append(search_result.embedding)
        
        if not embeddings:
            # 如果没有embedding，基于距离进行简单分组
            return self._cluster_by_distance(search_results)
        
        embeddings = np.array(embeddings)
        
        # 根据方法选择聚类算法
        if method == "hdbscan" and self.hdbscan_available:
            labels = self._hdbscan_clustering(embeddings)
        elif method == "kmeans" and self.sklearn_available:
            labels = self._kmeans_clustering(embeddings)
        else:
            # 备用方法：基于距离的简单聚类
            return self._cluster_by_distance(search_results)
        
        # 构建聚类结果
        return self._build_clusters(search_results, labels, embeddings)
    
    def _hdbscan_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """使用HDBSCAN进行聚类"""
        try:
            import hdbscan
            
            # 动态调整参数
            n_samples = len(embeddings)
            min_cluster_size = max(2, min(5, n_samples // 4))
            min_samples = max(1, min_cluster_size - 1)
            
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='cosine'
            )
            
            labels = clusterer.fit_predict(embeddings)
            
            # 如果所有点都被标记为噪声，使用距离聚类
            if np.all(labels == -1):
                logger.warning("HDBSCAN将所有点标记为噪声，使用距离聚类")
                return self._distance_based_labels(embeddings)
            
            return labels
            
        except Exception as e:
            logger.error(f"HDBSCAN聚类失败: {e}")
            return self._distance_based_labels(embeddings)
    
    def _kmeans_clustering(self, embeddings: np.ndarray, n_clusters: Optional[int] = None) -> np.ndarray:
        """使用KMeans进行聚类"""
        try:
            from sklearn.cluster import KMeans
            
            n_samples = len(embeddings)
            if n_clusters is None:
                # 自动确定聚类数量
                n_clusters = min(max(2, n_samples // 5), 8)
            
            n_clusters = min(n_clusters, n_samples)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            
            return labels
            
        except Exception as e:
            logger.error(f"KMeans聚类失败: {e}")
            return self._distance_based_labels(embeddings)
    
    def _distance_based_labels(self, embeddings: np.ndarray) -> np.ndarray:
        """基于距离的简单聚类标签生成"""
        n_samples = len(embeddings)
        if n_samples <= 3:
            return np.zeros(n_samples)
        
        # 简单的基于距离的分组
        labels = np.zeros(n_samples)
        threshold = 0.3  # 距离阈值
        
        current_label = 0
        for i in range(n_samples):
            if labels[i] != 0:  # 已经分配标签
                continue
                
            labels[i] = current_label
            
            # 找到相似的点
            for j in range(i + 1, n_samples):
                if labels[j] == 0:  # 未分配标签
                    # 计算余弦相似度
                    similarity = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    if similarity > (1 - threshold):
                        labels[j] = current_label
            
            current_label += 1
        
        return labels
    
    def _cluster_by_distance(self, search_results: List[SearchResult]) -> List[Cluster]:
        """基于距离的简单聚类"""
        if not search_results:
            return []
        
        # 按距离排序
        sorted_results = sorted(search_results, key=lambda x: x.distance)
        
        clusters = []
        current_cluster = []
        distance_threshold = 0.1  # 距离阈值
        
        for result in sorted_results:
            if not current_cluster or abs(result.distance - current_cluster[-1].distance) <= distance_threshold:
                current_cluster.append(result)
            else:
                # 创建新聚类
                if current_cluster:
                    cluster = Cluster(
                        cluster_id=len(clusters),
                        documents=current_cluster.copy(),
                        centroid=None,
                        size=len(current_cluster),
                        avg_distance=sum(doc.distance for doc in current_cluster) / len(current_cluster)
                    )
                    clusters.append(cluster)
                current_cluster = [result]
        
        # 添加最后一个聚类
        if current_cluster:
            cluster = Cluster(
                cluster_id=len(clusters),
                documents=current_cluster,
                centroid=None,
                size=len(current_cluster),
                avg_distance=sum(doc.distance for doc in current_cluster) / len(current_cluster)
            )
            clusters.append(cluster)
        
        return clusters
    
    def _build_clusters(self, search_results: List[SearchResult], labels: np.ndarray, embeddings: np.ndarray) -> List[Cluster]:
        """构建聚类结果"""
        clusters = {}
        
        for i, (result, label) in enumerate(zip(search_results, labels)):
            if label == -1:  # 噪声点，创建单独的聚类
                label = f"noise_{i}"
            
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(result)
        
        # 转换为Cluster对象
        cluster_list = []
        for cluster_id, documents in clusters.items():
            # 计算聚类中心
            centroid = None
            if isinstance(cluster_id, int) and cluster_id >= 0:
                cluster_embeddings = [embeddings[i] for i, label in enumerate(labels) if label == cluster_id]
                if cluster_embeddings:
                    centroid = np.mean(cluster_embeddings, axis=0).tolist()
            
            cluster = Cluster(
                cluster_id=cluster_id if isinstance(cluster_id, int) else len(cluster_list),
                documents=documents,
                centroid=centroid,
                size=len(documents),
                avg_distance=sum(doc.distance for doc in documents) / len(documents)
            )
            cluster_list.append(cluster)
        
        return cluster_list
    
    def reorder_by_similarity(self, clusters: List[Cluster], strategy: str = "distance") -> List[Cluster]:
        """
        根据相似度策略重新排序聚类
        
        Args:
            clusters: 聚类列表
            strategy: 排序策略 ("distance", "cluster_size", "cluster_center")
            
        Returns:
            重新排序的聚类列表
        """
        if not clusters:
            return clusters
        
        # 对每个聚类内的文档进行排序
        for cluster in clusters:
            cluster.documents.sort(key=lambda x: x.distance)
        
        # 根据策略对聚类进行排序
        if strategy == "distance":
            # 按平均距离排序（相似度高的在前）
            clusters.sort(key=lambda x: x.avg_distance)
        elif strategy == "cluster_size":
            # 按聚类大小排序（大聚类在前）
            clusters.sort(key=lambda x: x.size, reverse=True)
        elif strategy == "cluster_center":
            # 按聚类中心与查询的相似度排序
            clusters.sort(key=lambda x: min(doc.distance for doc in x.documents))
        
        return clusters
    
    def calculate_cluster_metrics(self, clusters: List[Cluster]) -> Dict[str, float]:
        """
        计算聚类指标
        
        Args:
            clusters: 聚类列表
            
        Returns:
            聚类指标字典
        """
        if not clusters:
            return {}
        
        total_docs = sum(cluster.size for cluster in clusters)
        avg_cluster_size = total_docs / len(clusters)
        
        # 计算聚类内距离方差
        intra_cluster_variance = 0
        for cluster in clusters:
            if cluster.size > 1:
                distances = [doc.distance for doc in cluster.documents]
                variance = np.var(distances)
                intra_cluster_variance += variance * cluster.size
        
        intra_cluster_variance /= total_docs
        
        # 计算聚类间距离
        inter_cluster_distances = []
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist1 = clusters[i].avg_distance
                dist2 = clusters[j].avg_distance
                inter_cluster_distances.append(abs(dist1 - dist2))
        
        avg_inter_cluster_distance = np.mean(inter_cluster_distances) if inter_cluster_distances else 0
        
        return {
            "num_clusters": len(clusters),
            "total_documents": total_docs,
            "avg_cluster_size": avg_cluster_size,
            "intra_cluster_variance": intra_cluster_variance,
            "avg_inter_cluster_distance": avg_inter_cluster_distance,
            "largest_cluster_size": max(cluster.size for cluster in clusters),
            "smallest_cluster_size": min(cluster.size for cluster in clusters)
        }


def create_clustering_service() -> ClusteringService:
    """创建聚类服务实例"""
    return ClusteringService()