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
    keywords: Optional[List[str]] = None  # 新增：聚类关键词
    representative_doc: Optional[SearchResult] = None  # 新增：代表性文档
    visualization_coords: Optional[Tuple[float, float]] = None  # 新增：可视化坐标
    
    def __post_init__(self):
        self.size = len(self.documents)
        if self.documents:
            self.avg_distance = sum(doc.distance for doc in self.documents) / len(self.documents)
            # 选择距离最小的文档作为代表性文档
            if not self.representative_doc:
                self.representative_doc = min(self.documents, key=lambda x: x.distance)

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
                    keywords = self._extract_cluster_keywords(current_cluster)
                    cluster = Cluster(
                        cluster_id=len(clusters),
                        documents=current_cluster.copy(),
                        centroid=None,
                        size=len(current_cluster),
                        avg_distance=sum(doc.distance for doc in current_cluster) / len(current_cluster),
                        keywords=keywords
                    )
                    clusters.append(cluster)
                current_cluster = [result]
        
        # 添加最后一个聚类
        if current_cluster:
            keywords = self._extract_cluster_keywords(current_cluster)
            cluster = Cluster(
                cluster_id=len(clusters),
                documents=current_cluster,
                centroid=None,
                size=len(current_cluster),
                avg_distance=sum(doc.distance for doc in current_cluster) / len(current_cluster),
                keywords=keywords
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
            
            # 提取关键词
            keywords = self._extract_cluster_keywords(documents)
            
            cluster = Cluster(
                cluster_id=cluster_id if isinstance(cluster_id, int) else len(cluster_list),
                documents=documents,
                centroid=centroid,
                size=len(documents),
                avg_distance=sum(doc.distance for doc in documents) / len(documents),
                keywords=keywords
            )
            cluster_list.append(cluster)
        
        return cluster_list
    
    def _extract_cluster_keywords(self, documents: List[SearchResult]) -> List[str]:
        """为聚类提取关键词"""
        try:
            from .keyword_extractor import create_keyword_extractor
            
            extractor = create_keyword_extractor()
            doc_contents = [doc.content for doc in documents if doc.content]
            
            if not doc_contents:
                return []
            
            keywords = extractor.extract_keywords_from_cluster(doc_contents, max_keywords=5)
            return keywords
            
        except Exception as e:
            logger.error(f"提取聚类关键词失败: {e}")
            return []
    
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
    
    def create_cluster_scatter_plot(self, clusters: List[Cluster], reduction_method: str = "auto") -> Dict[str, Any]:
        """
        创建聚类散点图数据
        
        Args:
            clusters: 聚类列表
            reduction_method: 降维方法 ("auto", "tsne", "umap", "pca")
            
        Returns:
            散点图数据字典
        """
        if not clusters:
            return {"x": [], "y": [], "cluster_ids": [], "contents": [], "distances": [], "method_used": "none"}
        
        # 收集所有文档的embedding
        all_embeddings = []
        cluster_labels = []
        contents = []
        distances = []
        
        for cluster in clusters:
            for doc in cluster.documents:
                if doc.embedding:
                    all_embeddings.append(doc.embedding)
                    cluster_labels.append(cluster.cluster_id)
                    contents.append(doc.content[:100] + "..." if len(doc.content) > 100 else doc.content)
                    distances.append(doc.distance)
        
        if not all_embeddings:
            return {"x": [], "y": [], "cluster_ids": [], "contents": [], "distances": [], "method_used": "none"}
        
        # 使用指定方法进行降维
        coords_2d = self._reduce_dimensions(np.array(all_embeddings), method=reduction_method)
        
        # 记录实际使用的方法
        actual_method = self._get_actual_reduction_method(len(all_embeddings), reduction_method)
        
        # 更新聚类的可视化坐标
        coord_idx = 0
        for cluster in clusters:
            cluster_coords = []
            for doc in cluster.documents:
                if doc.embedding:
                    cluster_coords.append(coords_2d[coord_idx])
                    coord_idx += 1
            
            if cluster_coords:
                # 计算聚类中心坐标
                center_x = np.mean([coord[0] for coord in cluster_coords])
                center_y = np.mean([coord[1] for coord in cluster_coords])
                cluster.visualization_coords = (center_x, center_y)
        
        return {
            "x": coords_2d[:, 0].tolist(),
            "y": coords_2d[:, 1].tolist(),
            "cluster_ids": cluster_labels,
            "contents": contents,
            "distances": distances,
            "method_used": actual_method,
            "total_points": len(all_embeddings)
        }
    
    def _get_actual_reduction_method(self, n_samples: int, requested_method: str) -> str:
        """获取实际使用的降维方法"""
        if requested_method == "auto":
            if n_samples > 1000:
                return "umap"
            elif n_samples > 100:
                return "tsne"
            else:
                return "pca"
        return requested_method
    
    def create_cluster_size_chart(self, clusters: List[Cluster]) -> Dict[str, Any]:
        """
        创建聚类大小饼图数据
        
        Args:
            clusters: 聚类列表
            
        Returns:
            饼图数据字典
        """
        if not clusters:
            return {"labels": [], "values": [], "colors": []}
        
        labels = []
        values = []
        colors = []
        
        # 生成颜色
        color_palette = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
        ]
        
        for i, cluster in enumerate(clusters):
            labels.append(f"聚类 {cluster.cluster_id}")
            values.append(cluster.size)
            colors.append(color_palette[i % len(color_palette)])
        
        return {
            "labels": labels,
            "values": values,
            "colors": colors
        }
    
    def create_cluster_heatmap(self, clusters: List[Cluster]) -> Dict[str, Any]:
        """
        创建聚类相似度热力图数据
        
        Args:
            clusters: 聚类列表
            
        Returns:
            热力图数据字典
        """
        if not clusters or len(clusters) < 2:
            return {"matrix": [], "labels": []}
        
        n_clusters = len(clusters)
        similarity_matrix = np.zeros((n_clusters, n_clusters))
        labels = [f"聚类 {cluster.cluster_id}" for cluster in clusters]
        
        # 计算聚类间相似度
        for i, cluster_i in enumerate(clusters):
            for j, cluster_j in enumerate(clusters):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    # 使用聚类中心的余弦相似度
                    if cluster_i.centroid and cluster_j.centroid:
                        centroid_i = np.array(cluster_i.centroid)
                        centroid_j = np.array(cluster_j.centroid)
                        
                        # 计算余弦相似度
                        dot_product = np.dot(centroid_i, centroid_j)
                        norm_i = np.linalg.norm(centroid_i)
                        norm_j = np.linalg.norm(centroid_j)
                        
                        if norm_i > 0 and norm_j > 0:
                            similarity = dot_product / (norm_i * norm_j)
                            similarity_matrix[i][j] = max(0, similarity)  # 确保非负
                        else:
                            similarity_matrix[i][j] = 0
                    else:
                        # 如果没有中心点，使用平均距离的倒数作为相似度
                        avg_dist = (cluster_i.avg_distance + cluster_j.avg_distance) / 2
                        similarity_matrix[i][j] = 1 / (1 + avg_dist)
        
        return {
            "matrix": similarity_matrix.tolist(),
            "labels": labels
        }
    
    def generate_cluster_summary(self, clusters: List[Cluster]) -> Dict[str, Any]:
        """
        生成聚类摘要信息
        
        Args:
            clusters: 聚类列表
            
        Returns:
            聚类摘要字典
        """
        if not clusters:
            return {
                "total_clusters": 0,
                "total_documents": 0,
                "avg_cluster_size": 0,
                "cluster_details": []
            }
        
        total_docs = sum(cluster.size for cluster in clusters)
        avg_cluster_size = total_docs / len(clusters)
        
        cluster_details = []
        for cluster in clusters:
            detail = {
                "cluster_id": cluster.cluster_id,
                "size": cluster.size,
                "avg_distance": cluster.avg_distance,
                "keywords": cluster.keywords or [],
                "representative_content": cluster.representative_doc.content[:200] + "..." 
                    if cluster.representative_doc and len(cluster.representative_doc.content) > 200 
                    else cluster.representative_doc.content if cluster.representative_doc else "",
                "visualization_coords": cluster.visualization_coords
            }
            cluster_details.append(detail)
        
        return {
            "total_clusters": len(clusters),
            "total_documents": total_docs,
            "avg_cluster_size": avg_cluster_size,
            "cluster_details": cluster_details
        }
    
    def _reduce_dimensions(self, embeddings: np.ndarray, method: str = "auto") -> np.ndarray:
        """
        降维处理
        
        Args:
            embeddings: 高维向量数组
            method: 降维方法 ("auto", "tsne", "umap", "pca")
            
        Returns:
            2D坐标数组
        """
        if len(embeddings) == 0:
            return np.array([]).reshape(0, 2)
        elif len(embeddings) == 1:
            return np.array([[0, 0]])
        
        # 自动选择降维方法
        if method == "auto":
            n_samples = len(embeddings)
            if n_samples > 1000:
                method = "umap"  # 大数据集使用UMAP
            elif n_samples > 100:
                method = "tsne"  # 中等数据集使用t-SNE
            else:
                method = "pca"   # 小数据集使用PCA
        
        try:
            # 尝试UMAP（适合大数据集）
            if method == "umap":
                try:
                    import umap
                    
                    # 动态调整参数
                    n_neighbors = min(15, max(2, len(embeddings) // 10))
                    min_dist = 0.1
                    
                    reducer = umap.UMAP(
                        n_components=2,
                        n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        random_state=42,
                        metric='cosine'
                    )
                    return reducer.fit_transform(embeddings)
                    
                except ImportError:
                    logger.warning("UMAP不可用，使用t-SNE降维")
                    method = "tsne"
                except Exception as e:
                    logger.warning(f"UMAP降维失败: {e}，使用t-SNE")
                    method = "tsne"
            
            # 尝试t-SNE（适合中等数据集）
            if method == "tsne":
                try:
                    from sklearn.manifold import TSNE
                    
                    # 动态调整参数
                    n_samples = len(embeddings)
                    perplexity = min(30, max(5, n_samples // 3))
                    
                    # 对于大数据集，先用PCA降维到50维
                    if n_samples > 500 and embeddings.shape[1] > 50:
                        from sklearn.decomposition import PCA
                        pca = PCA(n_components=50, random_state=42)
                        embeddings = pca.fit_transform(embeddings)
                        logger.info(f"大数据集先用PCA降维到50维")
                    
                    tsne = TSNE(
                        n_components=2,
                        perplexity=perplexity,
                        random_state=42,
                        n_iter=300,
                        learning_rate='auto',
                        init='pca'
                    )
                    return tsne.fit_transform(embeddings)
                    
                except ImportError:
                    logger.warning("sklearn不可用，使用PCA降维")
                    method = "pca"
                except Exception as e:
                    logger.warning(f"t-SNE降维失败: {e}，使用PCA")
                    method = "pca"
            
            # 使用PCA作为备选方案
            if method == "pca":
                try:
                    from sklearn.decomposition import PCA
                    
                    # 确保不超过样本数量
                    n_components = min(2, len(embeddings) - 1, embeddings.shape[1])
                    if n_components < 2:
                        return self._simple_2d_projection(embeddings)
                    
                    pca = PCA(n_components=2, random_state=42)
                    result = pca.fit_transform(embeddings)
                    
                    # 记录解释方差比
                    explained_variance = pca.explained_variance_ratio_
                    logger.info(f"PCA解释方差比: {explained_variance}")
                    
                    return result
                    
                except ImportError:
                    logger.warning("sklearn不可用，使用简单投影")
                    return self._simple_2d_projection(embeddings)
                except Exception as e:
                    logger.warning(f"PCA降维失败: {e}，使用简单投影")
                    return self._simple_2d_projection(embeddings)
            
        except Exception as e:
            logger.error(f"降维失败: {e}，使用简单投影")
            return self._simple_2d_projection(embeddings)
    
    def _simple_2d_projection(self, embeddings: np.ndarray) -> np.ndarray:
        """
        简单的2D投影（当sklearn不可用时的备选方案）
        
        Args:
            embeddings: 高维向量数组
            
        Returns:
            2D坐标数组
        """
        # 使用前两个维度，或者计算主要的两个方向
        if embeddings.shape[1] >= 2:
            return embeddings[:, :2]
        else:
            # 如果维度不足，补零
            result = np.zeros((len(embeddings), 2))
            result[:, :embeddings.shape[1]] = embeddings
            return result


def create_clustering_service() -> ClusteringService:
    """创建聚类服务实例"""
    return ClusteringService()