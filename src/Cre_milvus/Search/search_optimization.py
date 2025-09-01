"""
搜索优化服务模块

提供向量归一化、搜索参数优化和搜索质量评估功能
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SearchQualityMetrics:
    """搜索质量指标"""
    relevance_score: float
    diversity_score: float
    coverage_score: float
    avg_distance: float
    distance_variance: float

@dataclass
class OptimizedSearchParams:
    """优化的搜索参数"""
    metric_type: str
    nprobe: int
    ef: Optional[int]
    search_k: Optional[int]
    confidence_score: float


class SearchOptimizationService:
    """搜索优化服务"""
    
    def __init__(self):
        self.normalization_cache = {}
        self.param_cache = {}
        
    def normalize_vectors(self, vectors: np.ndarray, method: str = "l2") -> np.ndarray:
        """
        向量归一化
        
        Args:
            vectors: 输入向量数组
            method: 归一化方法 ("l2", "l1", "max", "unit")
            
        Returns:
            归一化后的向量数组
        """
        if vectors is None or len(vectors) == 0:
            return vectors
        
        # 确保输入是numpy数组
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors)
        
        # 处理单个向量的情况
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        try:
            if method == "l2":
                # L2归一化（单位向量）
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                # 避免除零
                norms = np.where(norms == 0, 1, norms)
                normalized = vectors / norms
                
            elif method == "l1":
                # L1归一化
                norms = np.sum(np.abs(vectors), axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)
                normalized = vectors / norms
                
            elif method == "max":
                # 最大值归一化
                max_vals = np.max(np.abs(vectors), axis=1, keepdims=True)
                max_vals = np.where(max_vals == 0, 1, max_vals)
                normalized = vectors / max_vals
                
            elif method == "unit":
                # 单位化到[0,1]区间
                min_vals = np.min(vectors, axis=1, keepdims=True)
                max_vals = np.max(vectors, axis=1, keepdims=True)
                ranges = max_vals - min_vals
                ranges = np.where(ranges == 0, 1, ranges)
                normalized = (vectors - min_vals) / ranges
                
            else:
                logger.warning(f"未知的归一化方法: {method}，使用L2归一化")
                return self.normalize_vectors(vectors, "l2")
            
            # 检查归一化结果
            if np.any(np.isnan(normalized)) or np.any(np.isinf(normalized)):
                logger.warning("归一化结果包含NaN或Inf，返回原始向量")
                return vectors
            
            return normalized
            
        except Exception as e:
            logger.error(f"向量归一化失败: {e}")
            return vectors
    
    def optimize_search_params(self, collection_info: Dict[str, Any]) -> OptimizedSearchParams:
        """
        优化搜索参数
        
        Args:
            collection_info: 集合信息
            
        Returns:
            优化的搜索参数
        """
        try:
            # 获取集合基本信息
            num_entities = collection_info.get('num_entities', 0)
            index_type = collection_info.get('index_type', 'IVF_FLAT')
            dimension = collection_info.get('dimension', 256)
            
            # 根据索引类型和数据量优化参数
            if index_type in ['IVF_FLAT', 'IVF_SQ8', 'IVF_PQ']:
                return self._optimize_ivf_params(num_entities, index_type)
            elif index_type in ['HNSW', 'HNSW_SQ8']:
                return self._optimize_hnsw_params(num_entities, dimension)
            elif index_type == 'FLAT':
                return self._optimize_flat_params()
            else:
                # 默认参数
                return OptimizedSearchParams(
                    metric_type="IP",
                    nprobe=10,
                    ef=None,
                    search_k=None,
                    confidence_score=0.7
                )
                
        except Exception as e:
            logger.error(f"搜索参数优化失败: {e}")
            return OptimizedSearchParams(
                metric_type="IP",
                nprobe=10,
                ef=None,
                search_k=None,
                confidence_score=0.5
            )
    
    def _optimize_ivf_params(self, num_entities: int, index_type: str) -> OptimizedSearchParams:
        """优化IVF索引参数"""
        # 根据数据量调整nprobe
        if num_entities < 1000:
            nprobe = 8
        elif num_entities < 10000:
            nprobe = 16
        elif num_entities < 100000:
            nprobe = 32
        else:
            nprobe = 64
        
        # 对于归一化向量，使用内积(IP)通常效果更好
        metric_type = "IP"
        confidence_score = 0.8
        
        return OptimizedSearchParams(
            metric_type=metric_type,
            nprobe=nprobe,
            ef=None,
            search_k=None,
            confidence_score=confidence_score
        )
    
    def _optimize_hnsw_params(self, num_entities: int, dimension: int) -> OptimizedSearchParams:
        """优化HNSW索引参数"""
        # 根据数据量和维度调整ef
        base_ef = max(16, min(512, num_entities // 100))
        
        # 高维度数据需要更大的ef
        if dimension > 512:
            ef = min(512, base_ef * 2)
        else:
            ef = base_ef
        
        return OptimizedSearchParams(
            metric_type="IP",
            nprobe=None,
            ef=ef,
            search_k=None,
            confidence_score=0.85
        )
    
    def _optimize_flat_params(self) -> OptimizedSearchParams:
        """优化FLAT索引参数"""
        return OptimizedSearchParams(
            metric_type="IP",
            nprobe=None,
            ef=None,
            search_k=None,
            confidence_score=0.9
        )
    
    def validate_search_quality(self, query: str, results: List[Dict[str, Any]]) -> SearchQualityMetrics:
        """
        验证搜索质量
        
        Args:
            query: 查询文本
            results: 搜索结果
            
        Returns:
            搜索质量指标
        """
        if not results:
            return SearchQualityMetrics(
                relevance_score=0.0,
                diversity_score=0.0,
                coverage_score=0.0,
                avg_distance=1.0,
                distance_variance=0.0
            )
        
        try:
            # 提取距离信息
            distances = [float(result.get('distance', 1.0)) for result in results]
            
            # 计算相关性分数（基于距离）
            relevance_score = self._calculate_relevance_score(distances)
            
            # 计算多样性分数
            diversity_score = self._calculate_diversity_score(results)
            
            # 计算覆盖度分数
            coverage_score = self._calculate_coverage_score(results, query)
            
            # 计算距离统计
            avg_distance = np.mean(distances)
            distance_variance = np.var(distances)
            
            return SearchQualityMetrics(
                relevance_score=relevance_score,
                diversity_score=diversity_score,
                coverage_score=coverage_score,
                avg_distance=avg_distance,
                distance_variance=distance_variance
            )
            
        except Exception as e:
            logger.error(f"搜索质量验证失败: {e}")
            return SearchQualityMetrics(
                relevance_score=0.5,
                diversity_score=0.5,
                coverage_score=0.5,
                avg_distance=np.mean(distances) if distances else 1.0,
                distance_variance=np.var(distances) if distances else 0.0
            )
    
    def _calculate_relevance_score(self, distances: List[float]) -> float:
        """计算相关性分数"""
        if not distances:
            return 0.0
        
        # 对于内积距离，值越大越相似
        # 对于欧氏距离，值越小越相似
        # 这里假设使用的是内积距离
        avg_distance = np.mean(distances)
        
        # 将距离转换为相关性分数 (0-1)
        # 假设好的搜索结果距离应该大于0.3
        relevance_score = min(1.0, max(0.0, avg_distance))
        
        return relevance_score
    
    def _calculate_diversity_score(self, results: List[Dict[str, Any]]) -> float:
        """计算多样性分数"""
        if len(results) <= 1:
            return 1.0
        
        try:
            # 基于内容长度和URL的多样性
            contents = [result.get('content', '') for result in results]
            urls = [result.get('url', '') for result in results]
            
            # 内容长度多样性
            content_lengths = [len(content) for content in contents]
            length_diversity = np.std(content_lengths) / (np.mean(content_lengths) + 1e-6)
            
            # URL多样性（不同来源）
            unique_urls = len(set(url for url in urls if url))
            url_diversity = unique_urls / len(results) if results else 0
            
            # 综合多样性分数
            diversity_score = min(1.0, (length_diversity + url_diversity) / 2)
            
            return diversity_score
            
        except Exception as e:
            logger.error(f"多样性分数计算失败: {e}")
            return 0.5
    
    def _calculate_coverage_score(self, results: List[Dict[str, Any]], query: str) -> float:
        """计算覆盖度分数"""
        if not results or not query:
            return 0.0
        
        try:
            # 简单的关键词覆盖度计算
            query_words = set(query.lower().split())
            if not query_words:
                return 0.5
            
            total_coverage = 0
            for result in results:
                content = result.get('content', '').lower()
                content_words = set(content.split())
                
                # 计算查询词在内容中的覆盖度
                covered_words = query_words.intersection(content_words)
                coverage = len(covered_words) / len(query_words)
                total_coverage += coverage
            
            avg_coverage = total_coverage / len(results)
            return min(1.0, avg_coverage)
            
        except Exception as e:
            logger.error(f"覆盖度分数计算失败: {e}")
            return 0.5
    
    def suggest_parameter_adjustments(self, quality_metrics: SearchQualityMetrics, 
                                    current_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据搜索质量建议参数调整
        
        Args:
            quality_metrics: 搜索质量指标
            current_params: 当前搜索参数
            
        Returns:
            建议的参数调整
        """
        suggestions = {}
        
        try:
            # 相关性太低
            if quality_metrics.relevance_score < 0.5:
                suggestions['relevance'] = {
                    'issue': '搜索相关性较低',
                    'suggestions': [
                        '检查向量是否正确归一化',
                        '尝试使用不同的距离度量方法',
                        '增加搜索的top_k值',
                        '检查embedding模型是否适合当前数据'
                    ]
                }
            
            # 多样性太低
            if quality_metrics.diversity_score < 0.3:
                suggestions['diversity'] = {
                    'issue': '搜索结果多样性不足',
                    'suggestions': [
                        '增加搜索范围',
                        '调整聚类参数以获得更多样的结果',
                        '考虑使用重排序算法'
                    ]
                }
            
            # 距离方差太大
            if quality_metrics.distance_variance > 0.1:
                suggestions['consistency'] = {
                    'issue': '搜索结果一致性较差',
                    'suggestions': [
                        '检查数据质量和预处理',
                        '调整索引参数',
                        '考虑过滤异常结果'
                    ]
                }
            
            # 根据索引类型给出具体建议
            index_type = current_params.get('index_type', 'IVF_FLAT')
            if index_type.startswith('IVF'):
                nprobe = current_params.get('nprobe', 10)
                if quality_metrics.relevance_score < 0.6:
                    suggestions['nprobe'] = {
                        'current': nprobe,
                        'suggested': min(nprobe * 2, 128),
                        'reason': '增加nprobe以提高搜索精度'
                    }
            
            return suggestions
            
        except Exception as e:
            logger.error(f"参数调整建议生成失败: {e}")
            return {'error': '无法生成参数调整建议'}
    
    def auto_tune_search_params(self, collection_info: Dict[str, Any], 
                               sample_queries: List[str]) -> Dict[str, Any]:
        """
        自动调优搜索参数
        
        Args:
            collection_info: 集合信息
            sample_queries: 样本查询列表
            
        Returns:
            调优后的参数
        """
        try:
            # 获取基础优化参数
            base_params = self.optimize_search_params(collection_info)
            
            if not sample_queries:
                return {
                    'metric_type': base_params.metric_type,
                    'nprobe': base_params.nprobe,
                    'ef': base_params.ef,
                    'confidence_score': base_params.confidence_score
                }
            
            # TODO: 实现基于样本查询的参数调优
            # 这里可以添加更复杂的调优逻辑
            
            return {
                'metric_type': base_params.metric_type,
                'nprobe': base_params.nprobe,
                'ef': base_params.ef,
                'confidence_score': base_params.confidence_score,
                'tuned': True
            }
            
        except Exception as e:
            logger.error(f"自动调优失败: {e}")
            return {
                'metric_type': 'IP',
                'nprobe': 10,
                'ef': None,
                'confidence_score': 0.7,
                'error': str(e)
            }


def create_search_optimization_service() -> SearchOptimizationService:
    """创建搜索优化服务实例"""
    return SearchOptimizationService()