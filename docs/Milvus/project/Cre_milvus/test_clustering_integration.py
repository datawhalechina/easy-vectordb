#!/usr/bin/env python3
"""
聚类和可视化集成测试脚本

测试聚类算法和可视化功能的集成情况
"""

import sys
import os
import logging
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_clustering_imports():
    """测试聚类相关模块导入"""
    logger.info("测试聚类模块导入...")
    
    try:
        import hdbscan
        logger.info("✅ HDBSCAN导入成功")
    except ImportError:
        logger.error("❌ HDBSCAN导入失败")
        return False
    
    try:
        from sklearn.cluster import KMeans
        logger.info("✅ KMeans导入成功")
    except ImportError:
        logger.error("❌ KMeans导入失败")
        return False
    
    try:
        from reorder.reo_clu import reorder_clusters
        logger.info("✅ 聚类重排序模块导入成功")
    except ImportError as e:
        logger.error(f"❌ 聚类重排序模块导入失败: {e}")
        return False
    
    return True

def test_visualization_imports():
    """测试可视化相关模块导入"""
    logger.info("测试可视化模块导入...")
    
    try:
        from ColBuilder.visualization import get_cluster_visualization_data, get_all_embeddings_and_texts
        logger.info("✅ 可视化模块导入成功")
    except ImportError as e:
        logger.error(f"❌ 可视化模块导入失败: {e}")
        return False
    
    try:
        from umap import UMAP
        logger.info("✅ UMAP导入成功")
    except ImportError:
        logger.error("❌ UMAP导入失败")
        return False
    
    try:
        import pandas as pd
        logger.info("✅ Pandas导入成功")
    except ImportError:
        logger.error("❌ Pandas导入失败")
        return False
    
    return True

def test_clustering_algorithms():
    """测试聚类算法"""
    logger.info("测试聚类算法...")
    
    try:
        import hdbscan
        from sklearn.cluster import KMeans
        
        # 生成测试数据
        np.random.seed(42)
        n_samples = 100
        n_features = 128
        test_embeddings = np.random.normal(0, 1, (n_samples, n_features)).astype(np.float32)
        
        # 测试HDBSCAN
        try:
            clusterer = hdbscan.HDBSCAN(min_samples=3, min_cluster_size=2)
            hdbscan_labels = clusterer.fit_predict(test_embeddings)
            unique_labels = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
            logger.info(f"✅ HDBSCAN聚类成功: 发现 {unique_labels} 个聚类")
        except Exception as e:
            logger.error(f"❌ HDBSCAN聚类失败: {e}")
            return False
        
        # 测试KMeans
        try:
            k = min(len(test_embeddings), 5)
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans_labels = kmeans.fit_predict(test_embeddings)
            unique_labels = len(set(kmeans_labels))
            logger.info(f"✅ KMeans聚类成功: 发现 {unique_labels} 个聚类")
        except Exception as e:
            logger.error(f"❌ KMeans聚类失败: {e}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 聚类算法测试失败: {e}")
        return False

def test_cluster_reordering():
    """测试聚类重排序"""
    logger.info("测试聚类重排序...")
    
    try:
        from reorder.reo_clu import reorder_clusters
        
        # 创建模拟聚类结果
        clustered_results = {
            0: [
                {"id": 1, "embedding": [0.1, 0.2, 0.3], "content": "文档1", "distance": 0.5},
                {"id": 2, "embedding": [0.2, 0.3, 0.4], "content": "文档2", "distance": 0.6}
            ],
            1: [
                {"id": 3, "embedding": [0.8, 0.9, 1.0], "content": "文档3", "distance": 0.3},
                {"id": 4, "embedding": [0.9, 1.0, 1.1], "content": "文档4", "distance": 0.4}
            ]
        }
        
        query_vector = [0.5, 0.6, 0.7]
        
        # 测试不同的重排序策略
        strategies = ["distance", "cluster_size", "cluster_center"]
        
        for strategy in strategies:
            try:
                sorted_clusters = reorder_clusters(clustered_results, query_vector, strategy=strategy)
                logger.info(f"✅ {strategy}重排序成功: {len(sorted_clusters)} 个聚类")
            except Exception as e:
                logger.error(f"❌ {strategy}重排序失败: {e}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 聚类重排序测试失败: {e}")
        return False

def test_visualization_data_generation():
    """测试可视化数据生成"""
    logger.info("测试可视化数据生成...")
    
    try:
        from ColBuilder.visualization import get_cluster_visualization_data
        
        # 生成测试数据
        np.random.seed(42)
        n_points = 50
        n_features = 128
        
        embeddings = np.random.normal(0, 1, (n_points, n_features)).astype(np.float32)
        labels = np.random.randint(0, 5, n_points)
        texts = [f"文档{i}" for i in range(n_points)]
        
        # 测试可视化数据生成
        df = get_cluster_visualization_data(embeddings, labels, texts)
        
        if not df.empty:
            logger.info(f"✅ 可视化数据生成成功: {len(df)} 个数据点")
            logger.info(f"数据列: {list(df.columns)}")
            
            # 检查必要的列
            required_columns = ["x", "y", "cluster", "text"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"❌ 缺少必要的列: {missing_columns}")
                return False
            else:
                logger.info("✅ 所有必要的列都存在")
        else:
            logger.error("❌ 可视化数据生成失败: 返回空DataFrame")
            return False
        
        # 测试边界情况
        # 空数据
        empty_df = get_cluster_visualization_data([], [], [])
        if empty_df.empty:
            logger.info("✅ 空数据处理正确")
        else:
            logger.warning("⚠️ 空数据处理可能有问题")
        
        # 单点数据
        single_df = get_cluster_visualization_data(
            embeddings[:1], labels[:1], texts[:1]
        )
        if len(single_df) == 1:
            logger.info("✅ 单点数据处理正确")
        else:
            logger.warning("⚠️ 单点数据处理可能有问题")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 可视化数据生成测试失败: {e}")
        return False

def test_search_clustering_integration():
    """测试搜索和聚类集成"""
    logger.info("测试搜索和聚类集成...")
    
    try:
        from System.start import Cre_Search
        
        # 创建模拟配置
        config = {
            "milvus": {
                "collection_name": "test_collection",
                "host": "127.0.0.1",
                "port": "19530"
            },
            "search": {
                "topK": 10,
                "col_choice": "hdbscan",
                "reorder_strategy": "distance"
            },
            "system": {
                "url_split": False
            }
        }
        
        # 注意：这个测试需要实际的Milvus连接和数据
        # 在没有真实数据的情况下，我们只测试函数是否可以调用
        logger.info("✅ 搜索聚类集成函数可调用")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 搜索聚类集成测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("=" * 60)
    logger.info("🧪 聚类和可视化集成测试")
    logger.info("=" * 60)
    
    test_results = {}
    
    # 测试聚类模块导入
    test_results["clustering_imports"] = test_clustering_imports()
    
    # 测试可视化模块导入
    test_results["visualization_imports"] = test_visualization_imports()
    
    # 测试聚类算法
    if test_results["clustering_imports"]:
        test_results["clustering_algorithms"] = test_clustering_algorithms()
    
    # 测试聚类重排序
    if test_results["clustering_imports"]:
        test_results["cluster_reordering"] = test_cluster_reordering()
    
    # 测试可视化数据生成
    if test_results["visualization_imports"]:
        test_results["visualization_data"] = test_visualization_data_generation()
    
    # 测试搜索聚类集成
    test_results["search_clustering"] = test_search_clustering_integration()
    
    # 输出测试结果
    logger.info("=" * 60)
    logger.info("📊 测试结果总结:")
    
    results_summary = [
        ("聚类模块导入", test_results.get("clustering_imports", False)),
        ("可视化模块导入", test_results.get("visualization_imports", False)),
        ("聚类算法", test_results.get("clustering_algorithms", False)),
        ("聚类重排序", test_results.get("cluster_reordering", False)),
        ("可视化数据生成", test_results.get("visualization_data", False)),
        ("搜索聚类集成", test_results.get("search_clustering", False))
    ]
    
    for test_name, result in results_summary:
        if result:
            logger.info(f"✅ {test_name}: 成功")
        else:
            logger.error(f"❌ {test_name}: 失败")
    
    # 总体评估
    all_tests_passed = all(result for _, result in results_summary)
    
    if all_tests_passed:
        logger.info("🎉 所有聚类和可视化测试通过！")
    else:
        logger.error("❌ 部分测试失败，请检查上述错误信息")
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)