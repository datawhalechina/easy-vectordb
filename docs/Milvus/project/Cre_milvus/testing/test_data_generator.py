"""
测试数据生成模块

生成用于性能测试的向量数据
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import logging
from sklearn.datasets import make_blobs
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType,
    utility
)

logger = logging.getLogger(__name__)


class TestDataGenerator:
    """
    测试数据生成器
    """
    
    def __init__(self, host: str = "localhost", port: str = "19530"):
        """
        初始化测试数据生成器
        
        参数:
            host: Milvus主机地址
            port: Milvus端口
        """
        self.host = host
        self.port = port
        self.connection_alias = "test_data_gen"
    
    def connect_milvus(self):
        """连接到Milvus"""
        try:
            connections.connect(
                alias=self.connection_alias,
                host=self.host,
                port=self.port,
                timeout=10
            )
            logger.info(f"连接Milvus成功: {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"连接Milvus失败: {e}")
            raise
    
    def disconnect_milvus(self):
        """断开Milvus连接"""
        if connections.has_connection(self.connection_alias):
            connections.disconnect(self.connection_alias)
            logger.info("断开Milvus连接")
    
    def generate_random_vectors(self, num_vectors: int, dimension: int, 
                              use_clusters: bool = True, num_clusters: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成随机向量数据
        
        参数:
            num_vectors: 向量数量
            dimension: 向量维度
            use_clusters: 是否使用聚类生成（更真实的数据分布）
            num_clusters: 聚类数量
        
        返回:
            (向量数组, 标签数组)
        """
        logger.info(f"生成 {num_vectors} 个 {dimension} 维向量...")
        
        if use_clusters:
            # 使用聚类生成更真实的数据分布
            vectors, labels = make_blobs(
                n_samples=num_vectors,
                centers=num_clusters,
                n_features=dimension,
                random_state=42,
                cluster_std=1.0
            )
        else:
            # 生成完全随机的向量
            vectors = np.random.normal(0, 1, (num_vectors, dimension))
            labels = np.random.randint(0, num_clusters, num_vectors)
        
        # 归一化向量
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms
        vectors = vectors.astype(np.float32)
        
        return vectors, labels
    
    def create_test_collection(self, collection_name: str = "locust_test_collection",
                             dimension: int = 256, num_vectors: int = 100000,
                             batch_size: int = 5000, drop_existing: bool = True) -> Dict[str, Any]:
        """
        创建测试集合并插入数据
        
        参数:
            collection_name: 集合名称
            dimension: 向量维度
            num_vectors: 向量数量
            batch_size: 批量插入大小
            drop_existing: 是否删除已存在的集合
        
        返回:
            创建结果信息
        """
        try:
            self.connect_milvus()
            
            # 检查并删除已存在的集合
            if utility.has_collection(collection_name, using=self.connection_alias):
                if drop_existing:
                    utility.drop_collection(collection_name, using=self.connection_alias)
                    logger.info(f"已删除旧集合: {collection_name}")
                else:
                    logger.warning(f"集合 {collection_name} 已存在，跳过创建")
                    return {"status": "exists", "collection_name": collection_name}
            
            # 定义集合结构
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
                FieldSchema(name="category", dtype=DataType.INT64),
            ]
            schema = CollectionSchema(fields, f"测试集合，维度: {dimension}")
            collection = Collection(collection_name, schema, using=self.connection_alias)
            
            # 生成测试数据
            vectors, labels = self.generate_random_vectors(num_vectors, dimension)
            
            # 批量插入数据
            logger.info("开始插入数据...")
            total_inserted = 0
            
            for i in range(0, num_vectors, batch_size):
                batch_end = min(i + batch_size, num_vectors)
                batch_vectors = vectors[i:batch_end].tolist()
                batch_labels = labels[i:batch_end].tolist()
                
                # 插入数据
                collection.insert([batch_vectors, batch_labels], using=self.connection_alias)
                total_inserted += len(batch_vectors)
                
                if total_inserted % (batch_size * 4) == 0:  # 每插入4个批次打印一次进度
                    logger.info(f"已插入 {total_inserted}/{num_vectors} 个向量")
            
            # 创建索引
            logger.info("创建索引...")
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": min(2048, num_vectors // 50)}  # 动态调整nlist
            }
            collection.create_index("vector", index_params, using=self.connection_alias)
            
            # 加载集合
            collection.load(using=self.connection_alias)
            
            logger.info(f"测试集合 {collection_name} 创建完成！")
            
            return {
                "status": "success",
                "collection_name": collection_name,
                "dimension": dimension,
                "num_vectors": num_vectors,
                "index_type": index_params["index_type"],
                "metric_type": index_params["metric_type"]
            }
            
        except Exception as e:
            logger.error(f"创建测试集合失败: {e}")
            return {"status": "error", "message": str(e)}
        
        finally:
            self.disconnect_milvus()
    
    def generate_test_queries(self, num_queries: int, dimension: int) -> List[List[float]]:
        """
        生成测试查询向量
        
        参数:
            num_queries: 查询数量
            dimension: 向量维度
        
        返回:
            查询向量列表
        """
        logger.info(f"生成 {num_queries} 个查询向量...")
        
        # 生成随机查询向量
        query_vectors = np.random.normal(0, 1, (num_queries, dimension)).astype(np.float32)
        
        # 归一化
        norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
        query_vectors = query_vectors / norms
        
        return query_vectors.tolist()
    
    def create_image_test_data(self, output_dir: str = "./test_images", 
                             num_images: int = 1000) -> Dict[str, Any]:
        """
        创建图像测试数据（生成CSV文件）
        
        参数:
            output_dir: 输出目录
            num_images: 图像数量
        
        返回:
            创建结果信息
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # 生成模拟图像数据信息
            image_data = []
            categories = [f"category_{i}" for i in range(20)]  # 20个类别
            
            for i in range(num_images):
                category = np.random.choice(categories)
                image_data.append({
                    "id": i,
                    "path": f"./images/{category}/image_{i:06d}.jpg",
                    "label": category,
                    "width": np.random.randint(224, 1024),
                    "height": np.random.randint(224, 1024)
                })
            
            # 保存为CSV文件
            df = pd.DataFrame(image_data)
            csv_path = output_path / "test_images.csv"
            df.to_csv(csv_path, index=False)
            
            logger.info(f"图像测试数据已保存到: {csv_path}")
            
            return {
                "status": "success",
                "csv_path": str(csv_path),
                "num_images": num_images,
                "categories": len(categories)
            }
            
        except Exception as e:
            logger.error(f"创建图像测试数据失败: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        获取集合信息
        
        参数:
            collection_name: 集合名称
        
        返回:
            集合信息
        """
        try:
            self.connect_milvus()
            
            if not utility.has_collection(collection_name, using=self.connection_alias):
                return {"status": "not_found", "message": f"集合 {collection_name} 不存在"}
            
            collection = Collection(collection_name, using=self.connection_alias)
            
            # 获取集合统计信息
            collection.load()
            stats = collection.get_stats()
            
            # 获取schema信息
            schema = collection.schema
            fields_info = []
            for field in schema.fields:
                field_info = {
                    "name": field.name,
                    "type": str(field.dtype),
                    "is_primary": field.is_primary
                }
                if hasattr(field, 'params'):
                    field_info["params"] = field.params
                fields_info.append(field_info)
            
            return {
                "status": "success",
                "collection_name": collection_name,
                "description": schema.description,
                "fields": fields_info,
                "stats": stats
            }
            
        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            return {"status": "error", "message": str(e)}
        
        finally:
            self.disconnect_milvus()
    
    def cleanup_test_data(self, collection_names: List[str]) -> Dict[str, Any]:
        """
        清理测试数据
        
        参数:
            collection_names: 要删除的集合名称列表
        
        返回:
            清理结果
        """
        try:
            self.connect_milvus()
            
            deleted_collections = []
            failed_collections = []
            
            for collection_name in collection_names:
                try:
                    if utility.has_collection(collection_name, using=self.connection_alias):
                        utility.drop_collection(collection_name, using=self.connection_alias)
                        deleted_collections.append(collection_name)
                        logger.info(f"已删除集合: {collection_name}")
                    else:
                        logger.warning(f"集合不存在: {collection_name}")
                except Exception as e:
                    failed_collections.append({"name": collection_name, "error": str(e)})
                    logger.error(f"删除集合失败 {collection_name}: {e}")
            
            return {
                "status": "completed",
                "deleted": deleted_collections,
                "failed": failed_collections
            }
            
        except Exception as e:
            logger.error(f"清理测试数据失败: {e}")
            return {"status": "error", "message": str(e)}
        
        finally:
            self.disconnect_milvus()