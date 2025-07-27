#!/usr/bin/env python3
"""
为Locust测试准备Milvus测试数据
"""

import numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sklearn.datasets import make_blobs
from tqdm import tqdm
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_collection(collection_name="locust_test_collection", 
                          dimension=256, 
                          num_vectors=100000,
                          batch_size=5000):
    """创建测试集合并插入数据"""
    
    # 连接到Milvus
    try:
        connections.connect("default", host="localhost", port="19530")
        logger.info("连接到Milvus成功")
    except Exception as e:
        logger.error(f"连接Milvus失败: {e}")
        return False
    
    # 删除已存在的集合
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        logger.info(f"删除已存在的集合: {collection_name}")
    
    # 定义集合schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
        FieldSchema(name="category", dtype=DataType.INT64),  # 添加分类字段用于过滤测试
    ]
    
    schema = CollectionSchema(fields, f"Locust测试集合，维度: {dimension}")
    collection = Collection(collection_name, schema)
    logger.info(f"创建集合成功: {collection_name}")
    
    # 生成测试向量数据
    logger.info(f"生成 {num_vectors} 个 {dimension} 维测试向量")
    
    # 使用聚类数据生成更真实的向量分布
    vectors, labels = make_blobs(
        n_samples=num_vectors,
        centers=20,  # 20个聚类中心
        n_features=dimension,
        random_state=42,
        cluster_std=1.0
    )
    
    # 归一化向量
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    vectors = vectors.astype(np.float32)
    
    # 生成分类标签
    categories = (labels % 10).astype(np.int64)  # 10个分类
    
    # 批量插入数据
    logger.info(f"开始插入数据，批次大小: {batch_size}")
    
    for i in tqdm(range(0, num_vectors, batch_size), desc="插入向量"):
        batch_end = min(i + batch_size, num_vectors)
        batch_vectors = vectors[i:batch_end].tolist()
        batch_categories = categories[i:batch_end].tolist()
        
        entities = [batch_vectors, batch_categories]
        collection.insert(entities)
    
    # 刷新数据
    collection.flush()
    logger.info("数据插入完成，开始创建索引")
    
    # 创建向量索引
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 2048}
    }
    
    collection.create_index("vector", index_params)
    logger.info("向量索引创建完成")
    
    # 创建标量字段索引（用于过滤测试）
    collection.create_index("category")
    logger.info("分类索引创建完成")
    
    # 加载集合
    collection.load()
    logger.info("集合加载完成")
    
    # 验证数据
    count = collection.num_entities
    logger.info(f"集合中共有 {count} 个向量")
    
    return True

def create_multiple_collections():
    """创建多个不同配置的测试集合"""
    configs = [
        {"name": "locust_small", "dimension": 128, "num_vectors": 50000},
        {"name": "locust_medium", "dimension": 256, "num_vectors": 100000},
        {"name": "locust_large", "dimension": 512, "num_vectors": 200000},
    ]
    
    for config in configs:
        logger.info(f"创建集合: {config['name']}")
        success = create_test_collection(
            collection_name=config["name"],
            dimension=config["dimension"],
            num_vectors=config["num_vectors"]
        )
        if success:
            logger.info(f"✅ 集合 {config['name']} 创建成功")
        else:
            logger.error(f"❌ 集合 {config['name']} 创建失败")

def main():
    parser = argparse.ArgumentParser(description="为Locust测试准备Milvus数据")
    parser.add_argument("--collection", default="locust_test_collection", help="集合名称")
    parser.add_argument("--dimension", type=int, default=256, help="向量维度")
    parser.add_argument("--num_vectors", type=int, default=100000, help="向量数量")
    parser.add_argument("--batch_size", type=int, default=5000, help="批次大小")
    parser.add_argument("--multiple", action="store_true", help="创建多个测试集合")
    
    args = parser.parse_args()
    
    print("🚀 Milvus Locust测试数据准备工具")
    print("="*50)
    
    if args.multiple:
        print("创建多个测试集合...")
        create_multiple_collections()
    else:
        print(f"创建单个测试集合: {args.collection}")
        print(f"维度: {args.dimension}, 向量数量: {args.num_vectors}")
        
        success = create_test_collection(
            collection_name=args.collection,
            dimension=args.dimension,
            num_vectors=args.num_vectors,
            batch_size=args.batch_size
        )
        
        if success:
            print("✅ 测试数据准备完成！")
            print(f"现在可以运行: locust -f milvus_locust_test.py")
        else:
            print("❌ 测试数据准备失败")

if __name__ == "__main__":
    main()