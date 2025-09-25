import pandas as pd
import numpy as np
from umap import UMAP
import logging as logger
from pymilvus import utility

def get_cluster_visualization_data(embeddings, labels, texts):
    """
    Generate cluster visualization data using UMAP for dimensionality reduction.
    
    Args:
        embeddings: List or array of embeddings (can be 1D or 2D)
        labels: Cluster labels for each embedding (NumPy array, list, or None)
        texts: Text associated with each embedding (list or None)
    
    Returns:
        DataFrame with UMAP coordinates, cluster labels, and texts
    """
    # Handle None input or empty embeddings
    if embeddings is None:
        return pd.DataFrame(columns=["x", "y", "cluster", "text"])
    
    embeddings = np.array(embeddings)
    
    # Check for empty arrays (0 points)
    if embeddings.size == 0:
        return pd.DataFrame(columns=["x", "y", "cluster", "text"])
    
    # Convert 1D array to 2D (single point)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    
    n_points = len(embeddings)
    
    # Set UMAP parameters
    n_neighbors = min(15, n_points - 1) if n_points > 1 else 1
    
    # Handle single point case
    if n_points == 1:
        umap_result = np.array([[0, 0]])
    else:
        umap = UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=n_neighbors,
            min_dist=0.1
        )
        umap_result = umap.fit_transform(embeddings)
    
    # Create DataFrame with results
    df = pd.DataFrame(umap_result, columns=["x", "y"])
    
    # FIXED: Use explicit None check instead of truthy check
    df["cluster"] = [str(l) for l in labels] if labels is not None else ["0"] * n_points
    df["text"] = texts if texts is not None else [""] * n_points
    
    # Filter out noise points (cluster = -1)
    if "-1" in df["cluster"].values:
        df = df[df["cluster"] != "-1"]
    
    return df

from pymilvus import Collection, connections
import numpy as np
from milvusBuilder.lazy_connection import get_lazy_connection

def get_all_embeddings_and_texts(collection_name, host="127.0.0.1", port="19530"):
    """
    从Milvus集合中获取所有嵌入向量和文本数据
    
    参数:
        collection_name: 集合名称
        host: Milvus主机地址
        port: Milvus端口
    
    返回:
        (ids, embeddings, texts): ID列表、嵌入向量数组、文本列表
    """
    try:
        lazy_connection = get_lazy_connection()
        
        # 使用延迟连接的上下文管理器
        with lazy_connection.get_connection(host, port) as connection_alias:
            logger.info(f"创建Collection对象: name={collection_name}, using={connection_alias}")
            print(f"创建Collection对象: name={collection_name}, using={connection_alias}")
            
            # 验证连接别名是否有效
            try:
                utility.list_collections(using=connection_alias)
                logger.info(f"✅ 连接别名验证通过: {connection_alias}")
                print(f"✅ 连接别名验证通过: {connection_alias}")
            except Exception as e:
                logger.error(f"❌ 连接别名验证失败: {connection_alias}, 错误: {e}")
                print(f"❌ 连接别名验证失败: {connection_alias}, 错误: {e}")
            
            collection = Collection(collection_name, using=connection_alias)
            collection.load()
            
            # 关键修复：等待集合加载完成
            from pymilvus import utility
            utility.wait_for_loading_complete(collection_name, using=connection_alias, timeout=300)
            
            # 验证集合状态
            load_state = utility.load_state(collection_name, using=connection_alias)
            if load_state != "Loaded":
                raise Exception(f"集合加载失败，当前状态: {load_state}")
            
            # 使用query方法获取所有数据
            # 注意：query_iterator在某些版本中可能不可用，使用query替代
            try:
                # 尝试使用query_iterator（推荐方式）
                iterator = collection.query_iterator(
                    batch_size=1000, 
                    expr="id >= 0", 
                    output_fields=["id", "embedding", "content"]
                )
                
                ids, embeddings, texts = [], [], []
                while True:
                    batch = iterator.next()
                    if len(batch) == 0:
                        break
                    for data in batch:
                        ids.append(data["id"])
                        embeddings.append(data["embedding"])
                        texts.append(data["content"])
                        
            except (AttributeError, Exception):
                # 如果query_iterator不可用，使用query方法
                # 先获取总数
                count_result = collection.query(
                    expr="id >= 0",
                    output_fields=["count(*)"],
                    limit=1
                )
                
                # 分批查询所有数据
                batch_size = 1000
                offset = 0
                ids, embeddings, texts = [], [], []
                
                while True:
                    batch_results = collection.query(
                        expr="id >= 0",
                        output_fields=["id", "embedding", "content"],
                        limit=batch_size,
                        offset=offset
                    )
                    
                    if not batch_results:
                        break
                    
                    for data in batch_results:
                        ids.append(data["id"])
                        embeddings.append(data["embedding"])
                        texts.append(data["content"])
                    
                    if len(batch_results) < batch_size:
                        break
                        
                    offset += batch_size
            
            return ids, np.array(embeddings), texts
        
    except Exception as e:
        print(f"获取集合数据失败: {e}")
        return [], np.array([]), []