"""
Faiss 向量数据库实现
"""

import numpy as np
import pickle
import os
from typing import List, Tuple, Dict, Any, Optional

class FaissVectorStore:
    def __init__(self, dimension: int):
        """初始化 Faiss 向量数据库"""
        self.dimension = dimension
        self.texts = []
        self.embeddings = []
        self.metadata = []
        self.index = self.faiss.IndexFlatIP(dimension)  # 使用内积 (Inner Product) 进行余弦相似度搜索
    
    def add_vectors(self, embeddings: List[List[float]], texts: List[str], metadata: Optional[List[Dict[str, Any]]] = None):
        """向数据库添加向量及其对应的文本内容，可选添加元数据"""
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # 归一化以支持余弦相似度
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        embeddings_array = embeddings_array / (norms + 1e-8)  # 添加极小值防止除以零
        
        self.index.add(embeddings_array)
        self.texts.extend(texts)
        self.embeddings.extend(embeddings)
        
        # 添加元数据（如果未提供，则默认为空字典）
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in texts])
    
    def search(self, query_embedding: List[float], k: int = 5, return_metadata: bool = False) -> List[Tuple[str, float, Optional[Dict[str, Any]]]]:
        """搜索相似向量，可选择返回元数据"""
        if self.index.ntotal == 0:
            return []
        
        query_array = np.array([query_embedding], dtype=np.float32)
        
        # 归一化查询向量
        norm = np.linalg.norm(query_array)
        if norm > 0:
            query_array = query_array / norm
        
        scores, indices = self.index.search(query_array, min(k, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # 有效索引
                metadata = self.metadata[idx] if return_metadata and idx < len(self.metadata) else None
                results.append((self.texts[idx], float(score), metadata))
        
        return results
    
    def save(self, filepath: str):
        """将向量数据库保存到本地磁盘"""
        data = {
            'texts': self.texts,
            'embeddings': self.embeddings,
            'dimension': self.dimension,
            'metadata': self.metadata
        }
        
        # 保存索引文件
        self.faiss.write_index(self.index, f"{filepath}.index")
        
        # 保存元数据和其它数据
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """从磁盘加载向量数据库"""
        # 加载索引文件
        if os.path.exists(f"{filepath}.index"):
            self.index = self.faiss.read_index(f"{filepath}.index")
        
        # 加载元数据和其它数据
        if os.path.exists(f"{filepath}.pkl"):
            with open(f"{filepath}.pkl", 'rb') as f:
                data = pickle.load(f)
                self.texts = data['texts']
                self.embeddings = data['embeddings']
                self.dimension = data['dimension']
                self.metadata = data.get('metadata', [])  # 向后兼容处理
    
    def clear(self):
        """清空所有存储的向量、文本和元数据"""
        self.texts = []
        self.embeddings = []
        self.metadata = []
        self.index = self.faiss.IndexFlatIP(self.dimension)