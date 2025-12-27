"""
FAISS索引构建模块
"""

import logging
import pickle
import os
import time
from typing import List, Dict, Any, Optional
import numpy as np

import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
logger = logging.getLogger(__name__)

class FAISSIndexConstructionModule:
    """FAISS索引构建模块 - 负责向量化和FAISS索引构建"""

    def __init__(self, 
                 index_path: str = "./faiss_index",
                 dimension: int = 512,
                 model_name: str = "BAAI/bge-small-zh-v1.5",
                 index_type: str = "IVF",
                 nlist: int = 100,
                 embedding_api_key:str=None ,
                 embedding_base_url: str=None
):
        """
        初始化FAISS索引构建模块

        Args:
            index_path: 索引文件保存路径
            dimension: 向量维度
            model_name: 嵌入模型名称
            index_type: 索引类型 ("Flat", "IVF", "HNSW")
            nlist: IVF索引的聚类中心数量
        """
        self.index_path = index_path
        self.dimension = int(dimension)
        self.model_name = model_name
        self.index_type = index_type
        self.nlist = int(nlist)
        self.embedding_api_key=embedding_api_key
        self.embedding_base_url=embedding_base_url
        # 确保索引目录存在
        os.makedirs(index_path, exist_ok=True)
        
        # 文件路径
        self.index_file = os.path.join(index_path, "faiss.index")
        self.metadata_file = os.path.join(index_path, "metadata.pkl")
        self.config_file = os.path.join(index_path, "config.pkl")
        
        self.embeddings = None
        self.index = None
        self.metadata = []  # 存储文档元数据
        self.id_to_index = {}  # ID到索引位置的映射
        self.index_ready = False
        
        self._setup_embeddings()
    def _setup_embeddings(self):
            """初始化嵌入模型：优先检查云端 API 配置，否则使用本地模型"""
            print(self.embedding_api_key)
            print(self.embedding_base_url)
            # 判断 API Key 和 Base URL 是否均已配置
            use_cloud_api = all([self.embedding_api_key, self.embedding_base_url])

            if use_cloud_api:
                logger.info(f"检测到 API 配置，正在通过 OpenAI 接口初始化模型: {self.model_name}")
                try:
                    self.embeddings = OpenAIEmbeddings(
                        model=self.model_name,
                        openai_api_key=self.embedding_api_key,
                        openai_api_base=self.embedding_base_url
                        # 如果你的 API 厂商（如 SiliconFlow）需要特定的维度参数，可以在此添加
                    )
                    logger.info("云端 Embedding API 初始化完成")
                except Exception as e:
                    logger.error(f"云端模型初始化失败: {e}，尝试回退到本地模型")
                    self._setup_local_embeddings()
            else:
                self._setup_local_embeddings()
    def _setup_local_embeddings(self):
            """初始化本地 HuggingFace 模型"""
            logger.info(f"正在初始化本地嵌入模型: {self.model_name}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("本地嵌入模型初始化完成")
    def _create_index(self, vectors: np.ndarray) -> faiss.Index:
        """
        创建FAISS索引 - 修复 int 与 str 比较及其他类型错误
        """
        n_vectors = vectors.shape[0]
        
        # 1. 强力转换 nlist 为整数
        try:
            # 处理可能出现的 str, float 或 None
            target_nlist = int(self.nlist) if self.nlist is not None else 100
        except (ValueError, TypeError):
            logger.warning(f"无效的 nlist 值: {self.nlist}，已回退到默认值 100")
            target_nlist = 100

        if self.index_type == "Flat":
            # 暴力搜索：IndexFlatIP 表示内积相似度（等同于余弦相似度，如果向量已归一化）
            index = faiss.IndexFlatIP(self.dimension)
            
        elif self.index_type == "IVF":
            # 2. 计算最终的 nlist。公式：min(设定值, 向量总数/10)
            # 确保 nlist 至少为 1，且不大于向量总数
            calculated_nlist = min(target_nlist, max(1, n_vectors // 10))
            
            # 定义量化器（IVF 必须依赖一个量化器来寻找聚类中心）
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, calculated_nlist)
            
            # 3. 训练索引（IVF 必须先训练才能添加数据）
            logger.info(f"正在训练 IVF 索引: 向量总数={n_vectors}, 聚类中心数={calculated_nlist}")
            index.train(vectors)
            
        elif self.index_type == "HNSW":
            # HNSW 是一种基于图的快速检索算法
            index = faiss.IndexHNSWFlat(self.dimension, 32)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 64
            
        else:
            raise ValueError(f"不支持的索引类型: {self.index_type}")
        
        return index
    def _safe_truncate(self, text: str, max_length: int) -> str:
        """
        安全截取字符串，处理None值
        
        Args:
            text: 输入文本
            max_length: 最大长度
            
        Returns:
            截取后的字符串
        """
        if text is None:
            return ""
        return str(text)[:max_length]
    
    def build_vector_index(self, chunks: List[Document]) -> bool:
        """
        构建向量索引（修复了 API 批处理限制导致的 400 错误）
        """
        logger.info(f"正在构建FAISS向量索引，文档数量: {len(chunks)}...")
        
        if not chunks:
            raise ValueError("文档块列表不能为空")
        
        try:
            # 1. 生成向量embeddings - 增加分批处理逻辑
            logger.info("正在生成向量embeddings...")
            texts = [chunk.page_content for chunk in chunks]
            
            # --- 修复代码开始 ---
            all_vectors = []
            batch_size = 32  # 建议设为 16 或 32，兼容大多数 API 限制
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                logger.info(f"正在处理第 {i} 到 {min(i + batch_size, len(texts))} 条数据的向量化...")
                
                # 获取当前批次的 embeddings
                batch_embeddings = self.embeddings.embed_documents(batch_texts)
                all_vectors.extend(batch_embeddings)
                
                # (可选) 增加极短的睡眠防止触发 API 速率限制 (QPS)
                # time.sleep(0.1) 
            
            vectors = np.array(all_vectors, dtype=np.float32)
            # --- 修复代码结束 ---
            
            # 2. 创建索引
            logger.info(f"创建{self.index_type}索引...")
            self.index = self._create_index(vectors)
            
            # 3. 添加向量到索引
            logger.info("添加向量到索引...")
            self.index.add(vectors)
            
            # 4. 准备元数据 (保持原有逻辑不变)
            logger.info("准备元数据...")
            self.metadata = []
            self.id_to_index = {}
            
            for i, chunk in enumerate(chunks):
                # ... 原有的元数据封装逻辑 ...
                chunk_id = chunk.metadata.get("chunk_id", f"chunk_{i}")
                metadata = {
                    "id": self._safe_truncate(chunk_id, 150),
                    "text": self._safe_truncate(chunk.page_content, 15000),
                    "node_id": self._safe_truncate(chunk.metadata.get("node_id", ""), 100),
                    "recipe_name": self._safe_truncate(chunk.metadata.get("recipe_name", ""), 300),
                    "node_type": self._safe_truncate(chunk.metadata.get("node_type", ""), 100),
                    "category": self._safe_truncate(chunk.metadata.get("category", ""), 100),
                    "cuisine_type": self._safe_truncate(chunk.metadata.get("cuisine_type", ""), 200),
                    "difficulty": int(chunk.metadata.get("difficulty", 0)),
                    "doc_type": self._safe_truncate(chunk.metadata.get("doc_type", ""), 50),
                    "chunk_id": self._safe_truncate(chunk_id, 150),
                    "parent_id": self._safe_truncate(chunk.metadata.get("parent_id", ""), 100)
                }
                self.metadata.append(metadata)
                self.id_to_index[chunk_id] = i
            
            # 5. 保存索引和元数据
            self.save_index()
            self.index_ready = True
            logger.info(f"向量索引构建完成，包含 {len(chunks)} 个向量")
            return True
            
        except Exception as e:
            logger.error(f"构建向量索引失败: {e}")
            import traceback
            logger.error(traceback.format_exc()) # 打印详细堆栈方便调试
            return False
    
    def save_index(self):
        """保存索引到磁盘"""
        try:
            # 保存FAISS索引
            faiss.write_index(self.index, self.index_file)
            
            # 保存元数据
            with open(self.metadata_file, 'wb') as f:
                pickle.dump({
                    'metadata': self.metadata,
                    'id_to_index': self.id_to_index
                }, f)
            
            # 保存配置
            config = {
                'dimension': self.dimension,
                'index_type': self.index_type,
                'nlist': self.nlist,
                'model_name': self.model_name,
                'total_vectors': len(self.metadata)
            }
            
            with open(self.config_file, 'wb') as f:
                pickle.dump(config, f)
            
            logger.info(f"索引已保存到: {self.index_path}")
            
        except Exception as e:
            logger.error(f"保存索引失败: {e}")
            raise
    
    def load_index(self) -> bool:
        """从磁盘加载索引"""
        try:
            if not os.path.exists(self.index_file):
                logger.info("索引文件不存在")
                return False
            
            # 加载FAISS索引
            self.index = faiss.read_index(self.index_file)
            
            # 加载元数据
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'rb') as f:
                    data = pickle.load(f)
                    self.metadata = data['metadata']
                    self.id_to_index = data['id_to_index']
            
            # 加载配置
            if os.path.exists(self.config_file):
                with open(self.config_file, 'rb') as f:
                    config = pickle.load(f)
                    logger.info(f"加载索引配置: {config}")
            
            self.index_ready = True
            logger.info(f"索引加载成功，包含 {len(self.metadata)} 个向量")
            return True
            
        except Exception as e:
            logger.error(f"加载索引失败: {e}")
            return False
    
    def has_collection(self) -> bool:
        """检查索引是否存在"""
        return os.path.exists(self.index_file) and os.path.exists(self.metadata_file)
    
    def load_collection(self) -> bool:
        """加载集合到内存（兼容接口）"""
        return self.load_index()
    
    def similarity_search(self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        相似度搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            filters: 过滤条件
            
        Returns:
            搜索结果列表
        """
        if not self.index_ready:
            raise ValueError("请先构建或加载向量索引")
        
        try:
            # 生成查询向量
            query_vector = self.embeddings.embed_query(query)
            query_vector = np.array([query_vector], dtype=np.float32)
            
            # 执行搜索
            scores, indices = self.index.search(query_vector, k)
            
            # 处理结果
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # FAISS返回-1表示没有找到足够的结果
                    continue
                
                if idx >= len(self.metadata):
                    continue
                
                metadata = self.metadata[idx]
                
                # 应用过滤器
                if filters and not self._apply_filters(metadata, filters):
                    continue
                
                result = {
                    "id": metadata["id"],
                    "score": float(score),  # FAISS返回的是内积分数，值越大相似度越高
                    "text": metadata["text"],
                    "metadata": {
                        "node_id": metadata["node_id"],
                        "recipe_name": metadata["recipe_name"],
                        "node_type": metadata["node_type"],
                        "category": metadata["category"],
                        "cuisine_type": metadata["cuisine_type"],
                        "difficulty": metadata["difficulty"],
                        "doc_type": metadata["doc_type"],
                        "chunk_id": metadata["chunk_id"],
                        "parent_id": metadata["parent_id"]
                    }
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            return []
    
    def _apply_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """应用过滤条件"""
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            metadata_value = metadata[key]
            
            if isinstance(value, list):
                if metadata_value not in value:
                    return False
            else:
                if metadata_value != value:
                    return False
        
        return True
    
    def add_documents(self, new_chunks: List[Document]) -> bool:
        """
        向现有索引添加新文档
        
        Args:
            new_chunks: 新的文档块列表
            
        Returns:
            是否添加成功
        """
        if not self.index_ready:
            raise ValueError("请先构建向量索引")
        
        logger.info(f"正在添加 {len(new_chunks)} 个新文档到索引...")
        
        try:
            # 生成向量
            texts = [chunk.page_content for chunk in new_chunks]
            vectors = self.embeddings.embed_documents(texts)
            vectors = np.array(vectors, dtype=np.float32)
            
            # 添加到索引
            start_idx = len(self.metadata)
            self.index.add(vectors)
            
            # 更新元数据
            for i, chunk in enumerate(new_chunks):
                chunk_id = chunk.metadata.get("chunk_id", f"new_chunk_{i}_{int(time.time())}")
                
                metadata = {
                    "id": self._safe_truncate(chunk_id, 150),
                    "text": self._safe_truncate(chunk.page_content, 15000),
                    "node_id": self._safe_truncate(chunk.metadata.get("node_id", ""), 100),
                    "recipe_name": self._safe_truncate(chunk.metadata.get("recipe_name", ""), 300),
                    "node_type": self._safe_truncate(chunk.metadata.get("node_type", ""), 100),
                    "category": self._safe_truncate(chunk.metadata.get("category", ""), 100),
                    "cuisine_type": self._safe_truncate(chunk.metadata.get("cuisine_type", ""), 200),
                    "difficulty": int(chunk.metadata.get("difficulty", 0)),
                    "doc_type": self._safe_truncate(chunk.metadata.get("doc_type", ""), 50),
                    "chunk_id": self._safe_truncate(chunk_id, 150),
                    "parent_id": self._safe_truncate(chunk.metadata.get("parent_id", ""), 100)
                }
                
                self.metadata.append(metadata)
                self.id_to_index[chunk_id] = start_idx + i
            
            # 保存更新后的索引
            self.save_index()
            
            logger.info("新文档添加完成")
            return True
            
        except Exception as e:
            logger.error(f"添加新文档失败: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息
        
        Returns:
            统计信息字典
        """
        try:
            if not self.index_ready:
                return {"error": "索引未就绪"}
            
            stats = {
                "collection_name": "faiss_index",
                "row_count": len(self.metadata),
                "index_type": self.index_type,
                "dimension": self.dimension,
                "index_size": self.index.ntotal if self.index else 0,
                "is_trained": self.index.is_trained if hasattr(self.index, 'is_trained') else True
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"获取集合统计信息失败: {e}")
            return {"error": str(e)}
    
    def delete_collection(self) -> bool:
        """
        删除集合
        
        Returns:
            是否删除成功
        """
        try:
            # 删除索引文件
            if os.path.exists(self.index_file):
                os.remove(self.index_file)
            
            # 删除元数据文件
            if os.path.exists(self.metadata_file):
                os.remove(self.metadata_file)
            
            # 删除配置文件
            if os.path.exists(self.config_file):
                os.remove(self.config_file)
            
            # 重置状态
            self.index = None
            self.metadata = []
            self.id_to_index = {}
            self.index_ready = False
            
            logger.info("FAISS索引已删除")
            return True
            
        except Exception as e:
            logger.error(f"删除索引失败: {e}")
            return False
    
    def close(self):
        """关闭连接（兼容接口）"""
        logger.info("FAISS索引模块已关闭")
    
    def __del__(self):
        """析构函数"""
        self.close()
