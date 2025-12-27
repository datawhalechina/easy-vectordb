"""
Milvus 索引构建模块
"""

import logging
import time
from typing import List, Dict, Any, Optional
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import os
os.environ["TK_DATA_PATH"] = "./tiktoken_cache" # 某些版本有效
os.environ["TIKTOKEN_CACHE_DIR"] = "./tiktoken_cache"
from pymilvus import (
    connections, 
    FieldSchema, 
    CollectionSchema, 
    DataType, 
    Collection, 
    utility
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

class MilvusIndexConstructionModule:
    """Milvus 索引构建模块 - 负责向量化和 Milvus 集合管理"""

    def __init__(self, 
                 host: str = "localhost",
                 port: str = "19530",
                 collection_name: str = "recipe_knowledge_base",
                 dimension: int = 512,
                 model_name: str = "BAAI/bge-small-zh-v1.5",
                 index_type: str = "IVF_FLAT",  # Milvus 常用: IVF_FLAT, HNSW, IVFSQ8
                 metric_type: str = "IP",      # IP (内积) 或 L2 (欧氏距离)
                 embedding_api_key: str = None,
                 embedding_base_url: str = None
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dimension = dimension
        self.model_name = model_name
        self.index_type = index_type
        self.metric_type = metric_type
        self.embedding_api_key = embedding_api_key
        self.embedding_base_url = embedding_base_url
        
        self.embeddings = None
        self.collection = None
        
        # 1. 初始化嵌入模型
        self._setup_embeddings()
        
        # 2. 连接 Milvus 服务
        self._connect_milvus()

    def _connect_milvus(self):
        """建立与 Milvus 服务器的连接"""
        try:
            connections.connect("default", host=self.host, port=self.port,timeout=30 )
            logger.info(f"成功连接至 Milvus: {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Milvus 连接失败: {e}")
            raise

    def _setup_embeddings(self):
        """初始化嵌入模型（逻辑保留原样）"""
        use_cloud_api = all([self.embedding_api_key, self.embedding_base_url])
        if use_cloud_api:
            try:
                self.embeddings = OpenAIEmbeddings(
                    model=self.model_name,
                    openai_api_key=self.embedding_api_key,
                    openai_api_base=self.embedding_base_url,
                    
                    # --- 核心新增参数 ---
                    
                    # 1. 减小 Batch Size: 
                    # 默认通常是 1000。对于 BGE-M3 这种私有部署模型，
                    # 建议减小到 20-50，防止单个请求处理时间过长导致 502。
                    chunk_size=20, 
                    
                    # 2. 增加超时时间 (单位: 秒):
                    # 防止模型推理太慢导致连接被 Nginx 主动切断。
                    request_timeout=120,
                    
                    # 3. 最大重试次数:
                    max_retries=5
                )
                logger.info("云端 Embedding API 初始化完成")
            except Exception as e:
                logger.error(f"云端模型初始化失败: {e}，回退到本地模型")
                self._setup_local_embeddings()
        else:
            self._setup_local_embeddings()

    def _setup_local_embeddings(self):
        logger.info(f"正在初始化本地嵌入模型: {self.model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def _get_or_create_collection(self):
        """定义 Schema 并创建/获取 Collection"""
        if utility.has_collection(self.collection_name):
            # 如果已经存在旧的但配置错误的集合，建议先删除或直接加载
            self.collection = Collection(self.collection_name)
            logger.info(f"成功连接到现有集合: {self.collection_name}")
            return

        logger.info(f"正在创建集合: {self.collection_name}，维度: {self.dimension}")
        
        # 定义字段
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            # 核心修改点：显式使用 dim 关键字
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=int(self.dimension)),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=500), # 稍微调大长度上限
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=30000),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]
        
        schema = CollectionSchema(fields, description="Recipe Knowledge Base")
        self.collection = Collection(self.collection_name, schema)
        logger.info(f"集合 {self.collection_name} 创建成功")
    def has_collection(self) -> bool:
        """检查指定的集合是否存在"""
        try:
            return utility.has_collection(self.collection_name)
        except Exception as e:
            logger.error(f"检查集合是否存在时出错: {e}")
            return False
    def build_vector_index(self, chunks: List[Document], batch_size: int = 50) -> bool:
        """
        构建 Milvus 索引 (增强版：分批嵌入、分批插入、延时保护)
        """
        try:
            # 1. 确保集合存在并清空旧索引（如果需要重新构建）
            self._get_or_create_collection()
            
            total_chunks = len(chunks)
            logger.info(f"🚀 开始构建向量索引，总计: {total_chunks} 条数据，批次大小: {batch_size}")

            # 2. 分批处理流程
            for i in range(0, total_chunks, batch_size):
                batch_end = min(i + batch_size, total_chunks)
                batch_chunk_slice = chunks[i:batch_end]
                batch_texts = [chunk.page_content for chunk in batch_chunk_slice]
                
                # --- A. 获取当前批次的向量 ---
                logger.info(f"正在生成第 {i} 到 {batch_end} 条数据的向量...")
                batch_vectors = self.embeddings.embed_documents(batch_texts)
                
                # --- B. 准备当前批次的数据插入 ---
                # 对应字段: vector, chunk_id, text, metadata
                batch_ids = [c.metadata.get("chunk_id", str(j)) for j, c in enumerate(batch_chunk_slice, start=i)]
                batch_metadatas = [chunk.metadata for chunk in batch_chunk_slice]
                
                insert_data = [
                    batch_vectors, 
                    batch_ids, 
                    batch_texts, 
                    batch_metadatas
                ]
                
                # --- C. 插入 Milvus ---
                self.collection.insert(insert_data)
                
                # --- D. 延时策略 (防止压垮服务器或触发 API 频控) ---
                logger.info(f"✅ 批次 {batch_end}/{total_chunks} 写入成功，等待 0.2s...")
                time.sleep(0.2)

            # 3. 确保数据持久化
            logger.info("正在执行数据落盘 (Flush)...")
            self.collection.flush()
            
            # 4. 创建索引 (IVF_FLAT/HNSW)
            logger.info(f"正在创建索引: {self.index_type}...")
            index_params = {
                "metric_type": self.metric_type,
                "index_type": self.index_type,
                "params": {"nlist": 1024}
            }
            self.collection.create_index(field_name="vector", index_params=index_params)
            
            # 5. 加载集合到内存
            self.collection.load()
            
            # 修正之前的 f-string 日志
            logger.info(f"✨ 索引构建完成！当前集合实体总数: {self.collection.num_entities}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 索引构建失败: {e}")
            return False
        
    def load_collection(self) -> bool:
        """将集合加载到内存中，以便进行搜索"""
        try:
            if not self.collection:
                self.collection = Collection(self.collection_name)
            
            self.collection.load()
            logger.info(f"集合 {self.collection_name} 加载成功")
            return True
        except Exception as e:
            logger.error(f"加载集合失败: {e}")
            return False

    def similarity_search(self, query: str, k: int = 5, expr: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        相似度搜索
        
        Args:
            query: 查询文本
            k: 结果数
            expr: Milvus 的标量过滤表达式，例如 'difficulty > 1'
        """
        try:
            if self.collection is None:
                self._get_or_create_collection()
                self.collection.load()

            query_vector = self.embeddings.embed_query(query)
            
            search_params = {"metric_type": self.metric_type, "params": {"nprobe": 10}}
            
            results = self.collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=k,
                expr=expr,
                output_fields=["chunk_id", "text", "metadata"]
            )
            
            final_results = []
            for hit in results[0]:
                final_results.append({
                    "id": hit.entity.get("chunk_id"),
                    "score": hit.score,
                    "text": hit.entity.get("text"),
                    "metadata": hit.entity.get("metadata")
                })
            return final_results
            
        except Exception as e:
            logger.error(f"Milvus 搜索失败: {e}")
            return []

    def add_documents(self, new_chunks: List[Document]) -> bool:
        """向 Milvus 插入新文档"""
        return self.build_vector_index(new_chunks)

    def get_collection_stats(self) -> Dict[str, Any]:
        if self.collection:
            return {
                "row_count": self.collection.num_entities,
                "collection_name": self.collection_name
            }
        return {"error": "Collection not loaded"}

    def delete_collection(self) -> bool:
        try:
            utility.drop_collection(self.collection_name)
            logger.info(f"集合 {self.collection_name} 已删除")
            return True
        except Exception as e:
            logger.error(f"删除集合失败: {e}")
            return False

    def close(self):
        connections.disconnect("default")
        logger.info("Milvus 连接已断开")