"""
Milvus索引构建模块
"""

import logging
import time
from typing import List, Dict, Any, Optional

from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import numpy as np

logger = logging.getLogger(__name__)

class MilvusIndexConstructionModule:
    """Milvus索引构建模块 - 负责向量化和Milvus索引构建"""

    def __init__(self, 
                 host: str = "localhost", 
                 port: int = 19530,
                 collection_name: str = "cooking_knowledge",
                 dimension: int = 512,
                 model_name: str = "BAAI/bge-small-zh-v1.5"):
        """
        初始化Milvus索引构建模块

        Args:
            host: Milvus服务器地址
            port: Milvus服务器端口
            collection_name: 集合名称
            dimension: 向量维度
            model_name: 嵌入模型名称
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dimension = dimension
        self.model_name = model_name
        
        self.client = None
        self.embeddings = None
        self.collection_created = False
        
        self._setup_client()
        self._setup_embeddings()
    
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
    
    def _setup_client(self):
        """初始化Milvus客户端"""
        try:
            self.client = MilvusClient(
                uri=f"http://{self.host}:{self.port}"
            )
            logger.info(f"已连接到Milvus服务器: {self.host}:{self.port}")
            
            # 测试连接
            collections = self.client.list_collections()
            logger.info(f"连接成功，当前集合: {collections}")
            
        except Exception as e:
            logger.error(f"连接Milvus失败: {e}")
            raise
    
    def _setup_embeddings(self):
        """初始化嵌入模型"""
        logger.info(f"正在初始化嵌入模型: {self.model_name}")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        logger.info("嵌入模型初始化完成")
    
    def _create_collection_schema(self) -> CollectionSchema:
        """
        创建集合模式
        
        Returns:
            集合模式对象
        """
        # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=150, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=15000),
            FieldSchema(name="node_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="recipe_name", dtype=DataType.VARCHAR, max_length=300),
            FieldSchema(name="node_type", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="cuisine_type", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="difficulty", dtype=DataType.INT64),
            FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=150),
            FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=100)
        ]
        
        # 创建集合模式
        schema = CollectionSchema(
            fields=fields,
            description="中式烹饪知识图谱向量集合"
        )
        
        return schema
    
    def create_collection(self, force_recreate: bool = False) -> bool:
        """
        创建Milvus集合
        
        Args:
            force_recreate: 是否强制重新创建集合
        
        Returns:
            是否创建成功
        """
        try:
            # 检查集合是否存在
            if self.client.has_collection(self.collection_name):
                if force_recreate:
                    logger.info(f"删除已存在的集合: {self.collection_name}")
                    self.client.drop_collection(self.collection_name)
                else:
                    logger.info(f"集合 {self.collection_name} 已存在")
                    self.collection_created = True
                    return True
            
            # 创建集合
            schema = self._create_collection_schema()
            
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                metric_type="COSINE",  # 使用余弦相似度
                consistency_level="Strong"
            )
            
            logger.info(f"成功创建集合: {self.collection_name}")
            self.collection_created = True
            
            return True
            
        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            return False
    
    def create_index(self) -> bool:
        """
        创建向量索引
        
        Returns:
            是否创建成功
        """
        try:
            if not self.collection_created:
                raise ValueError("请先创建集合")
            
            # 使用prepare_index_params创建正确的IndexParams对象
            index_params = self.client.prepare_index_params()
            
            # 添加向量字段索引
            index_params.add_index(
                field_name="vector",
                index_type="HNSW",
                metric_type="COSINE",
                params={
                    "M": 16,
                    "efConstruction": 200
                }
            )
            
            self.client.create_index(
                collection_name=self.collection_name,
                index_params=index_params
            )
            
            logger.info("向量索引创建成功")
            return True
            
        except Exception as e:
            logger.error(f"创建索引失败: {e}")
            return False
    
    def build_vector_index(self, chunks: List[Document]) -> bool:
        """
        构建向量索引
        
        Args:
            chunks: 文档块列表
            
        Returns:
            是否构建成功
        """
        logger.info(f"正在构建Milvus向量索引，文档数量: {len(chunks)}...")
        
        if not chunks:
            raise ValueError("文档块列表不能为空")
        
        try:
            # 1. 创建集合（如果schema不兼容则强制重新创建）
            if not self.create_collection(force_recreate=True):
                return False
            
            # 2. 准备数据
            logger.info("正在生成向量embeddings...")
            texts = [chunk.page_content for chunk in chunks]
            vectors = self.embeddings.embed_documents(texts)
            
            # 3. 准备插入数据
            entities = []
            for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
                entity = {
                    "id": self._safe_truncate(chunk.metadata.get("chunk_id", f"chunk_{i}"), 150),
                    "vector": vector,
                    "text": self._safe_truncate(chunk.page_content, 15000),
                    "node_id": self._safe_truncate(chunk.metadata.get("node_id", ""), 100),
                    "recipe_name": self._safe_truncate(chunk.metadata.get("recipe_name", ""), 300),
                    "node_type": self._safe_truncate(chunk.metadata.get("node_type", ""), 100),
                    "category": self._safe_truncate(chunk.metadata.get("category", ""), 100),
                    "cuisine_type": self._safe_truncate(chunk.metadata.get("cuisine_type", ""), 200),
                    "difficulty": int(chunk.metadata.get("difficulty", 0)),
                    "doc_type": self._safe_truncate(chunk.metadata.get("doc_type", ""), 50),
                    "chunk_id": self._safe_truncate(chunk.metadata.get("chunk_id", f"chunk_{i}"), 150),
                    "parent_id": self._safe_truncate(chunk.metadata.get("parent_id", ""), 100)
                }
                entities.append(entity)
            
            # 4. 批量插入数据
            logger.info("正在插入向量数据...")
            batch_size = 100
            for i in range(0, len(entities), batch_size):
                batch = entities[i:i + batch_size]
                self.client.insert(
                    collection_name=self.collection_name,
                    data=batch
                )
                logger.info(f"已插入 {min(i + batch_size, len(entities))}/{len(entities)} 条数据")
            
            # 5. 创建索引
            if not self.create_index():
                return False
            
            # 6. 加载集合到内存
            self.client.load_collection(self.collection_name)
            logger.info("集合已加载到内存")
            
            # 7. 等待索引构建完成
            logger.info("等待索引构建完成...")
            time.sleep(2)
            
            logger.info(f"向量索引构建完成，包含 {len(chunks)} 个向量")
            return True
            
        except Exception as e:
            logger.error(f"构建向量索引失败: {e}")
            return False
    
    def add_documents(self, new_chunks: List[Document]) -> bool:
        """
        向现有索引添加新文档
        
        Args:
            new_chunks: 新的文档块列表
            
        Returns:
            是否添加成功
        """
        if not self.collection_created:
            raise ValueError("请先构建向量索引")
        
        logger.info(f"正在添加 {len(new_chunks)} 个新文档到索引...")
        
        try:
            # 生成向量
            texts = [chunk.page_content for chunk in new_chunks]
            vectors = self.embeddings.embed_documents(texts)
            
            # 准备插入数据
            entities = []
            for i, (chunk, vector) in enumerate(zip(new_chunks, vectors)):
                entity = {
                    "id": self._safe_truncate(chunk.metadata.get("chunk_id", f"new_chunk_{i}_{int(time.time())}"), 150),
                    "vector": vector,
                    "text": self._safe_truncate(chunk.page_content, 15000),
                    "node_id": self._safe_truncate(chunk.metadata.get("node_id", ""), 100),
                    "recipe_name": self._safe_truncate(chunk.metadata.get("recipe_name", ""), 300),
                    "node_type": self._safe_truncate(chunk.metadata.get("node_type", ""), 100),
                    "category": self._safe_truncate(chunk.metadata.get("category", ""), 100),
                    "cuisine_type": self._safe_truncate(chunk.metadata.get("cuisine_type", ""), 200),
                    "difficulty": int(chunk.metadata.get("difficulty", 0)),
                    "doc_type": self._safe_truncate(chunk.metadata.get("doc_type", ""), 50),
                    "chunk_id": self._safe_truncate(chunk.metadata.get("chunk_id", f"new_chunk_{i}_{int(time.time())}"), 150),
                    "parent_id": self._safe_truncate(chunk.metadata.get("parent_id", ""), 100)
                }
                entities.append(entity)
            
            # 插入数据
            self.client.insert(
                collection_name=self.collection_name,
                data=entities
            )
            
            logger.info("新文档添加完成")
            return True
            
        except Exception as e:
            logger.error(f"添加新文档失败: {e}")
            return False
    
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
        if not self.collection_created:
            raise ValueError("请先构建或加载向量索引")
        
        try:
            # 生成查询向量
            query_vector = self.embeddings.embed_query(query)
            
            # 构建过滤表达式
            filter_expr = ""
            if filters:
                filter_conditions = []
                for key, value in filters.items():
                    if isinstance(value, str):
                        filter_conditions.append(f'{key} == "{value}"')
                    elif isinstance(value, (int, float)):
                        filter_conditions.append(f'{key} == {value}')
                    elif isinstance(value, list):
                        # 支持IN操作
                        if all(isinstance(v, str) for v in value):
                            value_str = '", "'.join(value)
                            filter_conditions.append(f'{key} in ["{value_str}"]')
                        else:
                            value_str = ', '.join(map(str, value))
                            filter_conditions.append(f'{key} in [{value_str}]')
                
                if filter_conditions:
                    filter_expr = " and ".join(filter_conditions)
            
            # 执行搜索 - 修复参数传递
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 64}
            }
            
            # 构建搜索参数，避免重复传递
            search_kwargs = {
                "collection_name": self.collection_name,
                "data": [query_vector],
                "anns_field": "vector",
                "limit": k,
                "output_fields": ["text", "node_id", "recipe_name", "node_type", 
                                "category", "cuisine_type", "difficulty", "doc_type",
                                "chunk_id", "parent_id"],
                "search_params": search_params
            }
            
            # 只在有过滤条件时添加filter参数
            if filter_expr:
                search_kwargs["filter"] = filter_expr
                
            results = self.client.search(**search_kwargs)
            
            # 处理结果
            formatted_results = []
            if results and len(results) > 0:
                for hit in results[0]:  # results[0]因为我们只发送了一个查询向量
                    result = {
                        "id": hit["id"],
                        "score": hit["distance"],  # 注意：在COSINE距离中，值越大相似度越高
                        "text": hit["entity"]["text"],
                        "metadata": {
                            "node_id": hit["entity"]["node_id"],
                            "recipe_name": hit["entity"]["recipe_name"],
                            "node_type": hit["entity"]["node_type"],
                            "category": hit["entity"]["category"],
                            "cuisine_type": hit["entity"]["cuisine_type"],
                            "difficulty": hit["entity"]["difficulty"],
                            "doc_type": hit["entity"]["doc_type"],
                            "chunk_id": hit["entity"]["chunk_id"],
                            "parent_id": hit["entity"]["parent_id"]
                        }
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息
        
        Returns:
            统计信息字典
        """
        try:
            if not self.collection_created:
                return {"error": "集合未创建"}
            
            stats = self.client.get_collection_stats(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "row_count": stats.get("row_count", 0),
                "index_building_progress": stats.get("index_building_progress", 0),
                "stats": stats
            }
            
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
            if self.client.has_collection(self.collection_name):
                self.client.drop_collection(self.collection_name)
                logger.info(f"集合 {self.collection_name} 已删除")
                self.collection_created = False
                return True
            else:
                logger.info(f"集合 {self.collection_name} 不存在")
                return True
                
        except Exception as e:
            logger.error(f"删除集合失败: {e}")
            return False
    
    def has_collection(self) -> bool:
        """
        检查集合是否存在
        
        Returns:
            集合是否存在
        """
        try:
            return self.client.has_collection(self.collection_name)
        except Exception as e:
            logger.error(f"检查集合存在性失败: {e}")
            return False
    
    def load_collection(self) -> bool:
        """
        加载集合到内存
        
        Returns:
            是否加载成功
        """
        try:
            if not self.client.has_collection(self.collection_name):
                logger.error(f"集合 {self.collection_name} 不存在")
                return False
            
            self.client.load_collection(self.collection_name)
            self.collection_created = True
            logger.info(f"集合 {self.collection_name} 已加载到内存")
            return True
            
        except Exception as e:
            logger.error(f"加载集合失败: {e}")
            return False
    
    def close(self):
        """关闭连接"""
        if hasattr(self, 'client') and self.client:
            # Milvus客户端不需要显式关闭
            logger.info("Milvus连接已关闭")
    
    def __del__(self):
        """析构函数"""
        self.close() 