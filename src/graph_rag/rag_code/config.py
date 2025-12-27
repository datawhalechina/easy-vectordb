"""
基于图数据库的RAG系统配置文件
"""

from dataclasses import dataclass
from typing import Dict, Any
from dotenv import load_dotenv
import os
try:
    load_dotenv()
except UnicodeDecodeError:
    # 尝试其他编码
    load_dotenv(encoding='utf-16')  # 如果是 UTF-16
@dataclass
class GraphRAGConfig:
    """基于图数据库的RAG系统配置类"""

    # Neo4j数据库配置
    neo4j_uri: str = os.getenv("NEO4J_URI")
    neo4j_user: str =  os.getenv("NEO4J_USER")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD")
    neo4j_database: str = os.getenv("NEO4J_DATABASE")

    #选择向量数据库
    vector_db: str = os.getenv("VECTOR_DB")
    # Milvus配置
    milvus_host: str = os.getenv("MILVUS_HOST")
    milvus_port: int = os.getenv("MILVUS_PORT")
    milvus_collection_name: str = os.getenv("MILVUS_COLLECTION_NAME")
    milvus_dimension: int = os.getenv("MILVUS_DIMENSION")
    milvus_index_type: str = os.getenv("MILVUS_INDEX_TYPE")  # "Flat", "IVF", "HNSW"
    milvus_metric_type: str = os.getenv("MILVUS_METRIC_TYPE")

    # annoy配置
    annoy_index_path: str = os.getenv("ANNOY_INDEX_PATH", "./annoy_index")
    annoy_dimension: int = os.getenv("ANNOY_DIMENSION")
    annoy_metric_type: str = os.getenv("ANNOY_METRIC_TYPE")
    annoy_n_trees: int = os.getenv("ANNOY_N_TREES")
    # FAISS配置
    faiss_index_path: str = os.getenv("FAISS_INDEX_PATH")
    faiss_index_type: str = os.getenv("FAISS_INDEX_TYPE")  # "Flat", "IVF", "HNSW"
    faiss_nlist: int =  os.getenv("FAISS_NLIST")
    faiss_dimension: int = os.getenv("FAISS_DIMENSION")  # 保持原名称，FAISS也会使用这个值
    
    # 模型配置
    embedding_model: str = os.getenv("EMBEDDING_MODEL")
    embedding_api_key:str = os.getenv("SILICONFLOW_EMBEDING_API_KEY")
    embedding_base_url: str = os.getenv("SILICONFLOW_EMBEDDING_BASE_URL")
    llm_model: str = os.getenv("LLM_MODEL")
    llm_api_key:str = os.getenv("MOONSHOT_API_KEY")
    llm_base_url: str = os.getenv("MOONSHOT_BASE_URL")
    # 检索配置（LightRAG Round-robin策略）
    top_k: int = 5

    # 生成配置
    temperature: float = 0.1
    max_tokens: int = 2048

    # 图数据处理配置
    chunk_size: int = 100
    chunk_overlap: int = 20
    max_graph_depth: int = 2  # 图遍历最大深度

    def __post_init__(self):
        """初始化后的处理"""
        # LightRAG使用Round-robin策略，无需权重验证
        pass
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GraphRAGConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'neo4j_uri': self.neo4j_uri,
            'neo4j_user': self.neo4j_user,
            'neo4j_password': self.neo4j_password,
            'neo4j_database': self.neo4j_database,
            'embedding_model': self.embedding_model,
            'llm_model': self.llm_model,
            'llm_base_url': self.llm_base_url,
            'top_k': self.top_k,
            'vector_db': self.vector_db,
            'milvus_host': self.milvus_host,
            'milvus_port': self.milvus_port,
            'milvus_collection_name': self.milvus_collection_name,
            'milvus_dimension': self.milvus_dimension,
            'milvus_index_type': self.milvus_index_type,
            'milvus_metric_type': self.milvus_metric_type,
            'annoy_index_path': self.annoy_index_path,
            'annoy_dimension': self.annoy_dimension,
            'annoy_metric_type': self.annoy_metric_type,
            'annoy_n_trees': self.annoy_n_trees,
            'faiss_index_path': self.faiss_index_path,
            "llm_api_key": self.llm_api_key,
            'faiss_index_type': self.faiss_index_type,
            'faiss_nlist': self.faiss_nlist,
            'faiss_dimension': self.faiss_dimension,


            'embedding_api_key': self.embedding_api_key,
            "embedding_base_url":self.embedding_base_url,

            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'max_graph_depth': self.max_graph_depth
        }

# 默认配置实例
DEFAULT_CONFIG = GraphRAGConfig() 
#sk-iQ5zUoNiRcBAeluL2AcuUxxasfg6dzrmMXlh57Cc5eu0q6E0
if __name__ == "__main__":
    # 创建配置实例
    config = GraphRAGConfig()
    
    # 调用 to_dict 并打印
    config_dict = config.to_dict()
    print(config_dict)