import logging
import pickle
import os
import time
from typing import List, Dict, Any, Optional
import numpy as np

from annoy import AnnoyIndex  # 需要安装: pip install annoy
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

class AnnoyIndexConstructionModule:
    """Annoy索引构建模块 - 负责向量化和Annoy索引构建"""

    def __init__(self, 
                 index_path: str = "./annoy_index",
                 dimension: int = 512,
                 model_name: str = "BAAI/bge-small-zh-v1.5",
                 metric: str = "angular",  # 默认值
                 n_trees: int = 100,
                 embedding_api_key: str = None,
                 embedding_base_url: str = None
    ):
        # 1. 路径防御
        self.index_path = str(index_path) if index_path else "./annoy_index"
        
        # 2. 维度防御
        try:
            self.dimension = int(dimension) if dimension is not None else 512
        except:
            self.dimension = 512
            
        # 3. 度量方式防御 (解决你报错的核心点)
        # 如果传入的是 None，强制转为 "angular"
        self.metric = str(metric) if metric else "angular"
        
        # 4. 树数量防御
        try:
            self.n_trees = int(n_trees) if n_trees is not None else 100
        except:
            self.n_trees = 100

        self.model_name = model_name
        self.embedding_api_key = embedding_api_key
        self.embedding_base_url = embedding_base_url
        
        # 确保目录存在
        os.makedirs(self.index_path, exist_ok=True)
        
        # 文件路径拼接
        self.index_file = os.path.join(self.index_path, "annoy.index")
        self.metadata_file = os.path.join(self.index_path, "metadata.pkl")
        self.config_file = os.path.join(self.index_path, "config.pkl")
        
        self.embeddings = None
        self.index = None
        self.metadata = []
        self.index_ready = False
        
        self._setup_embeddings()
    def _setup_embeddings(self):
        """初始化嵌入模型"""
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
    import time
    import os

    def build_vector_index(self, chunks: List[Document]) -> bool:
        """构建Annoy向量索引（已增加分批处理与错误检查）"""
        logger.info(f"正在构建Annoy索引，文档数量: {len(chunks)}...")
        if not chunks: 
            logger.warning("没有可用的文档分块")
            return False
        
        try:
            # --- 1. 分批生成向量 (解决 502 报错) ---
            texts = [chunk.page_content for chunk in chunks]
            vectors = []
            batch_size = 50  # 💡 减小每批数量
            
            logger.info(f"开始向量化，每批大小: {batch_size}")
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                # 获取当前批次的向量
                batch_vectors = self.embeddings.embed_documents(batch_texts)
                vectors.extend(batch_vectors)
                
                # 💡 增加 0.2 秒延迟，防止压垮服务器网关
                time.sleep(0.2)
                if (i // batch_size) % 5 == 0:
                    logger.info(f"进度: {min(i + batch_size, len(texts))}/{len(texts)}")

            # --- 2. 初始化Annoy索引 ---
            if not hasattr(self, 'dimension') or self.dimension is None:
                raise ValueError("维度(dimension)未定义，请检查模型配置")
                
            annoy_index = AnnoyIndex(self.dimension, self.metric)
            
            # --- 3. 添加向量并准备元数据 ---
            self.metadata = []
            for i, (vec, chunk) in enumerate(zip(vectors, chunks)):
                annoy_index.add_item(i, vec)
                
                # 确保 id 不是 None
                chunk_id = chunk.metadata.get("chunk_id") or f"chunk_{i}"
                self.metadata.append({
                    "id": chunk_id,
                    "text": chunk.page_content,
                    **chunk.metadata 
                })
            
            # --- 4. 构建树 ---
            logger.info(f"开始构建Annoy树 (n_trees={self.n_trees})...")
            annoy_index.build(self.n_trees)
            
            # --- 5. 保存 (检查路径是否为 None) ---
            self.index = annoy_index
            
            # 🛡️ 防御性检查：防止出现 argument 2 must be str, not None
            if not hasattr(self, 'index_path') or self.index_path is None:
                logger.error("检测到 index_path 为空，将使用默认路径 './annoy_index.idx'")
                self.index_path = "./annoy_index.idx"
                
            self.save_index()
            
            self.index_ready = True
            return True
            
        except Exception as e:
            import traceback
            logger.error(f"构建Annoy索引失败: {str(e)}")
            logger.error(traceback.format_exc()) # 打印完整堆栈，方便定位 None 出现在哪一行
            return False

    def save_index(self):
        """保存到磁盘"""
        self.index.save(self.index_file)
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        config = {
            'dimension': self.dimension,
            'metric': self.metric,
            'n_trees': self.n_trees,
            'model_name': self.model_name
        }
        with open(self.config_file, 'wb') as f:
            pickle.dump(config, f)
        logger.info(f"Annoy索引已保存")

    def load_index(self) -> bool:
        """加载Annoy索引"""
        try:
            if not os.path.exists(self.index_file): return False
            
            # 必须先知道维度和度量才能加载
            if os.path.exists(self.config_file):
                with open(self.config_file, 'rb') as f:
                    config = pickle.load(f)
                    self.dimension = config.get('dimension', self.dimension)
                    self.metric = config.get('metric', self.metric)

            self.index = AnnoyIndex(self.dimension, self.metric)
            self.index.load(self.index_file) # 使用 mmap 加载
            
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'rb') as f:
                    self.metadata = pickle.load(f)
            
            self.index_ready = True
            logger.info(f"Annoy索引加载成功，包含 {len(self.metadata)} 个节点")
            return True
        except Exception as e:
            logger.error(f"加载失败: {e}")
            return False

    def similarity_search(self, query: str, k: int = 5, search_k: int = -1) -> List[Dict[str, Any]]:
        """
        相似度搜索
        search_k: 搜索时的遍历节点数，-1 表示使用默认值 (n_trees * k)
        """
        if not self.index_ready: return []
        
        try:
            query_vector = self.embeddings.embed_query(query)
            
            # Annoy 返回 (indices, distances)
            indices, distances = self.index.get_nns_by_vector(
                query_vector, k, search_k=search_k, include_distances=True
            )
            
            results = []
            for idx, dist in zip(indices, distances):
                if idx >= len(self.metadata): continue
                
                # 转换分数：Annoy angular 距离越小越相似
                # 转换公式取决于具体业务需求，这里保留原始距离或做简单映射
                score = 1 - dist if self.metric == "angular" else dist
                
                meta = self.metadata[idx]
                results.append({
                    "id": meta.get("id"),
                    "score": float(score),
                    "text": meta.get("text"),
                    "metadata": meta
                })
            return results
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []

    def delete_collection(self) -> bool:
        """删除索引文件"""
        for f in [self.index_file, self.metadata_file, self.config_file]:
            if os.path.exists(f): os.remove(f)
        self.index = None
        self.index_ready = False
        return True

    def close(self):
        # Annoy 不需要显式关闭，但在处理多进程映射时可以手动释放
        if self.index:
            self.index.unmap()
        logger.info("Annoy索引已卸载")
    def has_collection(self) -> bool:
        """
        检查 Annoy 索引文件和元数据文件是否存在
        """
        # 同时存在索引文件和元数据文件才认为知识库存在
        return os.path.exists(self.index_file) and os.path.exists(self.metadata_file)

    def load_collection(self) -> bool:
        """
        加载集合的兼容接口，内部调用现有的 load_index
        """
        return self.load_index()

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息的兼容接口
        """
        try:
            return {
                "row_count": len(self.metadata) if self.metadata else 0,
                "index_type": f"Annoy (Trees: {self.n_trees})",
                "dimension": self.dimension,
                "metric": self.metric
            }
        except Exception:
            return {"row_count": 0, "error": "无法获取统计信息"}