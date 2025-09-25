"""
Locust性能测试模块

从locustProj项目移植并集成到主系统中
"""

import time
import random
import logging
import numpy as np
from typing import Dict, Any, Optional
from locust import User, task, between, events
from pymilvus import connections, Collection, utility
import threading

logger = logging.getLogger(__name__)

# 全局连接管理
_connection_initialized = False
_shared_connection = "default"
_shared_collection = None
_shared_dimension = 256
_connection_lock = threading.Lock()


def init_shared_connection(host: str = "localhost", port: str = "19530", 
                          collection_name: str = "locust_test_collection"):
    """
    初始化共享的Milvus连接
    
    参数:
        host: Milvus主机地址
        port: Milvus端口
        collection_name: 测试集合名称
    """
    global _connection_initialized, _shared_collection, _shared_dimension
    
    with _connection_lock:
        if _connection_initialized:
            return
        
        try:
            # 连接到Milvus
            connections.connect(
                alias=_shared_connection,
                host=host,
                port=port,
                timeout=10
            )
            
            # 检查集合是否存在
            if not utility.has_collection(collection_name, using=_shared_connection):
                raise Exception(f"集合 '{collection_name}' 不存在，请先创建测试数据")
            
            # 获取集合对象
            logger.info(f"创建Collection对象: name={collection_name}, using={_shared_connection}")
            print(f"创建Collection对象: name={collection_name}, using={_shared_connection}")
            
            # 验证连接别名是否有效
            try:
                utility.list_collections(using=_shared_connection)
                logger.info(f"✅ 连接别名验证通过: {_shared_connection}")
                print(f"✅ 连接别名验证通过: {_shared_connection}")
            except Exception as e:
                logger.error(f"❌ 连接别名验证失败: {_shared_connection}, 错误: {e}")
                print(f"❌ 连接别名验证失败: {_shared_connection}, 错误: {e}")
            
            _shared_collection = Collection(collection_name, using=_shared_connection)
            
            # 获取向量维度
            schema = _shared_collection.schema
            for field in schema.fields:
                if field.name == "vector":
                    _shared_dimension = field.params.get('dim', 256)
                    break
            
            _connection_initialized = True
            logger.info(f"Milvus连接成功！向量维度: {_shared_dimension}")
            
        except Exception as e:
            logger.error(f"Milvus连接失败: {e}")
            raise


class MilvusLoadTest:
    """
    Milvus负载测试管理器
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化负载测试
        
        参数:
            config: 测试配置
        """
        self.config = config
        self.host = config.get('host', 'localhost')
        self.port = config.get('port', '19530')
        self.collection_name = config.get('collection_name', 'locust_test_collection')
        
    def start_test(self, users: int = 5, spawn_rate: float = 1.0, 
                   run_time: str = "60s") -> Dict[str, Any]:
        """
        启动性能测试
        
        参数:
            users: 并发用户数
            spawn_rate: 用户启动速率
            run_time: 运行时间
        
        返回:
            测试结果
        """
        try:
            # 初始化连接
            init_shared_connection(self.host, self.port, self.collection_name)
            
            # 这里应该启动Locust测试
            # 由于Locust通常作为独立进程运行，这里返回配置信息
            return {
                "status": "started",
                "config": {
                    "users": users,
                    "spawn_rate": spawn_rate,
                    "run_time": run_time,
                    "host": self.host,
                    "port": self.port,
                    "collection": self.collection_name
                }
            }
            
        except Exception as e:
            logger.error(f"启动测试失败: {e}")
            return {"status": "error", "message": str(e)}


class MilvusUser(User):
    """
    Milvus用户行为模拟类
    """
    
    wait_time = between(0.5, 2.0)  # 操作间隔时间
    
    def on_start(self):
        """用户启动时的初始化"""
        init_shared_connection()
        self.collection = _shared_collection
        self.dimension = _shared_dimension
    
    def generate_random_vector(self):
        """生成随机向量"""
        vector = np.random.normal(0, 1, self.dimension).astype(np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()
    
    @task(10)
    def single_vector_search(self):
        """单向量搜索（最常见的操作）"""
        self._perform_search("single_search", batch_size=1, top_k=10, nprobe=16)
    
    @task(5)
    def batch_vector_search(self):
        """批量向量搜索"""
        batch_size = random.randint(2, 5)
        self._perform_search("batch_search", batch_size=batch_size, top_k=10, nprobe=16)
    
    @task(3)
    def high_precision_search(self):
        """高精度搜索"""
        self._perform_search("high_precision_search", batch_size=1, top_k=50, nprobe=32)
    
    @task(2)
    def fast_search(self):
        """快速搜索"""
        self._perform_search("fast_search", batch_size=1, top_k=5, nprobe=8)
    
    def _perform_search(self, name: str, batch_size: int = 1, top_k: int = 10, nprobe: int = 16):
        """
        执行搜索操作
        
        参数:
            name: 操作名称
            batch_size: 批量大小
            top_k: 返回结果数量
            nprobe: 搜索参数
        """
        start_time = time.time()
        exception = None
        result_count = 0
        
        try:
            # 生成查询向量
            query_vectors = []
            for _ in range(batch_size):
                query_vectors.append(self.generate_random_vector())
            
            # 搜索参数
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": nprobe}
            }
            
            # 执行搜索
            results = self.collection.search(
                data=query_vectors,
                anns_field="vector",
                param=search_params,
                limit=top_k,
                timeout=30
            )
            
            # 统计结果数量
            result_count = sum(len(result) for result in results if result is not None)
            
        except Exception as e:
            exception = e
            logger.error(f"搜索失败: {e}")
        
        # 计算响应时间
        response_time = (time.time() - start_time) * 1000
        
        # 记录测试结果
        events.request.fire(
            request_type="SEARCH",
            name=name,
            response_time=response_time,
            response_length=result_count,
            exception=exception
        )


# Locust事件监听器
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """测试开始时的准备工作"""
    logger.info("=" * 50)
    logger.info("🚀 Milvus性能测试开始")
    logger.info(f"目标: localhost:19530")
    logger.info("=" * 50)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """测试结束时的清理工作"""
    if environment.stats and environment.stats.total:
        total = environment.stats.total
        logger.info(f"总请求数: {total.num_requests}")
        logger.info(f"失败请求数: {total.num_failures}")
        logger.info(f"平均响应时间: {total.avg_response_time:.2f}ms")
    
    # 断开连接
    if _connection_initialized and connections.has_connection(_shared_connection):
        connections.disconnect(_shared_connection)