#!/usr/bin/env python3
"""
Milvus Locust性能测试
针对本地Milvus (localhost:19530) 的简洁测试工具
"""

import time
import random
import numpy as np
import threading
import warnings
import os
from locust import User, task, between, events
from pymilvus import connections, Collection, utility
import logging

# 配置日志，抑制grpc的警告信息
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 抑制grpc相关的警告和异常信息
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""

# 抑制pymilvus的调试信息
logging.getLogger("pymilvus").setLevel(logging.WARNING)
logging.getLogger("grpc").setLevel(logging.ERROR)

# 全局连接管理
_connection_lock = threading.Lock()
_shared_connection = "milvus_shared"
_connection_initialized = False
_shared_collection = None
_shared_dimension = 256

def init_shared_connection():
    """初始化共享连接"""
    global _connection_initialized, _shared_collection, _shared_dimension
    
    with _connection_lock:
        if _connection_initialized:
            return
        
        try:
            # 如果连接已存在，先断开
            if connections.has_connection(_shared_connection):
                try:
                    connections.disconnect(_shared_connection)
                except:
                    pass
            
            # 创建共享连接，使用最简单的配置
            connections.connect(
                alias=_shared_connection,
                host="localhost",
                port="19530",
                timeout=10
            )
            
            # 检查集合是否存在
            if not utility.has_collection("locust_test_collection", using=_shared_connection):
                raise Exception("集合 'locust_test_collection' 不存在，请先创建测试数据")
            
            # 创建共享集合对象
            _shared_collection = Collection("locust_test_collection", using=_shared_connection)
            
            # 获取向量维度
            try:
                schema = _shared_collection.schema
                for field in schema.fields:
                    if field.name == "vector":
                        _shared_dimension = field.params.get('dim', 256)
                        break
            except Exception as e:
                logger.warning(f"获取维度失败，使用默认值256: {e}")
            
            _connection_initialized = True
            logger.info(f"共享连接初始化成功，向量维度: {_shared_dimension}")
            
        except Exception as e:
            logger.error(f"共享连接初始化失败: {e}")
            _connection_initialized = False
            raise

class MilvusUser(User):
    """Milvus用户行为模拟"""
    
    wait_time = between(0.5, 2.0)
    
    def on_start(self):
        """用户启动时初始化"""
        try:
            # 确保共享连接已初始化
            init_shared_connection()
            
            # 使用共享连接和集合
            self.connection_alias = _shared_connection
            self.collection = _shared_collection
            self.dimension = _shared_dimension
            
            logger.info(f"用户 {id(self)} 初始化成功，使用共享连接")
            
        except Exception as e:
            logger.error(f"用户初始化失败: {e}")
            raise
    
    def on_stop(self):
        """用户停止时的清理（共享连接不需要单独清理）"""
        logger.debug(f"用户 {id(self)} 停止")
    
    def generate_random_vector(self):
        """生成随机查询向量"""
        vector = np.random.normal(0, 1, self.dimension).astype(np.float32)
        # 归一化向量
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()
    
    @task(10)
    def single_vector_search(self):
        """单向量搜索 - 最常见的操作"""
        self._perform_search("single_search", batch_size=1, top_k=10, nprobe=16)
    
    @task(5)
    def batch_vector_search(self):
        """批量向量搜索"""
        batch_size = random.randint(2, 5)
        self._perform_search("batch_search", batch_size=batch_size, top_k=10, nprobe=16)
    
    @task(3)
    def high_precision_search(self):
        """高精度搜索 - 更多结果"""
        self._perform_search("high_precision", batch_size=1, top_k=50, nprobe=32)
    
    @task(2)
    def fast_search(self):
        """快速搜索 - 较少结果"""
        self._perform_search("fast_search", batch_size=1, top_k=5, nprobe=8)
    
    def _perform_search(self, name, batch_size=1, top_k=10, nprobe=16):
        """执行搜索操作的核心方法"""
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
                output_fields=[],
                timeout=30
            )
            
            # 统计结果数量
            if results:
                result_count = sum(len(result) for result in results if result is not None)
            
        except Exception as e:
            exception = e
            logger.error(f"搜索操作失败 [{name}]: {str(e)}")
        
        # 计算响应时间并记录指标
        response_time = (time.time() - start_time) * 1000
        
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
    """测试开始时的回调"""
    logger.info("=" * 50)
    logger.info("🚀 Milvus性能测试开始")
    logger.info(f"目标: localhost:19530/default")
    logger.info(f"集合: locust_test_collection")
    logger.info("=" * 50)

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """测试结束时的回调"""
    global _connection_initialized, _shared_connection
    
    logger.info("=" * 50)
    logger.info("📈 Milvus性能测试结束")
    
    if environment.stats and environment.stats.total:
        total = environment.stats.total
        logger.info(f"总请求数: {total.num_requests}")
        logger.info(f"失败请求数: {total.num_failures}")
        logger.info(f"平均响应时间: {total.avg_response_time:.2f}ms")
        logger.info(f"最大响应时间: {total.max_response_time:.2f}ms")
        logger.info(f"请求速率: {total.total_rps:.2f} RPS")
        
        if total.num_requests > 0:
            success_rate = (total.num_requests - total.num_failures) / total.num_requests * 100
            logger.info(f"成功率: {success_rate:.2f}%")
    
    # 清理共享连接
    try:
        if _connection_initialized and connections.has_connection(_shared_connection):
            connections.disconnect(_shared_connection)
            logger.info("共享连接已清理")
    except Exception as e:
        logger.warning(f"清理连接时出错: {e}")
    
    logger.info("=" * 50)

@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    """程序退出时的清理"""
    global _connection_initialized, _shared_connection
    
    try:
        if _connection_initialized and connections.has_connection(_shared_connection):
            connections.disconnect(_shared_connection)
    except:
        pass