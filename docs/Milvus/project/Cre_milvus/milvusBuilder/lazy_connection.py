"""
延迟连接模块
避免在模块导入时就尝试连接Milvus
"""

import logging
from typing import Optional, Dict, Any
import threading

logger = logging.getLogger(__name__)

class LazyMilvusConnection:
    """延迟Milvus连接类，只在真正需要时才建立连接"""
    
    def __init__(self):
        self._connection_manager = None
        self._lock = threading.Lock()
        self._initialized = False
    
    def _get_connection_manager(self):
        """延迟获取连接管理器"""
        if self._connection_manager is None:
            with self._lock:
                if self._connection_manager is None:
                    try:
                        from .connection_manager import get_connection_manager
                        self._connection_manager = get_connection_manager()
                        logger.info("连接管理器已初始化")
                    except Exception as e:
                        logger.error(f"连接管理器初始化失败: {e}")
                        raise
        return self._connection_manager
    
    def get_connection(self, host: str, port: int, timeout: int = 10):
        """获取连接的上下文管理器"""
        logger.info(f"请求连接: {host}:{port}")
        connection_manager = self._get_connection_manager()
        return connection_manager.get_connection(host, port, timeout)
    
    def test_connection(self, host: str, port: int, timeout: int = 5) -> bool:
        """测试连接是否可用，不建立实际连接"""
        try:
            connection_manager = self._get_connection_manager()
            return connection_manager.test_network_connection(host, port, timeout)
        except Exception as e:
            logger.error(f"连接测试失败: {e}")
            return False
    
    def is_available(self) -> bool:
        """检查连接管理器是否可用"""
        try:
            self._get_connection_manager()
            return True
        except:
            return False

# 全局延迟连接实例
_lazy_connection = LazyMilvusConnection()

def get_lazy_connection() -> LazyMilvusConnection:
    """获取全局延迟连接实例"""
    return _lazy_connection

def safe_milvus_operation(func, host: str, port: int, *args, **kwargs):
    """安全执行Milvus操作的装饰器函数"""
    try:
        lazy_conn = get_lazy_connection()
        if not lazy_conn.is_available():
            logger.warning("Milvus连接管理器不可用，跳过操作")
            return None
        
        with lazy_conn.get_connection(host, port) as connection_alias:
            return func(connection_alias, *args, **kwargs)
    
    except Exception as e:
        logger.error(f"Milvus操作失败: {e}")
        return None