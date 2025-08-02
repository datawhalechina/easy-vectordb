"""
持久化Milvus连接管理器
项目启动时建立连接，后续直接复用
"""

import logging
import threading
import time
import uuid
from typing import Optional, Dict, Any
from pymilvus import connections, utility, MilvusException
import socket

logger = logging.getLogger(__name__)

class PersistentMilvusConnection:
    """持久化Milvus连接类 - 支持动态配置"""
    
    def __init__(self):
        self._connection_alias: Optional[str] = None
        self._host: Optional[str] = None
        self._port: Optional[int] = None
        self._is_connected = False
        self._lock = threading.Lock()
        self._last_health_check = 0
        self._health_check_interval = 30  # 30秒检查一次
        self._auto_connect = False  # 是否自动连接
        
    def connect(self, host: str, port: int, timeout: int = 10, force_reconnect: bool = False) -> bool:
        """建立持久化连接"""
        with self._lock:
            try:
                # 如果已经连接到相同的服务器且不强制重连，直接返回
                if (not force_reconnect and self._is_connected and 
                    self._host == host and self._port == port):
                    if self._test_connection_health():
                        logger.info(f"复用现有连接: {self._connection_alias}")
                        return True
                    else:
                        logger.warning("现有连接不健康，重新连接")
                        self._disconnect_internal()
                
                # 如果连接参数改变，清理旧连接
                if self._host != host or self._port != port or force_reconnect:
                    logger.info(f"连接参数改变或强制重连: {self._host}:{self._port} → {host}:{port}")
                    self._disconnect_internal()
                
                # 测试网络连通性
                if not self._test_network(host, port):
                    raise ConnectionError(f"网络不通: {host}:{port}")
                
                # 生成新的连接别名
                self._connection_alias = f"persistent_{uuid.uuid4().hex[:8]}"
                
                logger.info(f"建立持久化连接: {host}:{port}")
                
                # 建立连接
                connections.connect(
                    alias=self._connection_alias,
                    host=host,
                    port=port,
                    timeout=timeout
                )
                
                # 验证连接
                collections = utility.list_collections(using=self._connection_alias)
                logger.info(f"连接成功，现有集合: {len(collections)} 个")
                
                # 保存连接信息
                self._host = host
                self._port = port
                self._is_connected = True
                self._last_health_check = time.time()
                
                logger.info(f"✅ 持久化连接建立成功: {self._connection_alias}")
                return True
                
            except Exception as e:
                logger.error(f"❌ 持久化连接建立失败: {e}")
                self._disconnect_internal()
                return False
    
    def _test_network(self, host: str, port: int, timeout: int = 3) -> bool:
        """测试网络连通性"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    def _test_connection_health(self) -> bool:
        """测试连接健康状态"""
        if not self._is_connected or not self._connection_alias:
            return False
        
        # 如果距离上次检查时间太短，直接返回True
        current_time = time.time()
        if current_time - self._last_health_check < self._health_check_interval:
            return True
        
        try:
            # 尝试执行简单操作
            utility.list_collections(using=self._connection_alias)
            self._last_health_check = current_time
            return True
        except Exception as e:
            logger.warning(f"连接健康检查失败: {e}")
            return False
    
    def get_connection_alias(self) -> Optional[str]:
        """获取连接别名"""
        with self._lock:
            if self._is_connected and self._test_connection_health():
                return self._connection_alias
            return None
    
    def is_connected(self) -> bool:
        """检查是否已连接"""
        with self._lock:
            return self._is_connected and self._test_connection_health()
    
    def get_connection_info(self) -> Dict[str, Any]:
        """获取连接信息"""
        with self._lock:
            return {
                "connected": self._is_connected,
                "host": self._host,
                "port": self._port,
                "alias": self._connection_alias,
                "last_health_check": self._last_health_check
            }
    
    def disconnect(self):
        """断开连接"""
        with self._lock:
            self._disconnect_internal()
    
    def _disconnect_internal(self):
        """内部断开连接方法"""
        if self._connection_alias:
            try:
                connections.disconnect(self._connection_alias)
                logger.info(f"断开连接: {self._connection_alias}")
            except Exception as e:
                logger.warning(f"断开连接失败: {e}")
        
        self._connection_alias = None
        self._host = None
        self._port = None
        self._is_connected = False
    
    def reconnect(self) -> bool:
        """重新连接"""
        if self._host and self._port:
            logger.info("尝试重新连接...")
            return self.connect(self._host, self._port, force_reconnect=True)
        return False
    
    def update_connection(self, host: str, port: int) -> bool:
        """更新连接配置（用户配置改变时调用）"""
        logger.info(f"更新连接配置: {host}:{port}")
        return self.connect(host, port, force_reconnect=True)
    
    def is_connection_valid_for(self, host: str, port: int) -> bool:
        """检查当前连接是否适用于指定的主机和端口"""
        with self._lock:
            return (self._is_connected and 
                    self._host == host and 
                    self._port == port and 
                    self._test_connection_health())

# 全局持久化连接实例
_persistent_connection = PersistentMilvusConnection()

def get_persistent_connection() -> PersistentMilvusConnection:
    """获取全局持久化连接实例"""
    return _persistent_connection

def initialize_milvus_connection(host: str, port: int) -> bool:
    """初始化Milvus连接（项目启动时调用）"""
    logger.info("=" * 50)
    logger.info("初始化Milvus持久化连接")
    logger.info("=" * 50)
    
    conn = get_persistent_connection()
    success = conn.connect(host, port)
    
    if success:
        logger.info("🎉 Milvus持久化连接初始化成功!")
    else:
        logger.error("❌ Milvus持久化连接初始化失败!")
    
    return success

def get_milvus_connection() -> Optional[str]:
    """获取可用的Milvus连接别名"""
    conn = get_persistent_connection()
    alias = conn.get_connection_alias()
    
    if not alias:
        logger.warning("持久化连接不可用，尝试重新连接...")
        if conn.reconnect():
            alias = conn.get_connection_alias()
    
    return alias

def check_milvus_connection_status() -> Dict[str, Any]:
    """检查Milvus连接状态"""
    conn = get_persistent_connection()
    return conn.get_connection_info()