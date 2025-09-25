import threading
import time
import logging
import uuid
import socket
from contextlib import contextmanager
from pymilvus import connections, utility
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class MilvusConnectionManager:
    """Milvus连接管理器，确保连接的线程安全和资源管理"""
    
    def __init__(self):
        self._lock = threading.RLock()  # 使用可重入锁
        self._active_connections: Dict[str, Dict[str, Any]] = {}
        self._connection_pool_size = 5
        
    def test_network_connection(self, host: str, port: int, timeout: int = 5) -> bool:
        """测试网络连接是否可用"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, int(port)))
            sock.close()
            return result == 0
        except Exception as e:
            logger.error(f"网络连接测试失败: {e}")
            return False
    
    def cleanup_stale_connections(self):
        """清理所有陈旧连接"""
        try:
            existing_connections = connections.list_connections()
            logger.info(f"发现现有连接: {existing_connections}")
            
            # connections.list_connections() 返回的是 [(alias, connection_info), ...] 格式
            if isinstance(existing_connections, list):
                for conn_item in existing_connections:
                    try:
                        # 提取连接别名
                        if isinstance(conn_item, tuple) and len(conn_item) >= 1:
                            conn_alias = conn_item[0]
                        else:
                            conn_alias = conn_item
                        
                        connections.disconnect(conn_alias)
                        logger.info(f"清理陈旧连接: {conn_alias}")
                    except Exception as e:
                        logger.warning(f"清理连接 {conn_item} 失败: {e}")
            
            # 清理内部记录
            self._active_connections.clear()
            time.sleep(0.5)  # 等待连接完全关闭
            
        except Exception as e:
            logger.warning(f"清理连接时出错: {e}")
    
    def create_connection(self, host: str, port: int, timeout: int = 10, max_retries: int = 3) -> str:
        """创建新的Milvus连接，返回连接别名"""
        import traceback
        
        # 添加调用栈信息，帮助调试
        stack_info = traceback.format_stack()
        logger.info(f"连接请求来源: {stack_info[-3].strip()}")
        
        with self._lock:
            # 生成唯一连接别名
            connection_alias = f"conn_{uuid.uuid4().hex[:8]}"
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"创建Milvus连接 (第{attempt + 1}次): {host}:{port}")
                    
                    # 网络连通性测试 - 使用更短的超时时间
                    logger.info(f"测试网络连通性: {host}:{port}")
                    if not self.test_network_connection(host, port, timeout=3):
                        raise ConnectionError(f"网络连接失败: {host}:{port}")
                    logger.info(f"网络连通性测试通过: {host}:{port}")
                    
                    # 如果是第一次尝试失败，清理所有连接
                    if attempt > 0:
                        logger.info("清理陈旧连接...")
                        self.cleanup_stale_connections()
                    
                    # 创建连接
                    logger.info(f"正在建立Milvus连接: {connection_alias}")
                    connections.connect(
                        alias=connection_alias,
                        host=host,
                        port=int(port),
                        timeout=timeout
                    )
                    logger.info(f"Milvus连接已建立: {connection_alias}")
                    
                    # 验证连接 - 使用指定的连接别名
                    try:
                        logger.info(f"验证连接: {connection_alias}")
                        collections = utility.list_collections(using=connection_alias)
                        server_version = utility.get_server_version(using=connection_alias)
                        logger.info(f"连接成功 [{connection_alias}]: 版本={server_version}, 集合={len(collections)}")
                        
                        # 记录连接信息
                        self._active_connections[connection_alias] = {
                            "host": host,
                            "port": port,
                            "created_at": time.time(),
                            "last_used": time.time()
                        }
                        
                        return connection_alias
                        
                    except Exception as verify_error:
                        logger.error(f"连接验证失败: {verify_error}")
                        try:
                            connections.disconnect(connection_alias)
                        except:
                            pass
                        raise verify_error
                
                except Exception as e:
                    logger.warning(f"连接尝试 {attempt + 1} 失败: {e}")
                    if attempt < max_retries - 1:
                        wait_time = min((attempt + 1) * 2, 10)  # 最大等待10秒
                        logger.info(f"等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                    else:
                        raise ConnectionError(f"经过 {max_retries} 次尝试后仍无法连接: {e}")
    
    def release_connection(self, connection_alias: str):
        """释放连接"""
        with self._lock:
            try:
                if connection_alias in connections.list_connections():
                    connections.disconnect(connection_alias)
                    logger.info(f"释放连接: {connection_alias}")
                
                # 从记录中移除
                self._active_connections.pop(connection_alias, None)
                
            except Exception as e:
                logger.warning(f"释放连接失败 {connection_alias}: {e}")
    
    @contextmanager
    def get_connection(self, host: str, port: int, timeout: int = 10):
        """上下文管理器，自动管理连接生命周期"""
        connection_alias = None
        try:
            connection_alias = self.create_connection(host, port, timeout)
            yield connection_alias
        finally:
            if connection_alias:
                self.release_connection(connection_alias)
    
    def get_connection_status(self) -> Dict[str, Any]:
        """获取连接状态信息"""
        with self._lock:
            return {
                "active_connections": len(self._active_connections),
                "connection_details": dict(self._active_connections),
                "pymilvus_connections": connections.list_connections()
            }
    
    def force_cleanup_all(self):
        """强制清理所有连接（紧急情况使用）"""
        with self._lock:
            logger.warning("执行强制连接清理...")
            self.cleanup_stale_connections()
            
            # 额外的清理步骤
            try:
                import gc
                gc.collect()  # 强制垃圾回收
                time.sleep(1)
            except:
                pass

# 全局连接管理器实例
_connection_manager = MilvusConnectionManager()

def get_connection_manager() -> MilvusConnectionManager:
    """获取全局连接管理器实例"""
    return _connection_manager