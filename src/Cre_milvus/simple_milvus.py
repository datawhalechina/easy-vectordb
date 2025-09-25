import logging
import time
import socket
from typing import Optional, Dict, Any
from pymilvus import connections, utility, MilvusException
from datetime import datetime

logger = logging.getLogger(__name__)

class SimpleMilvusConnection:
    """简化的Milvus连接类"""
    
    def __init__(self):
        self.connection_alias: Optional[str] = None
        self.host: Optional[str] = None
        self.port: Optional[int] = None
        self.use_lite: bool = False
        self.connected: bool = False
        self.connection_time: Optional[datetime] = None
        self.error_message: Optional[str] = None
    
    def connect(self, host: str, port: int, use_lite: bool = False, timeout: int = 10) -> bool:
        """建立Milvus连接"""
        try:
            print(f"🔧 开始连接Milvus: host={host}, port={port}, use_lite={use_lite}")
            logger.info(f"🔧 开始连接Milvus: host={host}, port={port}, use_lite={use_lite}")
            
            # 清理现有连接
            self._cleanup_connection()
            
            # 生成基于日期的连接别名（同一天内复用）
            from datetime import datetime
            date_str = datetime.now().strftime("%Y%m%d")
            alias = f"milvus_{date_str}"
            print(f"🔧 生成的连接别名: {alias}")
            print(f"🔧 连接参数: host={host}, port={port}, use_lite={use_lite}")
            logger.info(f"🔧 生成的连接别名: {alias}")
            logger.info(f"🔧 连接参数: host={host}, port={port}, use_lite={use_lite}")
            
            # 记录别名生成的详细信息
            logger.info(f"🔧 别名生成详情: date_str={date_str}, host={host}, port={port}")
            print(f"🔧 别名生成详情: date_str={date_str}, host={host}, port={port}")
            
            if use_lite and host.endswith('.db'):
                # 如果配置为使用lite但没有安装milvus-lite，则跳过
                logger.warning("⚠️ Milvus Lite未安装，跳过lite连接")
                self.error_message = "Milvus Lite未安装"
                return False
            else:
                # 使用标准Milvus连接
                if not self._test_network_connection(host, port, timeout):
                    logger.error(f"❌ 无法连接到 {host}:{port}")
                    self.error_message = f"无法连接到 {host}:{port}"
                    return False
                
                # 方案1：检查连接是否已存在
                connection_exists = False
                # try:
                #     print(f"🔍 检查连接 {alias} 是否已存在...")
                #     utility.list_collections(using=alias)
                #     connection_exists = True
                #     print(f"✅ 连接 {alias} 已存在，直接使用")
                #     logger.info(f"连接 {alias} 已存在，直接复用")
                # except Exception as e:
                #     print(f"🔧 连接 {alias} 不存在，需要创建新连接: {e}")
                #     logger.debug(f"连接检查失败: {e}")
                
                # 如果连接不存在，创建新连接
                if not connection_exists:
                    print(f"🔧 开始建立Milvus连接，使用别名: {alias}")
                    logger.info(f"🔧 开始建立Milvus连接，使用别名: {alias}")
                    import threading
                    from queue import Queue
                    
                    # 创建线程安全的结果队列
                    result_queue = Queue()
                    
                    def connect_task():
                        try:
                            conn = connections.connect(
                                alias=alias,
                                host=host,
                                port=port,
                                timeout=timeout
                            )
                            result_queue.put(('success', alias))
                        except Exception as e:
                            result_queue.put(('error', str(e)))
                    
                    # 使用官方建议的连接管理方式
                    if not connections.has_connection(alias="default"):
                        connections.add_connection(
                            default={"host": host, "port": port}
                        )
                    connections.connect(alias="default")
                    alias = "default"
                    
                    # 等待连接结果(最大等待5秒)
                    conn_thread.join(5)
                    if result_queue.empty():
                        raise TimeoutError(f"Milvus连接超时: {host}:{port}")
                    
                    status, msg = result_queue.get()
                    if status == 'error':
                        raise ConnectionError(msg)
                    print(f"🔧 新连接创建完成: {alias}")
                    logger.info(f"🔧 新连接创建完成: {alias}")
                else:
                    print(f"🔧 复用现有连接: {alias}")
                    logger.info(f"🔧 复用现有连接: {alias}")
            
            # 增强连接验证
            try:
                utility.list_collections(using=alias)
                logger.info(f"✅ 连接验证成功: {alias}")
            except Exception as e:
                logger.error(f"❌ 连接验证失败: {e}")
                raise ConnectionError(f"Milvus连接失败: {str(e)}")
            
            # 更新连接信息
            self.connection_alias = alias
            self.host = host
            self.port = port
            self.use_lite = use_lite
            self.connected = True
            self.connection_time = datetime.now()
            self.error_message = None
            
            logger.info(f"✅ Milvus连接成功: {host}:{port}")
            print(f"✅ Milvus连接成功，当前别名: {alias}")
            return True
            
        except Exception as e:
            error_msg = f"Milvus连接失败: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.error_message = error_msg
            self.connected = False
            self._cleanup_connection()
            return False
    
    def _test_network_connection(self, host: str, port: int, timeout: int = 3) -> bool:
        """测试网络连通性"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception as e:
            logger.warning(f"网络连通性测试失败: {e}")
            return False
    
    def _cleanup_connection(self) -> None:
        """清理现有连接"""
        if self.connection_alias:
            try:
                connections.disconnect(self.connection_alias)
                logger.debug(f"已断开连接: {self.connection_alias}")
            except Exception as e:
                logger.warning(f"断开连接失败: {e}")
        
        self.connection_alias = None
        self.connected = False
    
    def get_connection_alias(self) -> Optional[str]:
        """获取连接别名"""
        print(f"🔍 获取连接别名 - 当前状态: connected={self.connected}, alias={self.connection_alias}")
        logger.info(f"🔍 获取连接别名 - 当前状态: connected={self.connected}, alias={self.connection_alias}")
        
        if self.connected and self.connection_alias:
            # 简单的连接有效性检查
            print(f"🔍 检查连接有效性: {self.connection_alias}")
            logger.info(f"🔍 检查连接有效性: {self.connection_alias}")
            is_valid = self._is_connection_valid()
            print(f"🔍 连接有效性检查结果: {is_valid}")
            logger.info(f"🔍 连接有效性检查结果: {is_valid}")
            if is_valid:
                print(f"✅ 连接有效，返回别名: {self.connection_alias}")
                logger.info(f"✅ 连接有效，返回别名: {self.connection_alias}")
                return self.connection_alias
            else:
                logger.warning("连接已失效")
                print(f"❌ 连接已失效，别名: {self.connection_alias}")
                logger.warning(f"❌ 连接已失效，别名: {self.connection_alias}")
                self.connected = False
                return None
        
        print(f"❌ 无有效连接，返回None")
        logger.info(f"❌ 无有效连接，返回None")
        return None
    
    def _is_connection_valid(self) -> bool:
        """检查连接是否有效"""
        if not self.connection_alias:
            return False
        
        try:
            # 尝试执行简单操作来验证连接
            utility.list_collections(using=self.connection_alias)
            return True
        except Exception as e:
            logger.warning(f"连接验证失败: {e}")
            return False
    
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self.connected and self._is_connection_valid()
    
    def get_status(self) -> Dict[str, Any]:
        """获取连接状态"""
        return {
            "connected": self.connected,
            "host": self.host,
            "port": self.port,
            "use_lite": self.use_lite,
            "connection_alias": self.connection_alias,
            "connection_time": self.connection_time.isoformat() if self.connection_time else None,
            "error_message": self.error_message,
            "valid": self._is_connection_valid() if self.connected else False
        }
    
    def disconnect(self) -> None:
        """断开连接"""
        logger.info("🔌 断开Milvus连接")
        self._cleanup_connection()
        self.host = None
        self.port = None
        self.use_lite = False
        self.connection_time = None
        self.error_message = None

# 全局连接实例
_milvus_connection: Optional[SimpleMilvusConnection] = None

def get_milvus_connection() -> SimpleMilvusConnection:
    """获取全局Milvus连接实例"""
    global _milvus_connection
    if _milvus_connection is None:
        _milvus_connection = SimpleMilvusConnection()
    return _milvus_connection

def connect_milvus(host: str, port: int, use_lite: bool = False, timeout: int = 10) -> bool:
    """连接到Milvus"""
    conn = get_milvus_connection()
    return conn.connect(host, port, use_lite, timeout)

def get_milvus_alias() -> Optional[str]:
    """获取Milvus连接别名"""
    conn = get_milvus_connection()
    return conn.get_connection_alias()

def is_milvus_connected() -> bool:
    """检查Milvus是否已连接"""
    conn = get_milvus_connection()
    return conn.is_connected()

def get_milvus_status() -> Dict[str, Any]:
    """获取Milvus连接状态"""
    conn = get_milvus_connection()
    return conn.get_status()

def disconnect_milvus() -> None:
    """断开Milvus连接"""
    conn = get_milvus_connection()
    conn.disconnect()

def test_milvus_connection(host: str, port: int, use_lite: bool = False) -> bool:
    """测试Milvus连接（不保存连接状态）"""
    try:
        test_alias = f"test_connection_{int(time.time())}"
        
        if use_lite or host.endswith('.db'):
            connections.connect(alias=test_alias, uri=host)
        else:
            # 先测试网络
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result != 0:
                return False
            
            connections.connect(alias=test_alias, host=host, port=port, timeout=10)
        
        # 验证连接
        utility.list_collections(using=test_alias)
        
        # 清理测试连接
        connections.disconnect(test_alias)
        
        return True
        
    except Exception as e:
        logger.debug(f"连接测试失败: {e}")
        try:
            connections.disconnect(test_alias)
        except:
            pass
        return False

def initialize_milvus_from_config(config: Dict[str, Any]) -> bool:
    """从配置初始化Milvus连接"""
    try:
        milvus_cfg = config.get("milvus", {})
        host = milvus_cfg.get("host", "./milvus_lite.db")
        port = int(milvus_cfg.get("port", 19530))
        use_lite = milvus_cfg.get("use_lite", False) or host.endswith('.db')
        
        logger.info(f"🚀 从配置初始化Milvus连接: {host}:{port} (Lite: {use_lite})")
        
        success = connect_milvus(host, port, use_lite)
        if success:
            logger.info("✅ Milvus连接初始化成功")
        else:
            logger.error("❌ Milvus连接初始化失败")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Milvus连接初始化异常: {e}")
        return False

def ensure_milvus_connection(config: Dict[str, Any]) -> Optional[str]:
    """确保Milvus连接可用，返回连接别名"""
    try:
        # 检查现有连接
        alias = get_milvus_alias()
        if alias:
            logger.debug("✅ 使用现有Milvus连接")
            return alias
        
        # 尝试重新连接
        logger.info("🔄 Milvus连接不可用，尝试重新连接")
        if initialize_milvus_from_config(config):
            return get_milvus_alias()
        
        logger.error("❌ 无法建立Milvus连接")
        return None
        
    except Exception as e:
        logger.error(f"❌ 确保Milvus连接时出错: {e}")
        return None

def update_milvus_connection(host: str, port: int, use_lite: bool = False) -> bool:
    """更新Milvus连接配置"""
    try:
        logger.info(f"🔄 更新Milvus连接: {host}:{port} (Lite: {use_lite})")
        
        # 断开现有连接
        disconnect_milvus()
        
        # 建立新连接
        success = connect_milvus(host, port, use_lite)
        
        if success:
            logger.info("✅ Milvus连接更新成功")
        else:
            logger.error("❌ Milvus连接更新失败")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ 更新Milvus连接时出错: {e}")
        return False

# 兼容性函数 - 与现有代码保持兼容
def get_persistent_connection():
    """兼容性函数：返回简化的连接对象"""
    return get_milvus_connection()

def check_milvus_connection_status() -> Dict[str, Any]:
    """兼容性函数：检查Milvus连接状态"""
    return get_milvus_status()

def get_milvus_connection_alias() -> Optional[str]:
    """兼容性函数：获取连接别名"""
    return get_milvus_alias()