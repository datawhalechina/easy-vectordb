"""
连接初始化器
项目启动时初始化所有必要的连接
"""

import logging
import yaml
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConnectionInitializer:
    """连接初始化器"""
    
    def __init__(self):
        self.config: Optional[Dict[str, Any]] = None
        self.milvus_initialized = False
        self.redis_initialized = False
    
    def load_config(self, config_path: str = "config.yaml") -> bool:
        """加载配置文件"""
        try:
            if not os.path.exists(config_path):
                logger.error(f"配置文件不存在: {config_path}")
                return False
            
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
            
            logger.info("✅ 配置文件加载成功")
            return True
            
        except Exception as e:
            logger.error(f"❌ 配置文件加载失败: {e}")
            return False
    
    def initialize_milvus(self) -> bool:
        """初始化Milvus连接管理器（不立即连接）"""
        if not self.config:
            logger.error("配置未加载，无法初始化Milvus")
            return False
        
        try:
            # 只初始化连接管理器，不立即连接
            from milvusBuilder.persistent_connection import get_persistent_connection
            conn = get_persistent_connection()
            
            # 检查配置中是否有Milvus配置
            milvus_config = self.config.get("milvus", {})
            if milvus_config.get("host") and milvus_config.get("port"):
                host = milvus_config.get("host", "127.0.0.1")
                port = int(milvus_config.get("port", "19530"))
                
                logger.info(f"尝试使用配置文件中的Milvus地址: {host}:{port}")
                
                # 尝试连接，但失败不影响初始化
                success = conn.connect(host, port)
                if success:
                    logger.info("✅ 使用配置文件连接Milvus成功")
                else:
                    logger.warning("⚠️ 配置文件中的Milvus连接失败，等待用户配置")
            else:
                logger.info("📝 配置文件中无Milvus配置，等待用户在前端配置")
            
            self.milvus_initialized = True
            logger.info("✅ Milvus连接管理器初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"❌ Milvus初始化异常: {e}")
            return False
    
    def initialize_redis(self) -> bool:
        """初始化Redis连接（如果需要）"""
        # 这里可以添加Redis连接初始化逻辑
        # 目前先返回True
        self.redis_initialized = True
        logger.info("✅ Redis连接初始化成功（跳过）")
        return True
    
    def initialize_all(self, config_path: str = "config.yaml") -> bool:
        """初始化所有连接"""
        logger.info("=" * 60)
        logger.info("开始初始化所有连接")
        logger.info("=" * 60)
        
        # 1. 加载配置
        if not self.load_config(config_path):
            return False
        
        # 2. 初始化Milvus
        if not self.initialize_milvus():
            return False
        
        # 3. 初始化Redis（可选）
        if not self.initialize_redis():
            logger.warning("Redis初始化失败，但继续执行")
        
        logger.info("=" * 60)
        logger.info("🎉 所有连接初始化完成!")
        logger.info("=" * 60)
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """获取初始化状态"""
        return {
            "config_loaded": self.config is not None,
            "milvus_initialized": self.milvus_initialized,
            "redis_initialized": self.redis_initialized
        }
    
    def get_milvus_config(self) -> Dict[str, Any]:
        """获取Milvus配置"""
        if not self.config:
            return {}
        return self.config.get("milvus", {})
    
    def get_system_config(self) -> Dict[str, Any]:
        """获取系统配置"""
        if not self.config:
            return {}
        return self.config.get("system", {})

# 全局初始化器实例
_initializer = ConnectionInitializer()

def get_initializer() -> ConnectionInitializer:
    """获取全局初始化器实例"""
    return _initializer

def startup_initialize(config_path: str = "config.yaml") -> bool:
    """启动时初始化（项目启动时调用）"""
    initializer = get_initializer()
    return initializer.initialize_all(config_path)

def get_milvus_config() -> Dict[str, Any]:
    """获取Milvus配置"""
    initializer = get_initializer()
    return initializer.get_milvus_config()

def get_system_config() -> Dict[str, Any]:
    """获取系统配置"""
    initializer = get_initializer()
    return initializer.get_system_config()

def is_initialized() -> bool:
    """检查是否已初始化"""
    initializer = get_initializer()
    status = initializer.get_status()
    return status["config_loaded"] and status["milvus_initialized"]