"""
简化的持久化Milvus连接管理器
移除复杂的监控和状态管理，仅保留基本连接功能
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def get_milvus_connection() -> Optional[str]:
    """获取可用的Milvus连接别名（直接调用简化组件）"""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from simple_milvus import get_milvus_alias
    return get_milvus_alias()

def get_persistent_connection():
    """获取持久化连接实例（兼容性函数，直接返回简化连接）"""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from simple_milvus import get_milvus_connection
    return get_milvus_connection()

def initialize_milvus_connection(host: str, port: int) -> bool:
    """初始化Milvus连接（兼容性函数）"""
    logger.info("初始化Milvus连接（使用简化组件）")
    
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from simple_milvus import connect_milvus
    use_lite = host.endswith('.db')
    success = connect_milvus(host, port, use_lite)
    
    if success:
        logger.info("🎉 Milvus连接初始化成功!")
    else:
        logger.error("❌ Milvus连接初始化失败!")
    
    return success

def check_milvus_connection_status() -> Dict[str, Any]:
    """检查Milvus连接状态（兼容性函数）"""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from simple_milvus import get_milvus_status
    return get_milvus_status()