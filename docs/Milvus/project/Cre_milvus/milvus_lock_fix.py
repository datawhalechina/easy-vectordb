"""
Milvus数据库文件锁定问题的解决方案
"""

import os
import time
import logging
from pathlib import Path
from pymilvus import connections

logger = logging.getLogger(__name__)

def fix_milvus_lite_lock_issue(db_path: str = "./milvus_lite.db") -> str:
    """
    修复Milvus Lite数据库文件锁定问题
    返回可用的数据库文件路径
    """
    try:
        # 方案1: 检查文件是否被占用
        if os.path.exists(db_path):
            logger.info(f"检查数据库文件: {db_path}")
            
            # 尝试重命名文件来检查是否被占用
            try:
                temp_name = db_path + ".temp_check"
                os.rename(db_path, temp_name)
                os.rename(temp_name, db_path)
                logger.info("✅ 数据库文件可用")
                return db_path
            except OSError as e:
                logger.warning(f"⚠️ 数据库文件被占用: {e}")
                
                # 方案2: 使用带时间戳的新文件
                timestamp = int(time.time())
                new_db_path = f"./milvus_lite_{timestamp}.db"
                logger.info(f"🔄 使用新的数据库文件: {new_db_path}")
                return new_db_path
        else:
            logger.info("数据库文件不存在，将创建新文件")
            return db_path
            
    except Exception as e:
        logger.error(f"修复数据库锁定问题时出错: {e}")
        # 方案3: 使用临时目录
        import tempfile
        temp_dir = tempfile.gettempdir()
        temp_db_path = os.path.join(temp_dir, f"milvus_lite_{int(time.time())}.db")
        logger.info(f"🔄 使用临时数据库文件: {temp_db_path}")
        return temp_db_path

def cleanup_old_connections():
    """清理所有现有的Milvus连接"""
    try:
        # 获取所有连接别名
        connection_names = connections.list_connections()
        
        for name in connection_names:
            try:
                # 处理连接名称可能是元组的情况
                if isinstance(name, tuple):
                    alias = name[0]  # 取元组的第一个元素
                else:
                    alias = name
                
                connections.disconnect(alias)
                logger.info(f"✅ 已断开连接: {alias}")
            except Exception as e:
                logger.debug(f"断开连接 {name} 时出错: {e}")
        
        logger.info("🧹 连接清理完成")
        
    except Exception as e:
        logger.warning(f"清理连接时出错: {e}")

def safe_milvus_connect(host: str, port: int, use_lite: bool = False, max_retries: int = 3) -> tuple[bool, str]:
    """
    安全的Milvus连接，自动处理文件锁定问题
    返回 (成功状态, 实际使用的数据库路径)
    """
    
    # 先清理现有连接
    cleanup_old_connections()
    
    original_host = host
    
    for attempt in range(max_retries):
        try:
            logger.info(f"🔄 尝试连接 (第{attempt + 1}次): {host}:{port}")
            
            if use_lite or host.endswith('.db'):
                # 处理Milvus Lite文件锁定问题
                if attempt > 0:  # 第一次失败后使用修复方案
                    host = fix_milvus_lite_lock_issue(original_host)
                
                # 生成唯一的连接别名
                alias = f"default_{int(time.time())}_{attempt}"
                
                connections.connect(
                    alias=alias,
                    uri=host
                )
                
                # 测试连接
                from pymilvus import utility
                collections = utility.list_collections(using=alias)
                
                logger.info(f"✅ Milvus Lite连接成功: {host}")
                return True, host
                
            else:
                # 标准Milvus连接
                alias = f"default_{int(time.time())}_{attempt}"
                connections.connect(
                    alias=alias,
                    host=host,
                    port=port
                )
                
                from pymilvus import utility
                collections = utility.list_collections(using=alias)
                
                logger.info(f"✅ 标准Milvus连接成功: {host}:{port}")
                return True, host
                
        except Exception as e:
            logger.warning(f"❌ 连接尝试 {attempt + 1} 失败: {e}")
            
            if attempt < max_retries - 1:
                # 等待一段时间后重试
                time.sleep(1)
            else:
                logger.error(f"❌ 所有连接尝试都失败了")
    
    return False, original_host

def get_available_db_path(base_path: str = "./milvus_lite.db") -> str:
    """获取可用的数据库文件路径"""
    return fix_milvus_lite_lock_issue(base_path)

# 使用示例
if __name__ == "__main__":
    # 测试修复功能
    success, db_path = safe_milvus_connect("./milvus_lite.db", 19530, use_lite=True)
    if success:
        print(f"✅ 连接成功，使用数据库: {db_path}")
    else:
        print("❌ 连接失败")