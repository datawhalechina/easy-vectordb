#!/usr/bin/env python3
"""
Milvus连接测试工具
用于诊断和测试Milvus连接问题
"""

import socket
import time
import yaml
import logging
from pymilvus import connections, utility
from milvusBuilder.milvus import test_milvus_connection, milvus_connect_with_retry

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """加载配置文件"""
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"配置文件加载失败: {e}")
        return None

def test_network_connectivity(host, port, timeout=5):
    """测试网络连通性"""
    logger.info(f"测试网络连通性: {host}:{port}")
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        start_time = time.time()
        result = sock.connect_ex((host, int(port)))
        end_time = time.time()
        sock.close()
        
        if result == 0:
            logger.info(f"✓ 网络连接成功，耗时: {end_time - start_time:.2f}秒")
            return True
        else:
            logger.error(f"✗ 网络连接失败，错误代码: {result}")
            return False
    except Exception as e:
        logger.error(f"✗ 网络测试异常: {e}")
        return False

def test_milvus_service(host, port):
    """测试Milvus服务"""
    logger.info(f"测试Milvus服务: {host}:{port}")
    
    try:
        # 使用带重试的连接方法
        milvus_connect_with_retry("default", host, port, max_retries=2, timeout=10)
        
        # 获取服务信息
        collections = utility.list_collections()
        logger.info(f"✓ Milvus服务正常，现有集合: {collections}")
        
        # 清理连接
        connections.disconnect("default")
        return True
        
    except Exception as e:
        logger.error(f"✗ Milvus服务测试失败: {e}")
        return False

def comprehensive_test():
    """综合测试"""
    logger.info("开始Milvus连接综合测试")
    
    # 加载配置
    config = load_config()
    if not config:
        logger.error("无法加载配置文件，测试终止")
        return False
    
    milvus_config = config.get("milvus", {})
    host = milvus_config.get("host", "127.0.0.1")
    port = milvus_config.get("port", "19530")
    
    logger.info(f"测试目标: {host}:{port}")
    
    # 1. 网络连通性测试
    if not test_network_connectivity(host, port):
        logger.error("网络连通性测试失败，请检查:")
        logger.error("1. Milvus服务是否启动")
        logger.error("2. 防火墙设置")
        logger.error("3. 网络配置")
        return False
    
    # 2. Milvus服务测试
    if not test_milvus_service(host, port):
        logger.error("Milvus服务测试失败，请检查:")
        logger.error("1. Milvus服务状态")
        logger.error("2. 服务配置")
        logger.error("3. 资源使用情况")
        return False
    
    logger.info("✓ 所有测试通过，Milvus连接正常")
    return True

def quick_test():
    """快速测试"""
    config = load_config()
    if not config:
        return False
    
    milvus_config = config.get("milvus", {})
    host = milvus_config.get("host", "127.0.0.1")
    port = milvus_config.get("port", "19530")
    
    return test_milvus_connection(host, port, timeout=3)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # 快速测试模式
        if quick_test():
            print("✓ 快速测试通过")
            sys.exit(0)
        else:
            print("✗ 快速测试失败")
            sys.exit(1)
    else:
        # 综合测试模式
        if comprehensive_test():
            sys.exit(0)
        else:
            sys.exit(1)