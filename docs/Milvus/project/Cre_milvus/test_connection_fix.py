#!/usr/bin/env python3
"""
测试连接修复效果
验证新的连接管理器是否解决了线程阻塞问题
"""

import logging
import time
import threading
import yaml
from milvusBuilder.connection_manager import get_connection_manager
from milvusBuilder.milvus import milvus_connect_insert

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config():
    """加载配置文件"""
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"配置文件加载失败: {e}")
        return {
            "milvus": {
                "host": "127.0.0.1",
                "port": "19530"
            }
        }

def test_connection_manager():
    """测试连接管理器基本功能"""
    logger.info("=== 测试连接管理器基本功能 ===")
    
    config = load_config()
    milvus_config = config.get("milvus", {})
    host = milvus_config.get("host", "127.0.0.1")
    port = milvus_config.get("port", "19530")
    
    connection_manager = get_connection_manager()
    
    try:
        # 测试连接创建和释放
        logger.info("测试连接创建...")
        with connection_manager.get_connection(host, port) as conn_alias:
            logger.info(f"连接创建成功: {conn_alias}")
            
            # 获取连接状态
            status = connection_manager.get_connection_status()
            logger.info(f"连接状态: {status}")
            
        logger.info("连接已自动释放")
        
        # 验证连接已清理
        final_status = connection_manager.get_connection_status()
        logger.info(f"最终连接状态: {final_status}")
        
        return True
        
    except Exception as e:
        logger.error(f"连接管理器测试失败: {e}")
        return False

def test_concurrent_connections():
    """测试并发连接处理"""
    logger.info("=== 测试并发连接处理 ===")
    
    config = load_config()
    milvus_config = config.get("milvus", {})
    host = milvus_config.get("host", "127.0.0.1")
    port = milvus_config.get("port", "19530")
    
    connection_manager = get_connection_manager()
    results = []
    
    def worker_thread(thread_id):
        """工作线程函数"""
        try:
            logger.info(f"线程 {thread_id} 开始连接测试")
            with connection_manager.get_connection(host, port) as conn_alias:
                logger.info(f"线程 {thread_id} 连接成功: {conn_alias}")
                time.sleep(2)  # 模拟工作
                results.append(f"thread_{thread_id}_success")
            logger.info(f"线程 {thread_id} 完成")
        except Exception as e:
            logger.error(f"线程 {thread_id} 失败: {e}")
            results.append(f"thread_{thread_id}_failed")
    
    # 创建多个并发线程
    threads = []
    for i in range(3):
        thread = threading.Thread(target=worker_thread, args=(i,))
        threads.append(thread)
        thread.start()
    
    # 等待所有线程完成
    for thread in threads:
        thread.join(timeout=30)  # 30秒超时
    
    logger.info(f"并发测试结果: {results}")
    success_count = len([r for r in results if "success" in r])
    logger.info(f"成功: {success_count}/3")
    
    return success_count == 3

def test_insert_function():
    """测试插入函数"""
    logger.info("=== 测试插入函数 ===")
    
    config = load_config()
    milvus_config = config.get("milvus", {})
    host = milvus_config.get("host", "127.0.0.1")
    port = milvus_config.get("port", "19530")
    
    # 准备测试数据
    test_data = [
        {
            "id": 1,
            "content": "测试文档1",
            "embedding": [0.1] * 256,  # 256维向量
            "url": "http://test1.com"
        },
        {
            "id": 2,
            "content": "测试文档2", 
            "embedding": [0.2] * 256,
            "url": "http://test2.com"
        }
    ]
    
    # 索引参数
    index_param = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    
    try:
        result = milvus_connect_insert(
            CollectionName="test_connection_fix",
            IndexParam=index_param,
            ReplicaNum=1,
            dataList=test_data,
            url_split=True,
            Milvus_host=host,
            Milvus_port=port,
            insert_mode="覆盖（删除原有数据）"
        )
        
        logger.info(f"插入测试结果: {result}")
        return result.get("status") == "success"
        
    except Exception as e:
        logger.error(f"插入测试失败: {e}")
        return False

def test_stress_connections():
    """压力测试：快速创建和释放连接"""
    logger.info("=== 压力测试：快速连接创建释放 ===")
    
    config = load_config()
    milvus_config = config.get("milvus", {})
    host = milvus_config.get("host", "127.0.0.1")
    port = milvus_config.get("port", "19530")
    
    connection_manager = get_connection_manager()
    success_count = 0
    
    for i in range(10):
        try:
            logger.info(f"压力测试 {i+1}/10")
            with connection_manager.get_connection(host, port, timeout=5) as conn_alias:
                logger.info(f"快速连接 {i+1} 成功: {conn_alias}")
                success_count += 1
        except Exception as e:
            logger.error(f"快速连接 {i+1} 失败: {e}")
    
    logger.info(f"压力测试结果: {success_count}/10 成功")
    return success_count >= 8  # 允许少量失败

def main():
    """主测试函数"""
    logger.info("开始连接修复效果测试")
    
    tests = [
        ("连接管理器基本功能", test_connection_manager),
        ("并发连接处理", test_concurrent_connections),
        ("插入函数测试", test_insert_function),
        ("压力测试", test_stress_connections)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"开始测试: {test_name}")
        logger.info(f"{'='*50}")
        
        start_time = time.time()
        try:
            result = test_func()
            results[test_name] = "通过" if result else "失败"
        except Exception as e:
            logger.error(f"测试 {test_name} 出现异常: {e}")
            results[test_name] = f"异常: {str(e)}"
        
        end_time = time.time()
        logger.info(f"测试 {test_name} 完成，耗时: {end_time - start_time:.2f}秒")
    
    # 输出测试总结
    logger.info(f"\n{'='*50}")
    logger.info("测试总结")
    logger.info(f"{'='*50}")
    
    for test_name, result in results.items():
        logger.info(f"{test_name}: {result}")
    
    passed_count = len([r for r in results.values() if r == "通过"])
    total_count = len(results)
    
    logger.info(f"\n总体结果: {passed_count}/{total_count} 测试通过")
    
    if passed_count == total_count:
        logger.info("🎉 所有测试通过！连接问题已修复")
        return True
    else:
        logger.warning("⚠️  部分测试失败，可能仍存在问题")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)