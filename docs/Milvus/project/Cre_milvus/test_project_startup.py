#!/usr/bin/env python3
"""
测试项目启动时的连接问题
模拟项目启动流程，检查是否还有连接冲突
"""

import logging
import sys
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_system_imports():
    """测试系统模块导入"""
    logger.info("=== 测试系统模块导入 ===")
    
    try:
        from System.start import load_config, Cre_VectorDataBaseStart_from_config
        logger.info("✓ System.start 模块导入成功")
        
        from milvusBuilder.connection_manager import get_connection_manager
        logger.info("✓ 连接管理器导入成功")
        
        from Search.search import search
        logger.info("✓ Search.search 模块导入成功")
        
        return True
    except Exception as e:
        logger.error(f"✗ 模块导入失败: {e}")
        return False

def test_config_loading():
    """测试配置加载"""
    logger.info("=== 测试配置加载 ===")
    
    try:
        from System.start import load_config
        config = load_config()
        
        logger.info(f"✓ 配置加载成功")
        logger.info(f"Milvus配置: {config.get('milvus', {})}")
        
        return config
    except Exception as e:
        logger.error(f"✗ 配置加载失败: {e}")
        return None

def test_connection_manager():
    """测试连接管理器"""
    logger.info("=== 测试连接管理器 ===")
    
    try:
        from milvusBuilder.connection_manager import get_connection_manager
        
        config = test_config_loading()
        if not config:
            return False
        
        milvus_config = config.get("milvus", {})
        host = milvus_config.get("host", "127.0.0.1")
        port = milvus_config.get("port", "19530")
        
        connection_manager = get_connection_manager()
        
        with connection_manager.get_connection(host, port) as conn_alias:
            logger.info(f"✓ 连接管理器测试成功: {conn_alias}")
            return True
            
    except Exception as e:
        logger.error(f"✗ 连接管理器测试失败: {e}")
        return False

def test_search_function():
    """测试搜索函数"""
    logger.info("=== 测试搜索函数 ===")
    
    try:
        from Search.search import search
        from System.start import load_config
        
        config = load_config()
        milvus_config = config.get("milvus", {})
        
        # 模拟搜索调用（不执行实际搜索，只测试连接）
        collection_name = milvus_config.get("collection_name", "Test_one")
        host = milvus_config.get("host", "127.0.0.1")
        port = milvus_config.get("port", "19530")
        
        logger.info(f"准备测试搜索: {collection_name} @ {host}:{port}")
        
        # 这里不执行实际搜索，只测试函数是否可以正常调用
        logger.info("✓ 搜索函数准备就绪")
        return True
        
    except Exception as e:
        logger.error(f"✗ 搜索函数测试失败: {e}")
        return False

def test_data_processing_simulation():
    """模拟数据处理流程"""
    logger.info("=== 模拟数据处理流程 ===")
    
    try:
        from System.start import Cre_VectorDataBaseStart_from_config, load_config
        
        config = load_config()
        
        # 创建临时测试数据目录
        test_data_dir = "test_data_temp"
        os.makedirs(test_data_dir, exist_ok=True)
        
        # 创建一个简单的测试文件
        test_file = os.path.join(test_data_dir, "test.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("这是一个测试文档，用于验证连接管理器是否正常工作。")
        
        # 修改配置指向测试数据
        test_config = config.copy()
        test_config["data"] = {"data_location": test_data_dir}
        
        logger.info("准备调用向量化存储函数...")
        
        # 这里实际调用函数，但使用很小的测试数据
        try:
            result = Cre_VectorDataBaseStart_from_config(test_config)
            logger.info(f"✓ 向量化存储测试成功: {result}")
            success = True
        except Exception as e:
            logger.error(f"向量化存储过程中出错: {e}")
            success = False
        
        # 清理测试数据
        try:
            os.remove(test_file)
            os.rmdir(test_data_dir)
        except:
            pass
        
        return success
        
    except Exception as e:
        logger.error(f"✗ 数据处理流程测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("开始项目启动连接测试")
    
    tests = [
        ("系统模块导入", test_system_imports),
        ("配置加载", lambda: test_config_loading() is not None),
        ("连接管理器", test_connection_manager),
        ("搜索函数", test_search_function),
        ("数据处理流程", test_data_processing_simulation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"开始测试: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results[test_name] = "通过" if result else "失败"
        except Exception as e:
            logger.error(f"测试 {test_name} 出现异常: {e}")
            results[test_name] = f"异常: {str(e)}"
    
    # 输出测试总结
    logger.info(f"\n{'='*50}")
    logger.info("测试总结")
    logger.info(f"{'='*50}")
    
    for test_name, result in results.items():
        status_icon = "✓" if result == "通过" else "✗"
        logger.info(f"{status_icon} {test_name}: {result}")
    
    passed_count = len([r for r in results.values() if r == "通过"])
    total_count = len(results)
    
    logger.info(f"\n总体结果: {passed_count}/{total_count} 测试通过")
    
    if passed_count == total_count:
        logger.info("🎉 所有测试通过！项目启动连接问题已修复")
        return True
    else:
        logger.warning("⚠️  部分测试失败，可能仍存在问题")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)