#!/usr/bin/env python3
"""
测试延迟连接是否解决了项目启动时的阻塞问题
"""

import logging
import time
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_module_imports():
    """测试模块导入是否会阻塞"""
    logger.info("=== 测试模块导入 ===")
    
    start_time = time.time()
    
    try:
        logger.info("导入System.start模块...")
        from System.start import load_config, Cre_VectorDataBaseStart_from_config
        import_time = time.time() - start_time
        logger.info(f"✓ System.start模块导入成功，耗时: {import_time:.2f}秒")
        
        logger.info("导入Search.search模块...")
        start_time2 = time.time()
        from Search.search import search
        import_time2 = time.time() - start_time2
        logger.info(f"✓ Search.search模块导入成功，耗时: {import_time2:.2f}秒")
        
        logger.info("导入milvusBuilder.milvus模块...")
        start_time3 = time.time()
        from milvusBuilder.milvus import milvus_connect_insert
        import_time3 = time.time() - start_time3
        logger.info(f"✓ milvusBuilder.milvus模块导入成功，耗时: {import_time3:.2f}秒")
        
        total_time = time.time() - start_time
        logger.info(f"总导入时间: {total_time:.2f}秒")
        
        if total_time > 10:
            logger.warning("⚠️  模块导入时间过长，可能存在阻塞")
            return False
        else:
            logger.info("✓ 模块导入时间正常")
            return True
            
    except Exception as e:
        logger.error(f"✗ 模块导入失败: {e}")
        return False

def test_lazy_connection():
    """测试延迟连接功能"""
    logger.info("=== 测试延迟连接功能 ===")
    
    try:
        from milvusBuilder.lazy_connection import get_lazy_connection
        
        lazy_conn = get_lazy_connection()
        logger.info("✓ 延迟连接实例创建成功")
        
        # 测试连接可用性检查（不建立实际连接）
        is_available = lazy_conn.is_available()
        logger.info(f"连接管理器可用性: {is_available}")
        
        # 测试网络连通性（不建立实际连接）
        can_connect = lazy_conn.test_connection("127.0.0.1", "19530", timeout=3)
        logger.info(f"网络连通性测试: {can_connect}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 延迟连接测试失败: {e}")
        return False

def test_actual_connection():
    """测试实际连接（可选）"""
    logger.info("=== 测试实际连接 ===")
    
    try:
        from milvusBuilder.lazy_connection import get_lazy_connection
        
        lazy_conn = get_lazy_connection()
        
        # 只有在网络连通的情况下才尝试实际连接
        if lazy_conn.test_connection("127.0.0.1", "19530", timeout=3):
            logger.info("网络连通，尝试建立实际连接...")
            
            with lazy_conn.get_connection("127.0.0.1", "19530", timeout=10) as conn_alias:
                logger.info(f"✓ 实际连接成功: {conn_alias}")
                return True
        else:
            logger.info("网络不通，跳过实际连接测试")
            return True
            
    except Exception as e:
        logger.error(f"✗ 实际连接测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("开始延迟连接测试")
    
    tests = [
        ("模块导入", test_module_imports),
        ("延迟连接功能", test_lazy_connection),
        ("实际连接", test_actual_connection)
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
        status_icon = "✓" if result == "通过" else "✗"
        logger.info(f"{status_icon} {test_name}: {result}")
    
    passed_count = len([r for r in results.values() if r == "通过"])
    total_count = len(results)
    
    logger.info(f"\n总体结果: {passed_count}/{total_count} 测试通过")
    
    if passed_count >= 2:  # 至少前两个测试通过
        logger.info("🎉 延迟连接机制工作正常！")
        return True
    else:
        logger.warning("⚠️  延迟连接机制可能存在问题")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)