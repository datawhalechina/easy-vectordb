#!/usr/bin/env python3
"""
综合集成测试脚本

测试向量化存储、分块策略、聚类和测试功能的完整集成
"""

import sys
import os
import logging
import subprocess
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_test_script(script_name):
    """运行测试脚本"""
    try:
        result = subprocess.run([
            sys.executable, script_name
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def test_backend_api_integration():
    """测试后端API集成"""
    logger.info("测试后端API集成...")
    
    try:
        from backend_api import app
        logger.info("✅ 后端API模块导入成功")
        
        # 测试关键端点是否定义
        routes = [route.path for route in app.routes]
        expected_routes = [
            "/upload", "/search", "/visualization", 
            "/chunking/process", "/system/status"
        ]
        
        missing_routes = [route for route in expected_routes if route not in routes]
        if missing_routes:
            logger.warning(f"⚠️ 缺少路由: {missing_routes}")
        else:
            logger.info("✅ 所有关键路由都存在")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 后端API集成测试失败: {e}")
        return False

def test_testing_module_integration():
    """测试测试模块集成"""
    logger.info("测试测试模块集成...")
    
    try:
        from testing import MilvusLoadTest, PerformanceMonitor, TestDataGenerator
        logger.info("✅ 测试模块导入成功")
        
        # 测试性能监控器
        monitor = PerformanceMonitor()
        logger.info("✅ 性能监控器初始化成功")
        
        # 测试负载测试器
        load_test = MilvusLoadTest({
            "host": "localhost",
            "port": "19530",
            "collection_name": "test_collection"
        })
        logger.info("✅ 负载测试器初始化成功")
        
        # 测试数据生成器
        data_gen = TestDataGenerator()
        logger.info("✅ 测试数据生成器初始化成功")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试模块集成测试失败: {e}")
        return False

def test_config_integration():
    """测试配置集成"""
    logger.info("测试配置集成...")
    
    try:
        import yaml
        
        # 检查配置文件是否存在
        config_path = Path("config.yaml")
        if not config_path.exists():
            logger.error("❌ 配置文件不存在")
            return False
        
        # 加载配置
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        # 检查关键配置项
        required_sections = ["milvus", "chunking", "search", "system"]
        missing_sections = [section for section in required_sections if section not in config]
        
        if missing_sections:
            logger.error(f"❌ 配置文件缺少必要部分: {missing_sections}")
            return False
        
        logger.info("✅ 配置文件完整")
        
        # 检查分块策略配置
        chunking_config = config.get("chunking", {})
        strategy = chunking_config.get("strategy", "traditional")
        logger.info(f"✅ 当前分块策略: {strategy}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 配置集成测试失败: {e}")
        return False

def test_embedding_integration():
    """测试嵌入模型集成"""
    logger.info("测试嵌入模型集成...")
    
    try:
        from Search.embedding import embedder
        
        # 检查嵌入模型状态
        status = embedder.check_status()
        
        if status.get("model_loaded") and status.get("tokenizer_loaded"):
            logger.info(f"✅ 嵌入模型已加载: {status.get('model_name')}")
            logger.info(f"✅ 设备: {status.get('device')}")
            
            # 测试嵌入生成
            test_text = "这是一个测试文本"
            embedding = embedder.get_embedding(test_text)
            
            if embedding and len(embedding) > 0:
                logger.info(f"✅ 嵌入生成成功，维度: {len(embedding)}")
                return True
            else:
                logger.error("❌ 嵌入生成失败")
                return False
        else:
            logger.error("❌ 嵌入模型未正确加载")
            return False
            
    except Exception as e:
        logger.error(f"❌ 嵌入模型集成测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("=" * 80)
    logger.info("🧪 Cre_milvus 综合集成测试")
    logger.info("=" * 80)
    
    # 运行子测试脚本
    sub_tests = [
        ("分块策略集成", "test_chunking_integration.py"),
        ("聚类可视化集成", "test_clustering_integration.py")
    ]
    
    sub_test_results = {}
    
    for test_name, script_name in sub_tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        
        if Path(script_name).exists():
            success, stdout, stderr = run_test_script(script_name)
            sub_test_results[test_name] = success
            
            if success:
                logger.info(f"✅ {test_name}: 通过")
            else:
                logger.error(f"❌ {test_name}: 失败")
                if stderr:
                    logger.error(f"错误信息: {stderr}")
        else:
            logger.warning(f"⚠️ {test_name}: 测试脚本不存在")
            sub_test_results[test_name] = False
    
    # 运行直接集成测试
    logger.info(f"\n{'='*20} 直接集成测试 {'='*20}")
    
    direct_tests = [
        ("后端API集成", test_backend_api_integration),
        ("测试模块集成", test_testing_module_integration),
        ("配置集成", test_config_integration),
        ("嵌入模型集成", test_embedding_integration)
    ]
    
    direct_test_results = {}
    
    for test_name, test_func in direct_tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            direct_test_results[test_name] = result
        except Exception as e:
            logger.error(f"❌ {test_name}: 异常 - {e}")
            direct_test_results[test_name] = False
    
    # 汇总结果
    logger.info("\n" + "=" * 80)
    logger.info("📊 综合测试结果总结")
    logger.info("=" * 80)
    
    # 子测试结果
    logger.info("\n🔧 子测试结果:")
    for test_name, result in sub_test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"  {test_name}: {status}")
    
    # 直接测试结果
    logger.info("\n🔧 直接测试结果:")
    for test_name, result in direct_test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"  {test_name}: {status}")
    
    # 总体评估
    all_sub_tests_passed = all(sub_test_results.values())
    all_direct_tests_passed = all(direct_test_results.values())
    all_tests_passed = all_sub_tests_passed and all_direct_tests_passed
    
    logger.info("\n" + "=" * 80)
    if all_tests_passed:
        logger.info("🎉 所有集成测试通过！系统集成状态良好。")
        logger.info("✅ 向量化存储、分块策略、聚类和测试功能都已正确集成")
    else:
        logger.error("❌ 部分集成测试失败，需要修复以下问题:")
        
        if not all_sub_tests_passed:
            failed_sub_tests = [name for name, result in sub_test_results.items() if not result]
            logger.error(f"  子测试失败: {', '.join(failed_sub_tests)}")
        
        if not all_direct_tests_passed:
            failed_direct_tests = [name for name, result in direct_test_results.items() if not result]
            logger.error(f"  直接测试失败: {', '.join(failed_direct_tests)}")
    
    logger.info("=" * 80)
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)