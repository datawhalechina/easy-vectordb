#!/usr/bin/env python3
"""
应用启动脚本
项目启动时运行，建立所有必要的连接
"""

import logging
import sys
import os
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def startup_application():
    """启动应用程序"""
    logger.info("=" * 80)
    logger.info("🚀 应用程序启动中...")
    logger.info("=" * 80)
    
    try:
        # 1. 初始化所有连接
        logger.info("1. 初始化连接...")
        from System.connection_initializer import startup_initialize
        
        start_time = time.time()
        success = startup_initialize()
        end_time = time.time()
        
        if not success:
            logger.error("❌ 连接初始化失败，应用启动失败")
            return False
        
        logger.info(f"✅ 连接初始化成功，耗时: {end_time - start_time:.2f}秒")
        
        # 2. 验证连接状态
        logger.info("2. 验证连接状态...")
        from System.new_start import get_connection_status
        
        status = get_connection_status()
        logger.info(f"连接状态: {status}")
        
        if not status.get("overall_ready", False):
            logger.error("❌ 连接状态验证失败")
            return False
        
        logger.info("✅ 连接状态验证通过")
        
        # 3. 应用启动完成
        logger.info("=" * 80)
        logger.info("🎉 应用程序启动成功!")
        logger.info("现在可以直接使用向量数据库功能，无需重新连接")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 应用启动异常: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return False

def test_fast_build():
    """测试快速构建功能"""
    logger.info("=" * 60)
    logger.info("测试快速向量数据库构建")
    logger.info("=" * 60)
    
    try:
        # 加载配置
        import yaml
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        # 使用新的快速构建功能
        from System.new_start import fast_vector_database_build_from_config
        
        logger.info("开始快速构建测试...")
        start_time = time.time()
        result = fast_vector_database_build_from_config(config)
        end_time = time.time()
        
        logger.info(f"构建结果: {result}")
        logger.info(f"总耗时: {end_time - start_time:.2f}秒")
        
        if result.get("status") == "success":
            logger.info("🎉 快速构建测试成功!")
            return True
        else:
            logger.error("❌ 快速构建测试失败")
            return False
            
    except Exception as e:
        logger.error(f"❌ 快速构建测试异常: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return False

def main():
    """主函数"""
    logger.info("开始应用启动流程")
    
    try:
        # 1. 启动应用
        if not startup_application():
            logger.error("应用启动失败")
            return False
        
        # 2. 询问是否测试
        try:
            choice = input("\n是否测试快速构建功能? (y/n): ").lower().strip()
            if choice == 'y':
                test_fast_build()
            else:
                logger.info("跳过快速构建测试")
        except KeyboardInterrupt:
            logger.info("\n用户中断")
        
        logger.info("\n应用启动流程完成")
        return True
        
    except Exception as e:
        logger.error(f"应用启动流程异常: {e}")
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            logger.info("✅ 应用启动成功，可以开始使用")
            sys.exit(0)
        else:
            logger.error("❌ 应用启动失败")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n用户中断启动")
        sys.exit(1)
    except Exception as e:
        logger.error(f"启动异常: {e}")
        sys.exit(1)