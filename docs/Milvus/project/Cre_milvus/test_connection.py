#!/usr/bin/env python3
"""
Milvus连接测试脚本

用于诊断和测试Milvus连接问题
"""

import sys
import time
import logging
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_milvus_connection(host="127.0.0.1", port="19530", timeout=10):
    """
    测试Milvus连接
    
    参数:
        host: Milvus主机地址
        port: Milvus端口
        timeout: 连接超时时间（秒）
    
    返回:
        bool: 连接是否成功
    """
    try:
        logger.info(f"正在测试Milvus连接: {host}:{port}")
        
        # 先断开可能存在的连接
        try:
            connections.disconnect("default")
        except:
            pass
        
        # 尝试连接
        start_time = time.time()
        connections.connect(
            alias="default",
            host=host,
            port=int(port),
            timeout=timeout
        )
        
        connection_time = time.time() - start_time
        logger.info(f"✅ 连接成功！耗时: {connection_time:.2f}秒")
        
        # 测试基本操作
        logger.info("正在测试基本操作...")
        
        # 获取服务器版本
        try:
            version = utility.get_server_version()
            logger.info(f"✅ Milvus版本: {version}")
        except Exception as e:
            logger.warning(f"⚠️ 获取版本失败: {e}")
        
        # 列出集合
        try:
            collections = utility.list_collections()
            logger.info(f"✅ 现有集合: {collections}")
        except Exception as e:
            logger.warning(f"⚠️ 列出集合失败: {e}")
        
        # 测试创建临时集合
        test_collection_name = "test_connection_temp"
        try:
            # 如果测试集合已存在，先删除
            if utility.has_collection(test_collection_name):
                utility.drop_collection(test_collection_name)
                logger.info(f"删除已存在的测试集合: {test_collection_name}")
            
            # 创建测试集合
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128)
            ]
            schema = CollectionSchema(fields, "测试集合")
            collection = Collection(name=test_collection_name, schema=schema)
            
            logger.info(f"✅ 成功创建测试集合: {test_collection_name}")
            
            # 清理测试集合
            utility.drop_collection(test_collection_name)
            logger.info(f"✅ 成功删除测试集合: {test_collection_name}")
            
        except Exception as e:
            logger.error(f"❌ 集合操作测试失败: {e}")
        
        # 断开连接
        connections.disconnect("test_connection")
        logger.info("✅ 连接测试完成，已断开连接")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 连接失败: {e}")
        logger.error(f"错误类型: {type(e).__name__}")
        
        # 提供诊断建议
        if "timeout" in str(e).lower():
            logger.error("💡 诊断建议: 连接超时，请检查:")
            logger.error("   1. Milvus服务是否正在运行")
            logger.error("   2. 网络连接是否正常")
            logger.error("   3. 防火墙设置是否阻止连接")
        elif "connection refused" in str(e).lower():
            logger.error("💡 诊断建议: 连接被拒绝，请检查:")
            logger.error("   1. Milvus服务是否启动")
            logger.error("   2. 端口号是否正确")
            logger.error("   3. 服务是否监听在指定地址上")
        elif "name resolution" in str(e).lower():
            logger.error("💡 诊断建议: 域名解析失败，请检查:")
            logger.error("   1. 主机地址是否正确")
            logger.error("   2. DNS设置是否正常")
        
        return False

def test_with_different_methods(host="http://127.0.0.1", port="19530"):
    """
    使用不同的连接方法进行测试
    """
    logger.info("=" * 60)
    logger.info("开始使用不同方法测试Milvus连接")
    logger.info("=" * 60)
    
    # 方法1: 直接连接
    logger.info("方法1: 使用connections.connect()直接连接")
    success1 = test_milvus_connection(host, port)
    
    # 方法2: 使用add_connection + connect
    logger.info("\n方法2: 使用add_connection + connect")
    try:
        connections.add_connection(test_method2={"host": host, "port": port})
        connections.connect(alias="test_method2")
        logger.info("✅ 方法2连接成功")
        connections.disconnect("test_method2")
        success2 = True
    except Exception as e:
        logger.error(f"❌ 方法2连接失败: {e}")
        success2 = False
    
    # 总结
    logger.info("\n" + "=" * 60)
    logger.info("连接测试总结:")
    logger.info(f"方法1 (直接连接): {'✅ 成功' if success1 else '❌ 失败'}")
    logger.info(f"方法2 (add_connection): {'✅ 成功' if success2 else '❌ 失败'}")
    
    if success1 or success2:
        logger.info("🎉 至少有一种方法可以连接到Milvus")
        if success1 and not success2:
            logger.info("💡 建议: 使用直接连接方法，避免使用add_connection")
    else:
        logger.error("❌ 所有连接方法都失败了")
        logger.error("💡 请检查Milvus服务状态和网络配置")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Milvus连接测试工具")
    parser.add_argument("--host", default="127.0.0.1", help="Milvus主机地址")
    parser.add_argument("--port", default="19530", help="Milvus端口")
    parser.add_argument("--timeout", type=int, default=10, help="连接超时时间（秒）")
    
    args = parser.parse_args()
    
    logger.info("🔍 Milvus连接测试工具")
    logger.info(f"目标地址: {args.host}:{args.port}")
    logger.info(f"超时时间: {args.timeout}秒")
    
    # 执行测试
    test_with_different_methods(args.host, args.port)

if __name__ == "__main__":
    main()