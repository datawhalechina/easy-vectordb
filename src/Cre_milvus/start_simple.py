"""
统一系统启动脚本 - Cre_milvus项目的唯一启动入口
集成后端、前端启动，以及Milvus连接、向量模型、Qwen模型初始化和连接测试
"""

import logging
import time
import threading
from typing import Dict, Any
from simple_startup import SimpleServiceManager
from config_loader import load_config

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('system_startup.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class UnifiedSystemManager:
    """统一系统管理器 - 负责所有组件的初始化和启动"""
    
    def __init__(self):
        self.config = None
        self.service_manager = None
        self.initialization_status = {
            "config_loaded": False,
            "milvus_connected": False,
            "embedding_model_loaded": False,
            "qwen_model_loaded": False,
            "backend_started": False,
            "frontend_started": False,
            "connection_tested": False
        }
    
    def load_configuration(self) -> bool:
        """加载系统配置"""
        try:
            logger.info("📝 加载系统配置...")
            self.config = load_config()
            if self.config:
                logger.info("✅ 配置加载成功")
                self.initialization_status["config_loaded"] = True
                return True
            else:
                logger.error("❌ 配置加载失败")
                return False
        except Exception as e:
            logger.error(f"❌ 配置加载异常: {e}")
            return False
    
    def initialize_milvus_connection(self) -> bool:
        """初始化Milvus连接"""
        try:
            logger.info("🔗 初始化Milvus连接...")
            from simple_milvus import initialize_milvus_from_config
            
            success = initialize_milvus_from_config(self.config)
            if success:
                logger.info("✅ Milvus连接初始化成功")
                self.initialization_status["milvus_connected"] = True
                return True
            else:
                logger.warning("⚠️ Milvus连接初始化失败")
                logger.info("💡 请确保Milvus服务器正在运行 (端口19530)")
                logger.info("💡 可以使用Docker启动: docker run -p 19530:19530 milvusdb/milvus:latest")
                return False
        except Exception as e:
            logger.error(f"❌ Milvus连接初始化异常: {e}")
            logger.info("💡 系统将继续运行，但向量存储功能将不可用")
            return False
    
    def initialize_embedding_model(self) -> bool:
        """初始化向量模型"""
        try:
            logger.info("🧠 初始化向量模型...")
            from Search.embedding import SimpleEmbeddingGenerator
            
            # 尝试初始化embedding模型
            embedding_model = SimpleEmbeddingGenerator()
            test_text = "测试文本"
            test_embedding = embedding_model.get_embedding(test_text)
            
            if test_embedding and len(test_embedding) > 0:
                logger.info("✅ 向量模型初始化成功")
                self.initialization_status["embedding_model_loaded"] = True
                return True
            else:
                logger.error("❌ 向量模型初始化失败")
                return False
        except Exception as e:
            logger.error(f"❌ 向量模型初始化异常: {e}")
            logger.info("💡 系统将继续运行，但向量化功能可能受影响")
            logger.info("💡 建议检查网络连接或使用本地模型")
            return False
    
    def initialize_qwen_model(self) -> bool:
        """初始化Qwen模型（用于PPL分块）"""
        try:
            logger.info("🤖 初始化Qwen模型（PPL分块）...")
            from dataBuilder.chunking.meta_chunking import DependencyChecker
            
            dependency_checker = DependencyChecker()
            if dependency_checker.is_ppl_chunking_available():
                logger.info("✅ Qwen模型依赖检查通过")
                self.initialization_status["qwen_model_loaded"] = True
                return True
            else:
                logger.warning("⚠️ Qwen模型依赖不完整，PPL分块功能将不可用")
                logger.info("💡 系统将继续运行，使用其他分块策略（语义分块、传统分块）")
                logger.info("💡 如需PPL分块功能，请参考 PPL_SETUP_GUIDE.md 文件")
                return False
        except Exception as e:
            logger.error(f"❌ Qwen模型初始化异常: {e}")
            logger.info("💡 系统将继续运行，使用其他分块策略")
            logger.info("💡 如需PPL分块功能，请参考 PPL_SETUP_GUIDE.md 文件")
            return False
    
    def start_services(self) -> bool:
        """启动后端和前端服务"""
        try:
            # 从配置获取端口
            backend_port = self.config.get("system", {}).get("backend_port", 8509)
            frontend_port = self.config.get("system", {}).get("frontend_port", 8500)
            
            logger.info(f"🚀 启动服务 (后端: {backend_port}, 前端: {frontend_port})")
            
            self.service_manager = SimpleServiceManager(backend_port, frontend_port)
            
            # 启动后端服务
            if self.service_manager.start_backend():
                logger.info("✅ 后端服务启动成功")
                self.initialization_status["backend_started"] = True
            else:
                logger.error("❌ 后端服务启动失败")
                return False
            
            # 启动前端服务
            if self.service_manager.start_frontend():
                logger.info("✅ 前端服务启动成功")
                self.initialization_status["frontend_started"] = True
            else:
                logger.error("❌ 前端服务启动失败")
                return False
            
            return True
        except Exception as e:
            logger.error(f"❌ 服务启动异常: {e}")
            return False
    
    def test_connections(self) -> bool:
        """测试前后端连接"""
        try:
            logger.info("🔍 测试前后端连接...")
            
            if not self.service_manager:
                logger.error("❌ 服务管理器未初始化")
                return False
            
            # 等待服务完全启动
            time.sleep(3)
            
            health_status = self.service_manager.health_check()
            
            backend_healthy = health_status.get("backend", {}).get("healthy", False)
            frontend_healthy = health_status.get("frontend", {}).get("healthy", False)
            
            if backend_healthy and frontend_healthy:
                logger.info("✅ 前后端连接测试成功")
                self.initialization_status["connection_tested"] = True
                return True
            else:
                logger.error("❌ 前后端连接测试失败")
                logger.error(f"后端状态: {'健康' if backend_healthy else '异常'}")
                logger.error(f"前端状态: {'健康' if frontend_healthy else '异常'}")
                return False
        except Exception as e:
            logger.error(f"❌ 连接测试异常: {e}")
            return False
    
    def print_startup_summary(self):
        """打印启动总结"""
        logger.info("=" * 60)
        logger.info("🎉 系统启动完成！启动状态总结:")
        logger.info("=" * 60)
        
        for component, status in self.initialization_status.items():
            status_icon = "✅" if status else "❌"
            component_name = {
                "config_loaded": "配置加载",
                "milvus_connected": "Milvus连接",
                "embedding_model_loaded": "向量模型",
                "qwen_model_loaded": "Qwen模型",
                "backend_started": "后端服务",
                "frontend_started": "前端服务",
                "connection_tested": "连接测试"
            }.get(component, component)
            
            logger.info(f"{status_icon} {component_name}: {'成功' if status else '失败'}")
        
        if self.service_manager:
            backend_port = self.service_manager.backend_port
            frontend_port = self.service_manager.frontend_port
            logger.info("=" * 60)
            logger.info("🌐 访问地址:")
            logger.info(f"📊 后端API: http://localhost:{backend_port}")
            logger.info(f"🎨 前端界面: http://localhost:{frontend_port}")
            logger.info(f"📚 API文档: http://localhost:{backend_port}/docs")
        
        logger.info("=" * 60)
    
    def start_system(self) -> bool:
        """启动整个系统"""
        logger.info("🚀 Cre_milvus 统一系统启动")
        logger.info("=" * 60)
        
        try:
            # 1. 加载配置
            if not self.load_configuration():
                return False
            
            # 2. 初始化Milvus连接
            self.initialize_milvus_connection()
            
            # 3. 初始化向量模型
            self.initialize_embedding_model()
            
            # 4. 初始化Qwen模型
            self.initialize_qwen_model()
            
            # 5. 启动服务
            if not self.start_services():
                return False
            
            # 6. 测试连接
            self.test_connections()
            
            # 7. 打印启动总结
            self.print_startup_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 系统启动失败: {e}")
            import traceback
            logger.debug(f"详细错误: {traceback.format_exc()}")
            return False
    
    def stop_system(self):
        """停止系统"""
        if self.service_manager:
            self.service_manager.stop_all()

def main():
    """主函数 - 统一系统启动入口"""
    system_manager = UnifiedSystemManager()
    
    try:
        # 启动系统
        success = system_manager.start_system()
        
        if success:
            # 启动进程监控
            if system_manager.service_manager:
                monitor_thread = threading.Thread(
                    target=system_manager.service_manager.monitor_processes,
                    daemon=True,
                    name="ProcessMonitor"
                )
                monitor_thread.start()
            
            # 等待用户中断
            logger.info("按 Ctrl+C 停止系统")
            while True:
                time.sleep(1)
        else:
            logger.error("❌ 系统启动失败")
            
    except KeyboardInterrupt:
        logger.info("\n🛑 收到停止信号...")
    except Exception as e:
        logger.error(f"❌ 系统运行异常: {e}")
        import traceback
        logger.debug(f"详细错误: {traceback.format_exc()}")
    finally:
        system_manager.stop_system()
        logger.info("✅ 系统已停止")

if __name__ == "__main__":
    main()